"""Проверки повторной индексации на Qdrant в памяти, без вызовов внешних моделей."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from qdrant_client import QdrantClient, models

from backend import upload
from backend.catalog import CatalogValidationError, book_id, book_key
from backend.storage import (
    COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    VECTOR_SIZE,
    prepare_collection,
)


def example_book(**changes):
    return {
        "title": "Война и мир", "author": "Лев Толстой", "year": 1869,
        "category": "Роман", "description": "История семей в эпоху Наполеона.",
        **changes,
    }


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.client = QdrantClient(":memory:")
        self.addCleanup(self.client.close)
        prepare_collection(self.client)
        self.dense = [1.0] + [0.0] * (VECTOR_SIZE - 1)
        self.client_factory = self.start_patch("get_qdrant_client", return_value=self.client)
        self.enrich = self.start_patch("enrich_description", return_value="Обогащённое описание")
        self.embed = self.start_patch("get_dense_vector", return_value=self.dense)
        logging_patch = patch.object(upload.logger, "exception")
        logging_patch.start()
        self.addCleanup(logging_patch.stop)

    def start_patch(self, name, **kwargs):
        patcher = patch.object(upload, name, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def reset_models(self):
        for mock in (self.enrich, self.embed):
            mock.reset_mock()

    def import_data(self, data):
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as file:
            path = Path(file.name)
            self.addCleanup(path.unlink, missing_ok=True)
            json.dump(data, file, ensure_ascii=False)
        result = upload.add_books_from_json(str(path))
        self.assertEqual(result["total"], sum(result[key] for key in (
            "added", "updated", "repaired", "skipped", "failed",
        )))
        return result

    def read(self, book=None, point_id=None):
        point_id = point_id if point_id is not None else book_id(book or example_book())
        return self.client.retrieve(COLLECTION_NAME, [point_id], with_vectors=True)[0]

    def overwrite(self, point, vectors=None, payload=None):
        self.client.upsert(COLLECTION_NAME, [models.PointStruct(
            id=point.id,
            vector=point.vector if vectors is None else vectors,
            payload=point.payload if payload is None else payload,
        )])

    def seed(self):
        self.assertEqual(self.import_data([example_book()])["added"], 1)
        self.reset_models()
        return self.read()

    def test_entire_file_is_validated_before_database_or_models_are_called(self):
        for data in (
            [example_book(), {"title": "Неполная книга"}],
            [example_book(), example_book(year="1869")],
        ):
            with self.assertRaises(CatalogValidationError):
                self.import_data(data)
        for mock in (self.client_factory, self.enrich, self.embed):
            mock.assert_not_called()

    def test_duplicates_reordering_and_case_changes_do_not_call_models_again(self):
        first, second = example_book(), example_book(title="Другая книга")
        self.assertEqual(self.import_data([first, second, first])["added"], 2)
        self.reset_models()
        report = self.import_data({"books": [
            second, example_book(title="  ВОЙНА   И МИР ", author="ЛЕВ ТОЛСТОЙ"),
        ]})
        self.assertEqual(report["skipped"], 2)
        self.assertEqual(self.client.count(COLLECTION_NAME).count, 2)
        for mock in (self.enrich, self.embed):
            mock.assert_not_called()

    def test_gaps_in_old_numeric_ids_do_not_cause_overwrite(self):
        old_books = [example_book(title="Старая A"), example_book(title="Старая B")]
        self.client.upsert(COLLECTION_NAME, [
            models.PointStruct(id=i, vector={DENSE_VECTOR_NAME: self.dense}, payload=book)
            for i, book in zip((0, 2), old_books)
        ])
        before = self.client.retrieve(COLLECTION_NAME, [0, 2], with_vectors=True)
        self.assertEqual(self.import_data([example_book()])["added"], 1)
        self.assertEqual(self.client.count(COLLECTION_NAME).count, 3)
        self.assertEqual(self.client.retrieve(COLLECTION_NAME, [0, 2], with_vectors=True), before)

    def test_legacy_book_is_reindexed_once_and_keeps_id_zero_and_extra_payload(self):
        old = example_book(title=" ВОЙНА  И МИР ", author="ЛЕВ ТОЛСТОЙ", note="сохранить")
        self.client.upsert(COLLECTION_NAME, [models.PointStruct(
            id=0, vector={DENSE_VECTOR_NAME: self.dense}, payload=old,
        )])
        self.assertEqual(self.import_data([example_book()])["updated"], 1)
        self.assertEqual(self.read(point_id=0).payload["note"], "сохранить")
        self.assertEqual(self.client.count(COLLECTION_NAME).count, 1)
        self.reset_models()
        self.assertEqual(self.import_data([example_book()])["skipped"], 1)
        self.enrich.assert_not_called()
        self.embed.assert_not_called()

    def test_new_records_only_store_dense_and_are_skipped_on_repeat(self):
        point = self.seed()
        self.assertEqual(set(point.vector), {DENSE_VECTOR_NAME})
        self.assertNotIn("sparse_version", point.payload["indexing"])
        self.assertEqual(self.import_data([example_book()])["skipped"], 1)
        self.enrich.assert_not_called()
        self.embed.assert_not_called()

    def test_legacy_sparse_presence_or_version_does_not_trigger_reindex(self):
        point = self.seed()
        self.client.delete_collection(COLLECTION_NAME)
        self.client.create_collection(
            COLLECTION_NAME,
            vectors_config={DENSE_VECTOR_NAME: models.VectorParams(
                size=VECTOR_SIZE, distance=models.Distance.COSINE)},
            sparse_vectors_config={"sparse-vector": models.SparseVectorParams(
                modifier=models.Modifier.IDF)},
        )
        payload = {**point.payload, "indexing": {
            **point.payload["indexing"], "sparse_version": "old-unused-version"}}
        for sparse in (None, models.SparseVector(indices=[7], values=[1.0])):
            vectors = {DENSE_VECTOR_NAME: point.vector[DENSE_VECTOR_NAME]}
            if sparse is not None:
                vectors["sparse-vector"] = sparse
            self.overwrite(point, vectors=vectors, payload=payload)
            before = self.read()
            with patch.object(self.client, "upsert", wraps=self.client.upsert) as save:
                self.assertEqual(self.import_data([example_book()])["skipped"], 1)
                save.assert_not_called()
            self.assertEqual(self.read(), before)
            self.enrich.assert_not_called()
            self.embed.assert_not_called()

        changed = example_book(description="Исправленное описание")
        self.assertEqual(self.import_data([changed])["updated"], 1)
        current = self.read()
        self.assertEqual(current.id, point.id)
        self.assertEqual(current.payload["description"], changed["description"])
        self.assertEqual(set(current.vector), {DENSE_VECTOR_NAME})
        self.assertNotIn("sparse_version", current.payload["indexing"])
        self.assertEqual(self.import_data([example_book(title="Новая книга")])["added"], 1)
        self.assertEqual(self.client.count(COLLECTION_NAME).count, 2)

    def test_missing_or_zero_dense_is_repaired_without_llm(self):
        point = self.seed()
        for dense in (None, [0.0] * VECTOR_SIZE):
            vectors = {} if dense is None else {DENSE_VECTOR_NAME: dense}
            self.overwrite(point, vectors=vectors)
            self.reset_models()
            self.assertEqual(self.import_data([example_book()])["repaired"], 1)
            self.enrich.assert_not_called()
            self.embed.assert_called_once()

    def test_changed_source_updates_same_id_and_payload(self):
        point = self.seed()
        changed = example_book(description="Исправленное описание", year=1870)
        report = self.import_data([changed])
        self.assertEqual(report["updated"], 1)
        current = self.read()
        self.assertEqual(current.id, point.id)
        self.assertEqual(current.payload["description"], changed["description"])
        self.assertNotEqual(
            current.payload["indexing"]["source_hash"], point.payload["indexing"]["source_hash"],
        )
        for mock in (self.enrich, self.embed):
            mock.assert_called_once()

    def test_dense_version_recomputes_vector_without_enrichment(self):
        self.seed()
        with patch.object(upload, "DENSE_VERSION", "new-version"):
            self.assertEqual(self.import_data([example_book()])["updated"], 1)
            self.embed.assert_called_once()
            self.enrich.assert_not_called()
            self.reset_models()
            self.assertEqual(self.import_data([example_book()])["skipped"], 1)
            self.embed.assert_not_called()
            self.enrich.assert_not_called()

    def test_new_enrichment_version_rebuilds_description_and_dense(self):
        self.seed()
        with patch.object(upload, "ENRICHMENT_VERSION", "new-prompt"):
            self.assertEqual(self.import_data([example_book()])["updated"], 1)
        for mock in (self.enrich, self.embed):
            mock.assert_called_once()

    def test_partial_failure_preserves_successes_and_retry_only_processes_failed_book(self):
        books = [example_book(title=title) for title in ("A", "B", "C")]
        self.enrich.side_effect = ["Описание A", RuntimeError("API failure"), "Описание C"]
        report = self.import_data(books)
        self.assertEqual((report["added"], report["failed"], report["status"]), (2, 1, "partial"))
        self.assertEqual(report["errors"][0]["title"], "B")
        self.assertEqual(report["errors"][0]["stage"], "enrichment")
        self.assertEqual(self.client.count(COLLECTION_NAME).count, 2)
        self.enrich.side_effect = None
        self.reset_models()
        retry = self.import_data(books)
        self.assertEqual((retry["added"], retry["skipped"]), (1, 2))
        self.enrich.assert_called_once()
        self.embed.assert_called_once()

    def test_failed_update_leaves_entire_previous_record_unchanged(self):
        before = self.seed()
        self.embed.side_effect = RuntimeError("Embedding failure")
        report = self.import_data([example_book(description="Новое описание")])
        self.assertEqual(report["failed"], 1)
        self.assertEqual(report["errors"][0]["stage"], "dense")
        self.assertEqual(self.read(), before)

    def test_invalid_generated_vectors_and_empty_enrichment_are_not_saved(self):
        for mock, value, stage in (
            (self.enrich, "  ", "enrichment"),
            (self.embed, [0.0] * VECTOR_SIZE, "dense"),
            (self.embed, [1.0], "dense"),
        ):
            original = mock.return_value
            mock.return_value = value
            report = self.import_data([example_book()])
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["errors"][0]["stage"], stage)
            self.assertEqual(self.client.count(COLLECTION_NAME).count, 0)
            mock.return_value = original

    def test_write_failure_does_not_stop_other_books(self):
        books = [example_book(title=title) for title in ("A", "B", "C")]
        upsert = self.client.upsert

        def save(collection, points, **kwargs):
            if points[0].payload["title"] == "B":
                raise RuntimeError("Database write failed")
            self.assertTrue(kwargs["wait"])
            return upsert(collection, points, **kwargs)

        with patch.object(self.client, "upsert", side_effect=save):
            report = self.import_data(books)
        self.assertEqual((report["added"], report["failed"]), (2, 1))
        self.assertEqual(report["errors"][0]["stage"], "save")
        self.assertEqual(self.client.count(COLLECTION_NAME).count, 2)

    def test_lost_write_confirmation_can_be_retried_without_duplicate_or_model_calls(self):
        upsert = self.client.upsert

        def commit_then_fail(*args, **kwargs):
            upsert(*args, **kwargs)
            raise TimeoutError("Lost response")

        with patch.object(self.client, "upsert", side_effect=commit_then_fail):
            self.assertEqual(self.import_data([example_book()])["failed"], 1)
        self.reset_models()
        self.assertEqual(self.import_data([example_book()])["skipped"], 1)
        self.enrich.assert_not_called()
        self.assertEqual(self.client.count(COLLECTION_NAME).count, 1)

    def test_latest_duplicate_is_processed_and_older_record_is_removed(self):
        self.client.upsert(COLLECTION_NAME, [models.PointStruct(
            id=i, vector={DENSE_VECTOR_NAME: self.dense},
            payload={**example_book(), "created_at": created_at},
        ) for i, created_at in (
            (1, "2026-10-18T12:00:00+00:00"),
            (2, "2026-10-17T12:00:00+00:00"),
        )])
        report = self.import_data([example_book()])
        self.assertEqual(report["updated"], 1)
        self.assertEqual(report["duplicates_removed"], 1)
        self.assertEqual(report["failed"], 0)
        self.assertEqual(self.client.count(COLLECTION_NAME).count, 1)
        self.assertEqual(self.read(point_id=1).payload["created_at"], "2026-10-18T12:00:00+00:00")
        self.assertEqual(self.client.retrieve(COLLECTION_NAME, [2]), [])
        self.enrich.assert_called_once()

    def test_conflicting_point_id_is_not_overwritten(self):
        self.client.upsert(COLLECTION_NAME, [models.PointStruct(
            id=book_id(example_book()), vector={DENSE_VECTOR_NAME: self.dense},
            payload=example_book(title="Чужая книга"),
        )])
        before = self.read()
        self.assertEqual(self.import_data([example_book()])["failed"], 1)
        self.assertEqual(self.read(), before)
        self.enrich.assert_not_called()

    def test_edited_payload_is_detected_even_if_old_hash_was_left_in_place(self):
        point = self.seed()
        payload = {**point.payload, "description": "Ручное изменение"}
        self.overwrite(point, payload=payload)
        self.assertEqual(self.import_data([example_book()])["updated"], 1)
        self.enrich.assert_called_once()

    def test_edited_enriched_text_invalidates_dense(self):
        point = self.seed()
        self.overwrite(point, payload={**point.payload, "enriched_description": "Новая аннотация"})
        self.assertEqual(self.import_data([example_book()])["repaired"], 1)
        self.enrich.assert_not_called()
        self.embed.assert_called_once()

    def test_existing_books_are_read_beyond_first_page(self):
        books = [example_book(title=f"Книга {i}") for i in range(257)]
        self.client.upsert(COLLECTION_NAME, [models.PointStruct(
            id=i, vector={DENSE_VECTOR_NAME: self.dense}, payload=book,
        ) for i, book in enumerate(books)])
        self.assertEqual(
            upload.get_existing_books(),
            {book_key(book["title"], book["author"]): [i] for i, book in enumerate(books)},
        )

    def test_stale_catalog_snapshot_still_finds_existing_uuid(self):
        self.seed()
        with patch.object(upload, "get_existing_books", return_value={}):
            self.assertEqual(self.import_data([example_book()])["skipped"], 1)
        self.enrich.assert_not_called()


class VectorTests(unittest.TestCase):
    def test_dense_validation_rejects_nonfinite_boolean_and_zero_values(self):
        good = [1.0] + [0.0] * (VECTOR_SIZE - 1)
        self.assertTrue(upload.valid_dense(good))
        for vector in (
            None, [], [0.0] * VECTOR_SIZE, [[1.0]] * VECTOR_SIZE,
            [float("nan")] + good[1:], [float("inf")] + good[1:], [True] + good[1:],
        ):
            with self.subTest(vector_type=type(vector)):
                self.assertFalse(upload.valid_dense(vector))

    def test_embedding_receives_full_text_and_api_errors_propagate(self):
        client = Mock()
        client.embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(embedding=[1.0])],
        )
        text = "Описание книги. " * 1000
        with patch.object(upload, "get_embedding_client", return_value=client):
            self.assertEqual(upload.get_dense_vector(text), [1.0])
            self.assertEqual(client.embeddings.create.call_args.kwargs["input"], [text])
            self.assertNotIn("dimensions", client.embeddings.create.call_args.kwargs)
            self.assertEqual(client.embeddings.create.call_args.kwargs["extra_body"], {"input_type": "passage"})
            upload.get_dense_vector(text, input_type="query")
            self.assertEqual(client.embeddings.create.call_args.kwargs["extra_body"], {"input_type": "query"})
            with self.assertRaises(ValueError):
                upload.get_dense_vector(text, input_type="unknown")
            client.embeddings.create.side_effect = RuntimeError("API failure")
            with self.assertRaises(RuntimeError):
                upload.get_dense_vector(text)

    def test_enrichment_api_failure_is_not_silently_replaced_with_original_text(self):
        client = Mock()
        client.chat.completions.create.side_effect = RuntimeError("API failure")
        with patch.object(upload, "get_llm_client", return_value=client):
            with self.assertRaises(RuntimeError):
                upload.enrich_description(example_book())


class ApiClientTests(unittest.TestCase):
    def setUp(self):
        for factory in (upload.get_llm_client, upload.get_embedding_client):
            factory.cache_clear()
            self.addCleanup(factory.cache_clear)

    def test_llm_and_embeddings_use_separate_keys_and_endpoints(self):
        llm, embedding = Mock(), Mock()
        with patch.dict(os.environ, {
            "DEEPSEEK_API_KEY": "deepseek-test", "OPENROUTER_API_KEY": "router-test",
        }, clear=True), patch("dotenv.load_dotenv"), \
             patch("openai.OpenAI", side_effect=[llm, embedding]) as constructor:
            self.assertIs(upload.get_llm_client(), llm)
            self.assertIs(upload.get_embedding_client(), embedding)
            self.assertIs(upload.get_llm_client(), llm)
            self.assertIs(upload.get_embedding_client(), embedding)
        self.assertEqual(constructor.call_count, 2)
        llm_args, embedding_args = [call.kwargs for call in constructor.call_args_list]
        self.assertEqual(llm_args["api_key"], "deepseek-test")
        self.assertEqual(llm_args["base_url"], "https://api.deepseek.com")
        self.assertEqual(embedding_args["api_key"], "router-test")
        self.assertEqual(embedding_args["base_url"], "https://openrouter.ai/api/v1")

    def test_missing_key_never_falls_back_to_another_provider(self):
        cases = (
            (upload.get_llm_client, "DEEPSEEK_API_KEY", {"OPENROUTER_API_KEY": "router-test"}),
            (upload.get_embedding_client, "OPENROUTER_API_KEY", {"DEEPSEEK_API_KEY": "deepseek-test"}),
        )
        for factory, required, other_key in cases:
            with self.subTest(required=required), \
                 patch.dict(os.environ, {"CLOUDRU_API_KEY": "legacy-test", **other_key}, clear=True), \
                 patch("dotenv.load_dotenv"), patch("openai.OpenAI") as constructor:
                with self.assertRaisesRegex(ValueError, required):
                    factory()
                constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
