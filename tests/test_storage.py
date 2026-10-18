"""Проверки подготовки коллекции без API-ключей и вызовов моделей."""

import unittest
from unittest.mock import Mock, patch

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from backend.storage import (
    COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    VECTOR_SIZE,
    prepare_collection,
    wait_for_qdrant,
)


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.client = QdrantClient(":memory:")
        self.addCleanup(self.client.close)

    def create_existing(self, size=VECTOR_SIZE, distance=models.Distance.COSINE,
                        sparse=False, modifier=models.Modifier.IDF):
        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(size=size, distance=distance)
            },
            sparse_vectors_config=(
                {"sparse-vector": models.SparseVectorParams(modifier=modifier)}
                if sparse else None
            ),
        )

    def test_first_start_creates_expected_schema(self):
        prepare_collection(self.client)
        params = self.client.get_collection(COLLECTION_NAME).config.params
        self.assertEqual(params.vectors[DENSE_VECTOR_NAME].size, VECTOR_SIZE)
        self.assertEqual(params.vectors[DENSE_VECTOR_NAME].distance, models.Distance.COSINE)
        self.assertFalse(params.sparse_vectors)

    def test_second_start_preserves_uploaded_book_and_vectors(self):
        prepare_collection(self.client)
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=[models.PointStruct(
                id=7,
                vector={
                    DENSE_VECTOR_NAME: [1.0] + [0.0] * (VECTOR_SIZE - 1),
                },
                payload={"title": "Тестовая книга"},
            )],
        )
        before = self.client.retrieve(COLLECTION_NAME, [7], with_vectors=True)
        prepare_collection(self.client)
        self.assertEqual(self.client.count(COLLECTION_NAME).count, 1)
        self.assertEqual(
            self.client.retrieve(COLLECTION_NAME, [7], with_vectors=True), before
        )

    def test_wrong_dimension_is_rejected_without_recreating_collection(self):
        self.create_existing(size=3)
        self.client.upsert(COLLECTION_NAME, points=[models.PointStruct(
            id=1, vector={DENSE_VECTOR_NAME: [1.0, 0.0, 0.0]},
            payload={"title": "Существующая книга"},
        )])
        with self.assertRaisesRegex(RuntimeError, "размерность dense 3"):
            prepare_collection(self.client)
        self.assertEqual(self.client.count(COLLECTION_NAME).count, 1)
        self.assertEqual(
            self.client.get_collection(COLLECTION_NAME).config.params.vectors[DENSE_VECTOR_NAME].size,
            3,
        )

    def test_wrong_distance_is_rejected(self):
        self.create_existing(distance=models.Distance.DOT)
        with self.assertRaisesRegex(RuntimeError, "метрика dense"):
            prepare_collection(self.client)

    def test_legacy_sparse_schema_and_points_are_preserved_without_migration(self):
        for modifier in (None, models.Modifier.IDF):
            with self.subTest(modifier=modifier):
                if self.client.collection_exists(COLLECTION_NAME):
                    self.client.delete_collection(COLLECTION_NAME)
                self.create_existing(sparse=True, modifier=modifier)
                self.client.upsert(COLLECTION_NAME, [models.PointStruct(
                    id=1,
                    vector={DENSE_VECTOR_NAME: [1.0] + [0.0] * (VECTOR_SIZE - 1),
                            "sparse-vector": models.SparseVector(indices=[12], values=[1.0])},
                    payload={"title": "Старая книга"},
                )])
                before = self.client.retrieve(COLLECTION_NAME, [1], with_vectors=True)
                prepare_collection(self.client)
                self.assertEqual(self.client.retrieve(COLLECTION_NAME, [1], with_vectors=True), before)
                params = self.client.get_collection(COLLECTION_NAME).config.params
                self.assertIn("sparse-vector", params.sparse_vectors)

    def test_concurrent_collection_creation_is_validated(self):
        prepare_collection(self.client)
        with patch.object(self.client, "collection_exists", return_value=False), \
             patch.object(self.client, "create_collection", side_effect=UnexpectedResponse(
                 status_code=409, reason_phrase="Conflict", content=b"exists", headers={}
             )):
            prepare_collection(self.client)


class ReadinessTests(unittest.TestCase):
    @patch("backend.storage.time.sleep")
    def test_temporary_connection_failure_is_retried(self, sleep):
        client = Mock()
        client.get_collections.side_effect = [
            ResponseHandlingException(ConnectionError("not ready")), object()
        ]
        wait_for_qdrant(client, attempts=2, delay=0.1)
        self.assertEqual(client.get_collections.call_count, 2)
        sleep.assert_called_once_with(0.1)

    @patch("backend.storage.time.sleep")
    def test_unavailable_database_has_bounded_retries(self, sleep):
        client = Mock()
        client.get_collections.side_effect = ResponseHandlingException(ConnectionError())
        with self.assertRaisesRegex(RuntimeError, "после 3 попыток"):
            wait_for_qdrant(client, attempts=3, delay=0)
        self.assertEqual(client.get_collections.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    @patch("backend.storage.time.sleep")
    def test_authentication_failure_is_not_retried(self, sleep):
        client = Mock()
        client.get_collections.side_effect = UnexpectedResponse(
            status_code=401, reason_phrase="Unauthorized", content=b"", headers={}
        )
        with self.assertRaises(UnexpectedResponse):
            wait_for_qdrant(client)
        client.get_collections.assert_called_once()
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
