"""Поиск и генерация с контролируемыми ответами внешних API."""

import io
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests
from qdrant_client import QdrantClient

from backend import app, search, upload
from backend.catalog import book_id
from backend.storage import VECTOR_SIZE, prepare_collection
from backend.web import create_app


def book(point_id="1", **changes):
    return {
        "id": point_id, "title": "Космос", "author": "Автор", "year": 2000,
        "category": "Фантастика", "description": "Полёт к далёкой планете.",
        "enriched_description": "Путешествие исследователей к далёкой планете.",
        "dense_score": 0.7, **changes,
    }


def response_for(ids):
    return json.dumps({
        "recommendations": [{"id": value, "reason": "В описании есть космический полёт."} for value in ids],
        "summary": "Подборка по найденным описаниям.",
    }, ensure_ascii=False)


class SearchTests(unittest.TestCase):
    def test_embedding_failure_or_invalid_vector_never_becomes_empty_results(self):
        for value in (None, [0.0] * VECTOR_SIZE, [1.0]):
            with patch.object(search, "get_dense_vector", return_value=value), \
                 patch.object(search, "get_qdrant_client") as database:
                with self.assertRaisesRegex(search.SearchError, "вектор запроса"):
                    search.search_books("космос")
                database.assert_not_called()
        with patch.object(search, "get_dense_vector", side_effect=RuntimeError("API down")):
            with self.assertRaises(search.SearchError):
                search.search_books("космос")

    def test_database_failure_is_distinct_from_no_matches(self):
        database = Mock()
        database.query_points.side_effect = RuntimeError("database down")
        with patch.object(search, "get_dense_vector", return_value=[1.0] + [0.0] * (VECTOR_SIZE - 1)), \
             patch.object(search, "get_qdrant_client", return_value=database):
            with self.assertRaisesRegex(search.SearchError, "База книг"):
                search.search_books("космос")
            database.query_points.side_effect = None
            database.query_points.return_value = SimpleNamespace(points=[])
            self.assertEqual(search.search_books("космос"), [])

    def test_retrieved_books_keep_original_and_enriched_descriptions(self):
        payload = book()
        database = Mock()
        database.query_points.return_value = SimpleNamespace(points=[
            SimpleNamespace(id=7, score=0.8, payload=payload),
        ])
        with patch.object(search, "get_dense_vector", return_value=[1.0] + [0.0] * (VECTOR_SIZE - 1)) as embed, \
             patch.object(search, "get_qdrant_client", return_value=database):
            result = search.search_books("космос")[0]
            embed.assert_called_once_with("космос", input_type="query")
        self.assertEqual(result["id"], "7")
        self.assertEqual(result["description"], payload["description"])
        self.assertEqual(result["enriched_description"], payload["enriched_description"])

    def test_dense_mode_needs_no_openrouter_and_does_not_call_reranker(self):
        with patch.dict(os.environ, {"RERANK_ENABLED": "false", "OPENROUTER_API_KEY": ""}), \
             patch.object(search, "search_books", return_value=[book()]) as retrieve, \
             patch.object(search, "rerank_books") as rerank:
            search.validate_search_config()
            self.assertEqual(search.retrieve_books("космос"), [book()])
            retrieve.assert_called_once_with("космос", limit=5)
            rerank.assert_not_called()

    def test_enabled_reranker_failure_is_not_hidden_by_dense_fallback(self):
        with patch.dict(os.environ, {"RERANK_ENABLED": "true"}), \
             patch.object(search, "search_books", return_value=[book()]) as retrieve, \
             patch.object(search, "rerank_books", side_effect=search.SearchError("Реранкер недоступен")):
            with self.assertRaises(search.SearchError):
                search.retrieve_books("космос")
            retrieve.assert_called_once_with("космос", limit=20)

    def test_invalid_reranker_configuration_fails_explicitly(self):
        for env in (
            {"RERANK_ENABLED": "maybe"},
            {"RERANK_ENABLED": "true", "OPENROUTER_API_KEY": ""},
        ):
            with patch.dict(os.environ, env):
                with self.assertRaises(ValueError):
                    search.validate_search_config()

    def test_reranker_uses_its_own_key_descriptions_and_separate_scores(self):
        books = [book("1"), book("2", title="Другой космос")]
        response = Mock()
        response.json.return_value = {"results": [
            {"index": 0, "relevance_score": 0.3}, {"index": 1, "relevance_score": 0.9},
        ]}
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "router-test", "DEEPSEEK_API_KEY": "deepseek-test"}), \
             patch("requests.post", return_value=response) as post:
            result = search.rerank_books("космос", books)
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer router-test")
        document = post.call_args.kwargs["json"]["documents"][0]
        self.assertIn(books[0]["description"], document)
        self.assertIn(books[0]["enriched_description"], document)
        self.assertEqual([item["id"] for item in result], ["2", "1"])
        self.assertEqual(result[0]["dense_score"], 0.7)
        self.assertEqual(result[0]["rerank_score"], 0.9)
        self.assertNotIn("rerank_score", books[0])

    def test_reranker_http_and_schema_errors_are_explicit(self):
        response = Mock()
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "router-test"}), \
             patch("requests.post", return_value=response):
            response.raise_for_status.side_effect = requests.HTTPError("401")
            with self.assertRaises(search.SearchError):
                search.rerank_books("космос", [book()])
            response.raise_for_status.side_effect = None
            for data in (
                {}, {"results": []}, {"results": [{"index": 7, "relevance_score": 0.9}]},
                {"results": [{"index": True, "relevance_score": 0.9}]},
                {"results": [{"index": 0, "relevance_score": float("nan")}]},
            ):
                response.json.return_value = data
                with self.subTest(data=data), self.assertRaises(search.SearchError):
                    search.rerank_books("космос", [book()])
            response.json.return_value = {"results": [
                {"index": 0, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.8},
            ]}
            with self.assertRaises(search.SearchError):
                search.rerank_books("космос", [book(), book("2")])

    def test_generation_receives_descriptions_and_names_are_rendered_from_catalog(self):
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response_for(["1"])))],
        )
        with patch.object(search, "get_llm_client", return_value=client):
            answer = search.generate_answer("космос", [book()])
        content = json.loads(client.chat.completions.create.call_args.kwargs["messages"][1]["content"])
        self.assertEqual(content["books"][0]["description"], book()["description"])
        self.assertEqual(content["books"][0]["enriched_description"], book()["enriched_description"])
        options = client.chat.completions.create.call_args.kwargs
        self.assertEqual(options["model"], upload.LLM_MODEL)
        self.assertEqual(options["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertEqual(options["response_format"], {"type": "json_object"})
        self.assertIn("Космос — Автор", answer)

    def test_thinking_toggle_preserves_messages_and_rejects_invalid_mode_before_api(self):
        client = Mock()
        with patch.object(search, "get_llm_client", return_value=client):
            search.request_answer("космос", [book()])
            disabled = client.chat.completions.create.call_args.kwargs
            search.request_answer("космос", [book()], thinking="enabled")
            enabled = client.chat.completions.create.call_args.kwargs
            self.assertEqual(enabled["messages"], disabled["messages"])
            self.assertEqual(enabled["extra_body"], {"thinking": {"type": "enabled"}})
            client.reset_mock()
            with self.assertRaises(ValueError):
                search.request_answer("космос", [book()], thinking="typo")
            client.chat.completions.create.assert_not_called()

    def test_generation_rejects_unknown_duplicates_and_malformed_output(self):
        for content in (None, "", "not json", "[]", response_for(["unknown"]), response_for(["1", "1"])):
            with self.subTest(content=content), self.assertRaises(ValueError):
                search.render_answer(content, [book(), book("2")])
        content = json.dumps({"recommendations": [], "summary": "В описаниях нет подтверждения запросу."})
        self.assertEqual(search.render_answer(content, [book()]), "В описаниях нет подтверждения запросу.")

    def test_generation_api_failure_is_not_a_successful_answer(self):
        client = Mock()
        client.chat.completions.create.side_effect = RuntimeError("API down")
        with patch.object(search, "get_llm_client", return_value=client):
            with self.assertRaises(search.SearchError):
                search.generate_answer("космос", [book()])

    def test_graph_always_retrieves_before_generation_and_keeps_requests_separate(self):
        calls = []
        def retrieve(query):
            calls.append(("retrieve", query))
            return [book()]
        def generate(query, books):
            calls.append(("generate", query))
            return query
        with patch.object(app, "retrieve_books", side_effect=retrieve), \
             patch.object(app, "generate_answer", side_effect=generate):
            self.assertEqual(app.answer_question("Первый"), "Первый")
            self.assertEqual(app.answer_question("Второй"), "Второй")
        self.assertEqual(calls, [
            ("retrieve", "Первый"), ("generate", "Первый"),
            ("retrieve", "Второй"), ("generate", "Второй"),
        ])

    def test_empty_catalog_skips_generation_and_failure_propagates(self):
        with patch.object(app, "retrieve_books", return_value=[]) as retrieve, \
             patch.object(app, "generate_answer") as generate:
            self.assertIn("не найдено книг", app.answer_question("космос"))
            generate.assert_not_called()
            retrieve.side_effect = search.SearchError("База недоступна")
            with self.assertRaises(search.SearchError):
                app.answer_question("космос")
            generate.assert_not_called()


class LocalWorkflowTests(unittest.TestCase):
    def test_http_upload_qdrant_graph_and_answer_without_external_api(self):
        database = QdrantClient(":memory:")
        self.addCleanup(database.close)
        prepare_collection(database)
        record = {key: value for key, value in book().items() if key not in {
            "id", "dense_score", "enriched_description",
        }}
        vector = [1.0] + [0.0] * (VECTOR_SIZE - 1)
        model = Mock()
        model.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response_for([book_id(record)])))],
        )
        client = app.flask_app.test_client()
        with patch.dict(os.environ, {"RERANK_ENABLED": "false"}), \
             patch.object(upload, "get_qdrant_client", return_value=database), \
             patch.object(upload, "enrich_description", return_value="Обогащённое описание космоса"), \
             patch.object(upload, "get_dense_vector", return_value=vector), \
             patch.object(search, "get_qdrant_client", return_value=database), \
             patch.object(search, "get_dense_vector", return_value=vector), \
             patch.object(search, "get_llm_client", return_value=model), \
             patch("requests.post", side_effect=AssertionError("Внешний API запрещён")):
            response = client.post("/upload_books", data={
                "file": (io.BytesIO(json.dumps([record]).encode("utf-8")), "books.json"),
            })
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json["added"], 1)
            answer = client.post("/chat", json={"message": "Книга о космосе"})
            self.assertEqual(answer.status_code, 200)
            self.assertIn("Космос — Автор", answer.json["response"])
            model.chat.completions.create.assert_called_once()

    def test_http_distinguishes_service_failure_from_empty_catalog(self):
        answer = Mock(side_effect=search.SearchError("База книг недоступна."))
        web = create_app(answer, Mock())
        with self.assertLogs(web.logger.name, level="ERROR"):
            response = web.test_client().post("/chat", json={"message": "космос"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json, {"error": "База книг недоступна."})


if __name__ == "__main__":
    unittest.main()
