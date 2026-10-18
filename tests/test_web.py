"""HTTP-контракт страницы и загрузки без вызовов моделей."""

import io
import json
import unittest
from pathlib import Path
from unittest.mock import Mock

from backend.catalog import validate_catalog
from backend.web import create_app


class WebTests(unittest.TestCase):
    def setUp(self):
        self.answer = Mock(return_value="Подходящие книги")
        self.import_books = Mock()
        self.app = create_app(self.answer, self.import_books)
        self.client = self.app.test_client()

    def upload(self, content):
        return self.client.post("/upload_books", data={
            "file": (io.BytesIO(content), "books.json"),
        })

    def test_page_and_assets_are_served_from_same_app(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'/static/app.js', response.data)
        self.assertIn(b'/static/style.css', response.data)
        for path in ("/static/app.js", "/static/style.css"):
            with self.client.get(path) as asset:
                self.assertEqual(asset.status_code, 200)
                self.assertTrue(asset.data)

    def test_search_validates_input_and_passes_trimmed_text(self):
        for data in ({}, [], {"message": 1}, {"message": "   "}):
            self.assertEqual(self.client.post("/chat", json=data).status_code, 400)
        self.answer.assert_not_called()
        response = self.client.post("/chat", json={"message": "  Космос  "})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"response": "Подходящие книги"})
        self.answer.assert_called_once_with("Космос")

    def test_catalog_validation_is_reported_as_400_and_temp_file_is_removed(self):
        paths = []

        def import_invalid(path):
            paths.append(Path(path))
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            validate_catalog(data)

        self.import_books.side_effect = import_invalid
        for content, expected in (
            (b'{"books": [{"title": "Book"}]}', "author"),
            (b'{broken', "JSON"),
        ):
            response = self.upload(content)
            self.assertEqual(response.status_code, 400)
            self.assertIn(expected, response.json["error"])
            self.assertFalse(paths[-1].exists())

    def test_successful_uploads_use_separate_files_and_clean_them_up(self):
        paths = []

        def import_file(path):
            paths.append(Path(path))
            self.assertEqual(Path(path).read_bytes(), b'{"books": []}')
            return {"added": 2, "message": "Добавлено 2 книги", "status": "success"}

        self.import_books.side_effect = import_file
        for _ in range(2):
            response = self.upload(b'{"books": []}')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json["added"], 2)
            self.assertFalse(paths[-1].exists())
        self.assertNotEqual(paths[0], paths[1])

    def test_missing_file_and_backend_failure_return_errors(self):
        self.assertEqual(self.client.post("/upload_books").status_code, 400)
        self.import_books.assert_not_called()
        paths = []

        def fail(path):
            paths.append(Path(path))
            raise RuntimeError("internal details")

        self.import_books.side_effect = fail
        with self.assertLogs(self.app.logger.name, level="ERROR"):
            response = self.upload(b'[]')
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("internal details", response.json["error"])
        self.assertFalse(paths[0].exists())

    def test_partial_and_failed_imports_keep_detailed_report(self):
        for status in ("partial", "failed"):
            report = {
                "status": status, "failed": 1, "message": "Ошибка одной книги",
                "errors": [{"title": "Книга", "stage": "dense", "error": "Ошибка API"}],
            }
            self.import_books.return_value = report
            response = self.upload(b'[]')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json, report)


if __name__ == "__main__":
    unittest.main()
