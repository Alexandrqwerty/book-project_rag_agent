"""Опубликованные результаты пересчитываются; подготовка не заменяет чужой индекс."""

import copy
import hashlib
import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from qdrant_client import QdrantClient

from backend.storage import COLLECTION_NAME, VECTOR_SIZE
from evaluation.prepare_index import prepare_index
from evaluation.replay import rescore
from evaluation.run import digest, load_dataset

ROOT = Path(__file__).resolve().parents[1]


class PublishedExperimentTests(unittest.TestCase):
    def test_all_published_reports_match_checksums_labels_and_metrics(self):
        references = ROOT / "evaluation/reference"
        manifest = json.loads((references / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest), 9)
        with patch("evaluation.run.get_dense_vector", side_effect=AssertionError("API")), \
             patch("evaluation.run.get_qdrant_client", side_effect=AssertionError("Qdrant")), \
             patch("evaluation.run.request_answer", side_effect=AssertionError("API")):
            for entry in manifest:
                with self.subTest(report=entry["file"]):
                    raw = (references / entry["file"]).read_bytes()
                    self.assertEqual(hashlib.sha256(raw).hexdigest(), entry["sha256"])
                    dataset, queries = load_dataset(ROOT / entry["dataset"], 10000)
                    report = json.loads(raw)
                    before = digest(report)
                    result = rescore(report, dataset, queries)
                    self.assertEqual(digest(report), before)
                    self.assertEqual(len(result["rows"]), len(report["rows"]))
                    for config, summary in report.get("paired_summary", {}).items():
                        for key, value in summary.items():
                            self.assertEqual(result["paired_summary"][config][key], value)

    def test_replay_rejects_changed_labels_and_metrics(self):
        dataset, queries = load_dataset(ROOT / "evaluation/book_search_v4.json", 10000)
        report = json.loads((ROOT / "evaluation/reference/context_comparison.json").read_text(
            encoding="utf-8"))
        changed = copy.deepcopy(dataset)
        changed["version"] = "different"
        with self.assertRaises(ValueError):
            rescore(report, changed, queries)
        report["rows"][0]["metrics"]["precision_strong"] = -1
        with self.assertRaises(ValueError):
            rescore(report, dataset, queries)


class SnapshotIndexTests(unittest.TestCase):
    def setUp(self):
        self.client = QdrantClient(":memory:")
        self.addCleanup(self.client.close)
        dataset, _ = load_dataset(ROOT / "evaluation/book_search_v4.json", 15)
        self.dataset = dataset
        self.cache = Mock()
        self.cache.get.return_value = ([1.0] + [0.0] * (VECTOR_SIZE - 1), {})

    def test_preparation_preserves_snapshot_and_rejects_existing_collection(self):
        with patch("evaluation.prepare_index.get_dense_vector", side_effect=AssertionError("API")):
            self.assertEqual(prepare_index(self.client, self.dataset, self.cache), 47)
        for item in self.dataset["database_evidence_snapshot"]:
            point = self.client.retrieve(COLLECTION_NAME, [item["book_id"]])[0]
            for key, value in item.items():
                if key != "book_id":
                    self.assertEqual(point.payload[key], value)
        self.cache.reset_mock()
        with self.assertRaises(ValueError):
            prepare_index(self.client, self.dataset, self.cache)
        self.cache.get.assert_not_called()
        self.assertEqual(self.client.count(COLLECTION_NAME).count, 47)

    def test_failed_embedding_does_not_create_incomplete_collection(self):
        self.cache.get.side_effect = [self.cache.get.return_value, RuntimeError("embedding failed")]
        with self.assertRaises(RuntimeError):
            prepare_index(self.client, self.dataset, self.cache)
        self.assertFalse(self.client.collection_exists(COLLECTION_NAME))


if __name__ == "__main__":
    unittest.main()
