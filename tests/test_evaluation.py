"""Проверка пограничных случаев метрик и запрета API при повторном прогоне."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from evaluation.metrics import average_metrics, score_answer
from evaluation.run import Cache


def query(grades, behavior="recommend"):
    return {"expected_behavior": behavior,
            "judgments": [{"book_id": key, "grade": grade} for key, grade in grades.items()]}


def answer(ids):
    return {"recommendations": [{"id": key} for key in ids]}


class EvaluationTests(unittest.TestCase):
    def test_missing_recommendations_and_noise_have_different_penalties(self):
        q = query({"a": 2, "b": 2, "c": 2, "d": 2, "noise": 0})
        result = score_answer(answer(["a", "b", "noise"]), q,
                              ["a", "b", "c", "d", "noise"], ["a", "b", "noise"])
        self.assertAlmostEqual(result["precision_strong"], 2 / 3)
        self.assertEqual(result["coverage_capped_at_5"], 0.5)
        self.assertEqual(result["dense_recall_strong"], 1)
        self.assertEqual(result["context_recall_strong"], 0.5)

    def test_maximum_five_does_not_penalize_a_large_relevant_set(self):
        ids = [str(i) for i in range(8)]
        result = score_answer(answer(ids[:5]), query(dict.fromkeys(ids, 2)), ids, ids)
        self.assertEqual(result["coverage_capped_at_5"], 1)
        self.assertEqual(result["precision_strong"], 1)

    def test_empty_answer_on_answerable_query_is_zero_not_missing(self):
        result = score_answer(answer([]), query({"a": 2}), ["a"], ["a"])
        self.assertEqual(result["precision_strong"], 0)
        self.assertEqual(result["precision_supported"], 0)
        self.assertEqual(result["coverage_capped_at_5"], 0)
        self.assertEqual(result["false_refusal"], 1)

    def test_no_match_is_scored_separately(self):
        q = query({"a": 0}, "no_match")
        result = score_answer(answer([]), q, ["a"], ["a"])
        self.assertEqual(result["correct_no_match"], 1)
        self.assertIsNone(result["precision_strong"])
        self.assertIsNone(result["precision_supported"])
        self.assertIsNone(result["coverage_capped_at_5"])
        self.assertEqual(score_answer(answer(["a"]), q, ["a"], ["a"])["correct_no_match"], 0)

    def test_partial_match_is_not_silently_counted_as_strong(self):
        result = score_answer(answer(["b"]), query({"a": 2, "b": 1}), ["a", "b"], ["b"])
        self.assertEqual(result["partial_count"], 1)
        self.assertEqual(result["irrelevant_count"], 0)
        self.assertEqual(result["precision_strong"], 0)
        self.assertEqual(result["precision_supported"], 1)

    def test_supported_precision_distinguishes_partial_matches_from_noise(self):
        q = query({"exact": 2, "partial": 1, "noise": 0})
        ids = ["exact", "partial", "noise"]
        result = score_answer(answer(ids), q, ids, ids)
        self.assertAlmostEqual(result["precision_strong"], 1 / 3)
        self.assertAlmostEqual(result["precision_supported"], 2 / 3)
        self.assertEqual(result["irrelevant_count"], 1)

    def test_incomplete_labels_and_invalid_ids_are_not_successes(self):
        with self.assertRaises(ValueError):
            score_answer(answer([]), query({"a": None}), ["a"], ["a"])
        for ids in (["unknown"], ["a", "a"]):
            with self.assertRaises(ValueError):
                score_answer(answer(ids), query({"a": 2}), ["a"], ["a"])

    def test_api_error_is_excluded_from_quality_but_counted(self):
        metrics = score_answer(answer(["a"]), query({"a": 2}), ["a"], ["a"])
        result = average_metrics([{"status": "error"}, {"status": "success", "metrics": metrics}])
        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["precision_strong"], {"mean": 1, "queries": 1})

    def test_cache_reuses_successes_invalidates_parameters_and_never_stores_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Cache(tmp)
            produce = Mock(return_value={"vector": [1, 2]})
            value, timing = cache.get("embedding", {"model": "a"}, produce)
            self.assertFalse(timing["cached"])
            replay = Cache(tmp, offline=True)
            self.assertEqual(replay.get("embedding", {"model": "a"}, produce)[0], value)
            self.assertTrue(replay.get("embedding", {"model": "a"}, produce)[1]["cached"])
            produce.assert_called_once()
            with self.assertRaises(RuntimeError):
                replay.get("embedding", {"model": "b"}, produce)
            with self.assertRaises(RuntimeError):
                cache.get("answer", {}, Mock(side_effect=RuntimeError("API failed")))
            self.assertFalse(list((Path(tmp) / "answer").glob("*.json")))


if __name__ == "__main__":
    unittest.main()
