"""Повторные ответы должны использовать исходный эксперимент без подмены данных."""

import copy
import unittest

from backend.prompts import SYSTEM_PROMPT
from backend.search import ANSWER_MODEL
from evaluation.repeat_answers import fixed_inputs
from evaluation.run import digest


class FixedInputTests(unittest.TestCase):
    def setUp(self):
        self.dataset = {"version": "fixture"}
        self.query = {"id": "q1", "query": "нужная книга", "expected_behavior": "recommend",
                      "judgments": [{"book_id": "a", "grade": 2}, {"book_id": "b", "grade": 0}]}
        self.source = {"query_id": "q1", "query": "нужная книга", "configuration": "20-to-5",
                       "status": "success", "dense_candidates": [{"id": "a"}, {"id": "b"}],
                       "llm_candidates": [{"id": "a"}], "answer": {"recommendations": [{"id": "a"}]}}
        self.baseline = {"system_prompt": SYSTEM_PROMPT, "answer_model": ANSWER_MODEL,
                         "dataset_sha256": digest(self.dataset), "index_unchanged": True,
                         "rows": [self.source]}

    def select(self):
        return fixed_inputs(self.baseline, self.dataset, [self.query], ["q1"], "20-to-5")

    def test_preserves_original_context_and_query_without_mutation(self):
        before = copy.deepcopy(self.baseline)
        query, row = self.select()[0]
        self.assertIs(query, self.query)
        self.assertIs(row, self.source)
        self.assertEqual(row["llm_candidates"], [{"id": "a"}])
        self.assertEqual(self.baseline, before)

    def test_refuses_changed_prompt_model_labels_or_index(self):
        for key, value in (("system_prompt", "another prompt"), ("answer_model", "another model"),
                           ("dataset_sha256", "another dataset"), ("index_unchanged", False)):
            with self.subTest(key=key):
                previous = self.baseline[key]
                self.baseline[key] = value
                with self.assertRaises(ValueError):
                    self.select()
                self.baseline[key] = previous

    def test_refuses_ambiguous_source_query_mismatch_and_unlabelled_context(self):
        self.baseline["rows"].append(self.source)
        with self.assertRaises(ValueError):
            self.select()
        self.baseline["rows"].pop()
        self.source["query"] = "другой запрос"
        with self.assertRaises(ValueError):
            self.select()
        self.source["query"] = self.query["query"]
        self.source["llm_candidates"].append({"id": "unknown"})
        with self.assertRaises(ValueError):
            self.select()


if __name__ == "__main__":
    unittest.main()
