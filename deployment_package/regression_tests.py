import json
import unittest
from pathlib import Path

from cache import cache_get, cache_set, make_key
from config import load_config, validate_config
from adaptive_rag import choose_budget


BASE_DIR = Path(__file__).resolve().parent


class TestConfig(unittest.TestCase):

    def test_config_loads(self):
        config = load_config()

        self.assertIn("chunking", config)
        self.assertIn("retrieval", config)
        self.assertIn("evidence", config)
        self.assertIn("context", config)
        self.assertIn("ollama", config)

        validate_config(config)

    def test_chunk_configuration(self):
        config = load_config()

        chunk_size = config["chunking"]["chunk_size"]
        overlap = config["chunking"]["overlap"]

        self.assertGreater(chunk_size, 0)
        self.assertGreaterEqual(overlap, 0)
        self.assertLess(overlap, chunk_size)


class TestBudgetPolicy(unittest.TestCase):

    def test_insufficient_evidence_abstains(self):
        result = choose_budget(
            question="test question",
            evidence_recall=0.5,
            reranked_documents=[
                {
                    "id": "doc1",
                    "document": "test document",
                    "metadata": {}
                }
            ],
            policy={}
        )

        self.assertEqual(result["budget"], 0)
        self.assertEqual(result["reason"], "insufficient_evidence")

    def test_no_documents_abstains(self):
        result = choose_budget(
            question="test question",
            evidence_recall=1.0,
            reranked_documents=[],
            policy={}
        )

        self.assertEqual(result["budget"], 0)
        self.assertEqual(result["reason"], "no_retrieved_documents")

    def test_list_policy(self):
        policy = [
            {
                "question": "test question",
                "recommended_budget": 2
            }
        ]

        result = choose_budget(
            question="test question",
            evidence_recall=1.0,
            reranked_documents=[
                {
                    "id": "doc1",
                    "document": "test",
                    "metadata": {}
                }
            ],
            policy=policy
        )

        self.assertEqual(result["budget"], 2)
        self.assertEqual(
            result["reason"],
            "question_specific_policy"
        )

    def test_global_policy(self):
        policy = {
            "recommended_budget": 4
        }

        result = choose_budget(
            question="unknown question",
            evidence_recall=1.0,
            reranked_documents=[
                {
                    "id": "doc1",
                    "document": "test",
                    "metadata": {}
                }
            ],
            policy=policy
        )

        self.assertEqual(result["budget"], 4)


class TestCache(unittest.TestCase):

    def test_cache_roundtrip(self):
        key = make_key(
            "regression_test",
            {"question": "cache test"}
        )

        value = {
            "answer": "cached answer",
            "cache_hit": False
        }

        cache_set(key, value)

        loaded = cache_get(key)

        self.assertEqual(loaded, value)


class TestAdaptiveReport(unittest.TestCase):

    def test_report_schema(self):
        report_file = BASE_DIR / "adaptive_rag_report.json"

        if not report_file.exists():
            self.skipTest(
                "adaptive_rag_report.json not found"
            )

        with open(report_file, "r", encoding="utf-8") as file:
            report = json.load(file)

        self.assertIsInstance(report, dict)

        self.assertIn("config", report)
        self.assertIn("summary", report)

        results = report.get("results", [])

        self.assertIsInstance(results, list)

        for result in results:
            self.assertIn("question", result)
            self.assertIn("status", result)
            self.assertIn("budget", result)
            self.assertIn(
                "retrieval_evidence_recall",
                result
            )
            self.assertIn(
                "context_evidence_recall",
                result
            )


class TestFinalEvaluation(unittest.TestCase):

    def test_final_evaluation_schema(self):
        report_file = (
            BASE_DIR /
            "final_adaptive_evaluation.json"
        )

        if not report_file.exists():
            self.skipTest(
                "final_adaptive_evaluation.json not found"
            )

        with open(report_file, "r", encoding="utf-8") as file:
            report = json.load(file)

        self.assertIn("summary", report)

        question_results = report.get(
            "question_results",
            []
        )

        self.assertIsInstance(
            question_results,
            list
        )


class TestFailureMonitoring(unittest.TestCase):

        def test_failure_monitor_schema(self):
          report_file = (
                    BASE_DIR /
                    "failure_monitoring.json"
          )

          if not report_file.exists():
                    self.skipTest(
                    "failure_monitoring.json not found"
                    )

          with open(report_file, "r", encoding="utf-8") as file:
                    report = json.load(file)

          self.assertIn("summary", report)
          self.assertIn("question_results", report)

          summary = report["summary"]

          self.assertIn(
                    "category_counts",
                    summary
          )

          self.assertIsInstance(
                    summary["category_counts"],
                    dict
          )

          self.assertIsInstance(
                    report["question_results"],
                    list
          )

if __name__ == "__main__":
    unittest.main(verbosity=2)