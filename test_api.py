import unittest

from fastapi.testclient import TestClient

import api


class MockController:

    def query(self, question):

        return {
            "status": "answered",
            "answer": "Annual leave is provided according to company policy.",
            "retrieval_evidence_recall": 1.0,
            "context_evidence_recall": 1.0,
            "judgement": {
                "groundedness": 2,
                "relevance": 2,
                "completeness": 2,
                "unsupported_claims": 0
            }
        }


class TestRAGAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.original_controller = api.controller

        api.controller = MockController()

        cls.client = TestClient(api.app)

    @classmethod
    def tearDownClass(cls):

        api.controller = cls.original_controller

    def test_health(self):

        response = self.client.get(
            "/health"
        )

        self.assertEqual(
            response.status_code,
            200
        )

        self.assertEqual(
            response.json()["status"],
            "healthy"
        )

    def test_ready(self):

        response = self.client.get(
            "/ready"
        )

        self.assertEqual(
            response.status_code,
            200
        )

        self.assertEqual(
            response.json()["status"],
            "ready"
        )

    def test_query_contract(self):

        response = self.client.post(
            "/query",
            json={
                "question":
                    "What is the annual leave policy?"
            }
        )

        self.assertEqual(
            response.status_code,
            200
        )

        data = response.json()

        required_fields = [
            "request_id",
            "status",
            "question",
            "answer",
            "performance",
            "retrieval_evidence_recall",
            "context_evidence_recall",
            "judgement"
        ]

        for field in required_fields:

            self.assertIn(
                field,
                data
            )

    def test_question_normalization(self):

        response = self.client.post(
            "/query",
            json={
                "question":
                    "Question: What is the annual leave policy?"
            }
        )

        self.assertEqual(
            response.status_code,
            200
        )

        self.assertEqual(
            response.json()["question"],
            "What is the annual leave policy?"
        )

    def test_missing_question(self):

        response = self.client.post(
            "/query",
            json={}
        )

        self.assertEqual(
            response.status_code,
            422
        )

    def test_question_too_short(self):

        response = self.client.post(
            "/query",
            json={
                "question": "hi"
            }
        )

        self.assertEqual(
            response.status_code,
            422
        )

    def test_question_too_long(self):

        response = self.client.post(
            "/query",
            json={
                "question": "x" * 2001
            }
        )

        self.assertEqual(
            response.status_code,
            422
        )


if __name__ == "__main__":

    unittest.main(
        verbosity=2
    )