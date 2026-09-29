import json
from datetime import datetime
from pathlib import Path

from adaptive_rag import process_question


class RAGService:

    def __init__(self):
        self.started_at = datetime.now().isoformat()

    def query(self, question):
        if not question or not question.strip():
            return {
                "status": "error",
                "error": "Question cannot be empty.",
            }

        question = question.strip()

        try:
            result = process_question(question)

            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "answer": result.get("answer"),
                "answerable": result.get(
                    "answerable"
                ),
                "budget": result.get("budget"),
                "budget_reason": result.get(
                    "budget_reason"
                ),
                "evidence_recall": result.get(
                    "context_evidence_recall"
                ),
                "latency_seconds": result.get(
                    "latency_seconds"
                ),
                "judgement": result.get(
                    "judgement"
                ),
            }

        except Exception as exc:

            return {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }


def main():

    service = RAGService()

    question = input(
        "Question: "
    ).strip()

    result = service.query(question)

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()