import json
import time
from datetime import datetime

from adaptive_rag import process_question


class ProductionRAGController:

    def __init__(self):
        self.request_count = 0
        self.started_at = datetime.now().isoformat()

    def query(self, question):

        self.request_count += 1

        request_id = (
            f"rag-{self.request_count:06d}"
        )

        started = time.perf_counter()

        if not question or not question.strip():

            return {
                "request_id": request_id,
                "status": "error",
                "error": "Question cannot be empty.",
            }

        question = question.strip()

        try:

            result = process_question(
                question
            )

            latency = (
                time.perf_counter()
                - started
            )

            return {
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),

                "status": result.get(
                    "status",
                    "unknown",
                ),

                "question": question,

                "answerable": result.get(
                    "answerable"
                ),

                "answer": result.get(
                    "answer"
                ),

                "policy": {
                    "budget": result.get(
                        "budget"
                    ),
                    "reason": result.get(
                        "budget_reason"
                    ),
                },

                "evidence": {
                    "retrieval_recall": result.get(
                        "retrieval_evidence_recall"
                    ),
                    "context_recall": result.get(
                        "context_evidence_recall"
                    ),
                },

                "quality": result.get(
                    "judgement",
                    {},
                ),

                "performance": {
                    "controller_latency_seconds":
                        latency,
                    "generation_latency_seconds":
                        result.get(
                            "latency_seconds"
                        ),
                },

                "cache": {
                    "generation_hit":
                        result.get(
                            "generation_cache_hit",
                            False,
                        ),
                    "judge_hit":
                        result.get(
                            "judge_cache_hit",
                            False,
                        ),
                },

            }

        except Exception as exc:

            latency = (
                time.perf_counter()
                - started
            )

            return {
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "question": question,

                "performance": {
                    "controller_latency_seconds":
                        latency,
                },

                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }


def main():

    controller = ProductionRAGController()

    print(
        "Production RAG Controller"
    )
    print(
        "Type 'exit' to stop."
    )

    while True:

        question = input(
            "\nQuestion: "
        ).strip()

        if question.lower() == "exit":
            break

        result = controller.query(
            question
        )

        print(
            json.dumps(
                result,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()