import json
import time
from datetime import datetime
from pathlib import Path

from production_controller import (
    ProductionRAGController,
)


OUTPUT_FILE = Path(
    "capstone_run.json"
)


class CapstoneRAG:

    def __init__(self):

        self.controller = (
            ProductionRAGController()
        )

        self.started_at = (
            datetime.now().isoformat()
        )

        self.requests = 0

        self.results = []

    def query(self, question):

        self.requests += 1

        started = time.perf_counter()

        result = self.controller.query(
            question
        )

        elapsed = (
            time.perf_counter()
            - started
        )

        result[
            "capstone_latency_seconds"
        ] = elapsed

        self.results.append(result)

        return result

    def summary(self):

        total = len(self.results)

        answered = sum(
            1
            for r in self.results
            if r.get("status") == "answered"
        )

        abstained = sum(
            1
            for r in self.results
            if r.get("status") == "abstained"
        )

        errors = sum(
            1
            for r in self.results
            if r.get("status") == "error"
        )

        evidence = [
            r["evidence"]["context_recall"]
            for r in self.results
            if r.get("evidence", {}).get(
                "context_recall"
            ) is not None
        ]

        latencies = [
            r["capstone_latency_seconds"]
            for r in self.results
            if r.get(
                "capstone_latency_seconds"
            ) is not None
        ]

        generation_hits = sum(
            1
            for r in self.results
            if r.get("cache", {}).get(
                "generation_hit",
                False,
            )
        )

        return {
            "requests": total,

            "answered": answered,

            "abstained": abstained,

            "errors": errors,

            "answer_rate": (
                answered / total
                if total
                else 0.0
            ),

            "abstention_rate": (
                abstained / total
                if total
                else 0.0
            ),

            "error_rate": (
                errors / total
                if total
                else 0.0
            ),

            "mean_evidence_recall": (
                sum(evidence)
                / len(evidence)
                if evidence
                else None
            ),

            "mean_latency_seconds": (
                sum(latencies)
                / len(latencies)
                if latencies
                else None
            ),

            "generation_cache_hit_rate": (
                generation_hits / total
                if total
                else 0.0
            ),
        }

    def save(self):

        report = {
            "capstone_version": 1,

            "started_at": self.started_at,

            "finished_at":
                datetime.now().isoformat(),

            "summary": self.summary(),

            "results": self.results,
        }

        with open(
            OUTPUT_FILE,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                report,
                f,
                indent=2,
            )

        return report


def main():

    rag = CapstoneRAG()

    print()
    print(
        "======================================"
    )
    print(
        "      ADAPTIVE RAG CAPSTONE"
    )
    print(
        "======================================"
    )

    print(
        "Type 'exit' to finish."
    )

    while True:

        question = input(
            "\nQuestion: "
        ).strip()

        if question.lower() == "exit":
            break

        if not question:
            print(
                "Please enter a question."
            )
            continue

        result = rag.query(
            question
        )

        print()

        print(
            json.dumps(
                result,
                indent=2,
            )
        )

    report = rag.save()

    print()
    print(
        "======================================"
    )
    print(
        "CAPSTONE SUMMARY"
    )
    print(
        "======================================"
    )

    print(
        json.dumps(
            report["summary"],
            indent=2,
        )
    )

    print()
    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()