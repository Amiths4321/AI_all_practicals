import json
import time
from datetime import datetime
from pathlib import Path

from production_controller import ProductionRAGController


QUESTIONS_FILE = Path("evaluation_data.json")
OUTPUT_FILE = Path("end_to_end_benchmark.json")


def load_questions():
    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in [
            "questions",
            "evaluation",
            "examples",
        ]:
            if isinstance(data.get(key), list):
                return data[key]

    raise ValueError(
        "Could not find evaluation questions."
    )


def extract_question(item):
    if isinstance(item, str):
        return item

    if isinstance(item, dict):
        for key in [
            "question",
            "query",
            "prompt",
        ]:
            if item.get(key):
                return item[key]

    return None


def run_benchmark():

    raw_questions = load_questions()

    questions = [
        extract_question(item)
        for item in raw_questions
    ]

    questions = [
        q for q in questions
        if q
    ]

    controller = (
        ProductionRAGController()
    )

    results = []

    benchmark_started = time.perf_counter()

    for index, question in enumerate(
        questions,
        start=1,
    ):

        print(
            f"[{index}/{len(questions)}] "
            f"{question}"
        )

        started = time.perf_counter()

        result = controller.query(
            question
        )

        elapsed = (
            time.perf_counter()
            - started
        )

        result["benchmark_latency_seconds"] = (
            elapsed
        )

        results.append(result)

    benchmark_latency = (
        time.perf_counter()
        - benchmark_started
    )

    return {
        "benchmark_version": 1,

        "timestamp": datetime.now().isoformat(),

        "questions": len(questions),

        "total_benchmark_latency_seconds":
            benchmark_latency,

        "results": results,
    }


def summarize(report):

    results = report["results"]

    if not results:
        return {}

    successful = [
        r for r in results
        if r.get("status") == "answered"
    ]

    abstained = [
        r for r in results
        if r.get("status") == "abstained"
    ]

    errors = [
        r for r in results
        if r.get("status") == "error"
    ]

    evidence = [
        r["evidence"]["context_recall"]
        for r in results
        if r.get("evidence", {}).get(
            "context_recall"
        ) is not None
    ]

    latencies = [
        r["benchmark_latency_seconds"]
        for r in results
        if r.get(
            "benchmark_latency_seconds"
        ) is not None
    ]

    generation_cache_hits = sum(
        1
        for r in results
        if r.get("cache", {}).get(
            "generation_hit",
            False,
        )
    )

    judge_cache_hits = sum(
        1
        for r in results
        if r.get("cache", {}).get(
            "judge_hit",
            False,
        )
    )

    total = len(results)

    return {
        "questions": total,

        "answered": len(successful),

        "abstained": len(abstained),

        "errors": len(errors),

        "answer_rate": (
            len(successful) / total
            if total else 0
        ),

        "abstention_rate": (
            len(abstained) / total
            if total else 0
        ),

        "error_rate": (
            len(errors) / total
            if total else 0
        ),

        "mean_context_evidence_recall": (
            sum(evidence) / len(evidence)
            if evidence else None
        ),

        "mean_latency_seconds": (
            sum(latencies) / len(latencies)
            if latencies else None
        ),

        "generation_cache_hit_rate": (
            generation_cache_hits / total
            if total else 0
        ),

        "judge_cache_hit_rate": (
            judge_cache_hits / total
            if total else 0
        ),
    }


def main():

    report = run_benchmark()

    report["summary"] = summarize(
        report
    )

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

    summary = report["summary"]

    print()
    print(
        "End-to-end benchmark complete."
    )
    print(
        f"Questions: "
        f"{summary['questions']}"
    )
    print(
        f"Answered: "
        f"{summary['answered']}"
    )
    print(
        f"Abstained: "
        f"{summary['abstained']}"
    )
    print(
        f"Errors: "
        f"{summary['errors']}"
    )
    print(
        f"Evidence recall: "
        f"{summary['mean_context_evidence_recall']}"
    )
    print(
        f"Mean latency: "
        f"{summary['mean_latency_seconds']}"
    )
    print(
        f"Generation cache hit rate: "
        f"{summary['generation_cache_hit_rate']}"
    )
    print(
        f"Judge cache hit rate: "
        f"{summary['judge_cache_hit_rate']}"
    )
    print(
        f"Output: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()