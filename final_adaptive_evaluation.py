import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

ADAPTIVE_FILE = BASE_DIR / "adaptive_rag_report.json"
FIXED_FILE = BASE_DIR / "evidence_policy_benchmark.json"

OUTPUT_FILE = BASE_DIR / "final_adaptive_evaluation.json"

FIXED_BUDGET = "3"


def quality_index(judgement):
    if not judgement:
        return 0.0

    groundedness = judgement.get("groundedness", 0)
    relevance = judgement.get("relevance", 0)
    completeness = judgement.get("completeness", 0)

    return (
        groundedness
        + relevance
        + completeness
    ) / 6.0


def safe_average(values):
    values = [value for value in values if value is not None]

    if not values:
        return 0.0

    return sum(values) / len(values)


def extract_fixed_results():
    with open(FIXED_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    results = {}

    for question_data in data:
        question = question_data["question"]

        budget_data = question_data.get(
            "budgets",
            {}
        ).get(FIXED_BUDGET)

        if budget_data is None:
            continue

        results[question] = {
            "evidence_recall": budget_data.get(
                "evidence_recall",
                0.0
            ),
            "context_characters": budget_data.get(
                "context_characters",
                0
            ),
            "latency": budget_data.get(
                "generation_latency_seconds",
                0.0
            ),
            "judgement": budget_data.get(
                "judgement"
            ),
            "quality_index": quality_index(
                budget_data.get("judgement")
            )
        }

    return results


def extract_adaptive_results():
    with open(ADAPTIVE_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Support either:
    # {"question_results": [...]}
    # or a direct list.
    if isinstance(data, dict):
        question_results = data.get(
            "question_results",
            data.get("results", [])
        )
    else:
        question_results = data

    results = {}

    for item in question_results:
        question = item.get("question")

        if not question:
            continue

        judgement = item.get("judgement")

        results[question] = {
            "evidence_recall": item.get(
                "context_evidence_recall",
                item.get("evidence_recall", 0.0)
            ),
            "context_characters": item.get(
                "context_characters",
                0
            ),
            "latency": item.get(
                "generation_latency_seconds",
                item.get("latency_seconds", 0.0)
            ),
            "judgement": judgement,
            "quality_index": quality_index(judgement),
            "budget": item.get("budget"),
            "decision_reason": item.get(
                "budget_reason",
                item.get("reason")
            ),
            "abstained": item.get(
                "abstained",
                False
            )
        }

    return results


def compare_question(question, fixed, adaptive):
    if adaptive is None:
        return {
            "question": question,
            "status": "missing_adaptive_result"
        }

    fixed_quality = fixed["quality_index"]
    adaptive_quality = adaptive["quality_index"]

    return {
        "question": question,

        "fixed_budget": FIXED_BUDGET,
        "adaptive_budget": adaptive.get("budget"),

        "fixed_evidence_recall": fixed["evidence_recall"],
        "adaptive_evidence_recall": adaptive["evidence_recall"],

        "fixed_context_characters": fixed[
            "context_characters"
        ],
        "adaptive_context_characters": adaptive[
            "context_characters"
        ],

        "context_reduction": (
            1
            - adaptive["context_characters"]
            / fixed["context_characters"]
            if fixed["context_characters"] > 0
            else 0.0
        ),

        "fixed_latency": fixed["latency"],
        "adaptive_latency": adaptive["latency"],

        "fixed_quality_index": fixed_quality,
        "adaptive_quality_index": adaptive_quality,

        "quality_delta": (
            adaptive_quality - fixed_quality
        ),

        "evidence_delta": (
            adaptive["evidence_recall"]
            - fixed["evidence_recall"]
        ),

        "latency_delta": (
            adaptive["latency"]
            - fixed["latency"]
        ),

        "adaptive_decision_reason": adaptive.get(
            "decision_reason"
        ),

        "adaptive_abstained": adaptive.get(
            "abstained",
            False
        )
    }


def aggregate(comparisons):
    if not comparisons:
        return {}

    fixed_context = [
        item["fixed_context_characters"]
        for item in comparisons
    ]

    adaptive_context = [
        item["adaptive_context_characters"]
        for item in comparisons
    ]

    fixed_latency = [
        item["fixed_latency"]
        for item in comparisons
    ]

    adaptive_latency = [
        item["adaptive_latency"]
        for item in comparisons
    ]

    fixed_evidence = [
        item["fixed_evidence_recall"]
        for item in comparisons
    ]

    adaptive_evidence = [
        item["adaptive_evidence_recall"]
        for item in comparisons
    ]

    fixed_quality = [
        item["fixed_quality_index"]
        for item in comparisons
    ]

    adaptive_quality = [
        item["adaptive_quality_index"]
        for item in comparisons
    ]

    return {
        "questions": len(comparisons),

        "fixed_budget": FIXED_BUDGET,

        "fixed_average_context_characters": safe_average(
            fixed_context
        ),

        "adaptive_average_context_characters": safe_average(
            adaptive_context
        ),

        "context_reduction": (
            1
            - safe_average(adaptive_context)
            / safe_average(fixed_context)
            if safe_average(fixed_context) > 0
            else 0.0
        ),

        "fixed_average_latency_seconds": safe_average(
            fixed_latency
        ),

        "adaptive_average_latency_seconds": safe_average(
            adaptive_latency
        ),

        "fixed_average_evidence_recall": safe_average(
            fixed_evidence
        ),

        "adaptive_average_evidence_recall": safe_average(
            adaptive_evidence
        ),

        "fixed_average_quality_index": safe_average(
            fixed_quality
        ),

        "adaptive_average_quality_index": safe_average(
            adaptive_quality
        ),

        "average_quality_delta": (
            safe_average(adaptive_quality)
            - safe_average(fixed_quality)
        ),

        "average_evidence_delta": (
            safe_average(adaptive_evidence)
            - safe_average(fixed_evidence)
        ),

        "average_latency_delta_seconds": (
            safe_average(adaptive_latency)
            - safe_average(fixed_latency)
        )
    }


def main():
    print("Loading fixed-budget results...")
    fixed_results = extract_fixed_results()

    print("Loading adaptive results...")
    adaptive_results = extract_adaptive_results()

    comparisons = []

    for question, fixed in fixed_results.items():
        adaptive = adaptive_results.get(question)

        comparison = compare_question(
            question=question,
            fixed=fixed,
            adaptive=adaptive
        )

        comparisons.append(comparison)

    summary = aggregate(comparisons)

    report = {
        "summary": summary,
        "question_results": comparisons
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            report,
            file,
            indent=2
        )

    print()
    print("=" * 60)
    print("FINAL ADAPTIVE RAG EVALUATION")
    print("=" * 60)

    print(
        f"Questions: "
        f"{summary.get('questions', 0)}"
    )

    print(
        f"Fixed average context: "
        f"{summary.get('fixed_average_context_characters', 0):.1f}"
    )

    print(
        f"Adaptive average context: "
        f"{summary.get('adaptive_average_context_characters', 0):.1f}"
    )

    print(
        f"Context reduction: "
        f"{summary.get('context_reduction', 0) * 100:.1f}%"
    )

    print(
        f"Fixed evidence recall: "
        f"{summary.get('fixed_average_evidence_recall', 0):.3f}"
    )

    print(
        f"Adaptive evidence recall: "
        f"{summary.get('adaptive_average_evidence_recall', 0):.3f}"
    )

    print(
        f"Fixed quality index: "
        f"{summary.get('fixed_average_quality_index', 0):.3f}"
    )

    print(
        f"Adaptive quality index: "
        f"{summary.get('adaptive_average_quality_index', 0):.3f}"
    )

    print(
        f"Quality delta: "
        f"{summary.get('average_quality_delta', 0):+.3f}"
    )

    print(
        f"Latency delta: "
        f"{summary.get('average_latency_delta_seconds', 0):+.3f}s"
    )

    print()
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()