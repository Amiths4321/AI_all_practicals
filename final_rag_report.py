import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

EVALUATION_FILE = (
    BASE_DIR / "final_adaptive_evaluation.json"
)

ADAPTIVE_FILE = (
    BASE_DIR / "adaptive_rag_report.json"
)

OUTPUT_FILE = (
    BASE_DIR / "final_rag_report.json"
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def classify_result(item):
    evidence_delta = item.get(
        "evidence_delta",
        0.0
    )

    quality_delta = item.get(
        "quality_delta",
        0.0
    )

    context_reduction = item.get(
        "context_reduction",
        0.0
    )

    adaptive_abstained = item.get(
        "adaptive_abstained",
        False
    )

    if adaptive_abstained:
        if evidence_delta < 0:
            return "abstention_with_evidence_loss"

        return "abstention"

    if evidence_delta < 0:
        return "evidence_loss"

    if quality_delta < -0.05:
        return "quality_regression"

    if (
        context_reduction > 0.10
        and quality_delta >= -0.05
        and evidence_delta >= 0
    ):
        return "efficient_result"

    if (
        context_reduction <= 0.10
        and quality_delta >= 0
    ):
        return "no_material_saving"

    return "neutral"


def build_summary(evaluation, adaptive):
    summary = evaluation.get(
        "summary",
        {}
    )

    adaptive_summary = adaptive.get(
        "summary",
        {}
    )

    return {
        "questions": summary.get(
            "questions",
            0
        ),

        "fixed_budget": summary.get(
            "fixed_budget"
        ),

        "fixed_average_context_characters":
            summary.get(
                "fixed_average_context_characters",
                0.0
            ),

        "adaptive_average_context_characters":
            summary.get(
                "adaptive_average_context_characters",
                0.0
            ),

        "context_reduction":
            summary.get(
                "context_reduction",
                0.0
            ),

        "fixed_average_evidence_recall":
            summary.get(
                "fixed_average_evidence_recall",
                0.0
            ),

        "adaptive_average_evidence_recall":
            summary.get(
                "adaptive_average_evidence_recall",
                0.0
            ),

        "fixed_average_quality_index":
            summary.get(
                "fixed_average_quality_index",
                0.0
            ),

        "adaptive_average_quality_index":
            summary.get(
                "adaptive_average_quality_index",
                0.0
            ),

        "average_quality_delta":
            summary.get(
                "average_quality_delta",
                0.0
            ),

        "average_evidence_delta":
            summary.get(
                "average_evidence_delta",
                0.0
            ),

        "average_latency_delta_seconds":
            summary.get(
                "average_latency_delta_seconds",
                0.0
            ),

        "adaptive_answer_rate":
            adaptive_summary.get(
                "answer_rate",
                0.0
            ),

        "adaptive_abstention_rate":
            adaptive_summary.get(
                "abstention_rate",
                0.0
            )
    }


def main():

    print("Loading evaluation...")
    evaluation = load_json(
        EVALUATION_FILE
    )

    print("Loading adaptive report...")
    adaptive = load_json(
        ADAPTIVE_FILE
    )

    comparisons = evaluation.get(
        "question_results",
        []
    )

    question_reports = []

    classification_counts = {}

    for item in comparisons:

        classification = classify_result(
            item
        )

        classification_counts[
            classification
        ] = (
            classification_counts.get(
                classification,
                0
            ) + 1
        )

        question_reports.append({
            "question": item.get(
                "question"
            ),

            "classification":
                classification,

            "fixed_budget":
                item.get(
                    "fixed_budget"
                ),

            "adaptive_budget":
                item.get(
                    "adaptive_budget"
                ),

            "fixed_evidence_recall":
                item.get(
                    "fixed_evidence_recall",
                    0.0
                ),

            "adaptive_evidence_recall":
                item.get(
                    "adaptive_evidence_recall",
                    0.0
                ),

            "fixed_quality_index":
                item.get(
                    "fixed_quality_index",
                    0.0
                ),

            "adaptive_quality_index":
                item.get(
                    "adaptive_quality_index",
                    0.0
                ),

            "quality_delta":
                item.get(
                    "quality_delta",
                    0.0
                ),

            "context_reduction":
                item.get(
                    "context_reduction",
                    0.0
                ),

            "fixed_latency":
                item.get(
                    "fixed_latency",
                    0.0
                ),

            "adaptive_latency":
                item.get(
                    "adaptive_latency",
                    0.0
                ),

            "adaptive_decision_reason":
                item.get(
                    "adaptive_decision_reason"
                ),

            "adaptive_abstained":
                item.get(
                    "adaptive_abstained",
                    False
                )
        })

    report = {
        "summary": build_summary(
            evaluation,
            adaptive
        ),

        "classification_counts":
            classification_counts,

        "question_results":
            question_reports
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
    print("=" * 70)
    print("FINAL RAG REPORT")
    print("=" * 70)

    summary = report["summary"]

    print(
        "Questions:",
        summary["questions"]
    )

    print(
        "Fixed context:",
        round(
            summary[
                "fixed_average_context_characters"
            ],
            1
        )
    )

    print(
        "Adaptive context:",
        round(
            summary[
                "adaptive_average_context_characters"
            ],
            1
        )
    )

    print(
        "Context reduction:",
        round(
            summary["context_reduction"] * 100,
            1
        ),
        "%"
    )

    print(
        "Fixed evidence recall:",
        round(
            summary[
                "fixed_average_evidence_recall"
            ],
            3
        )
    )

    print(
        "Adaptive evidence recall:",
        round(
            summary[
                "adaptive_average_evidence_recall"
            ],
            3
        )
    )

    print(
        "Fixed quality:",
        round(
            summary[
                "fixed_average_quality_index"
            ],
            3
        )
    )

    print(
        "Adaptive quality:",
        round(
            summary[
                "adaptive_average_quality_index"
            ],
            3
        )
    )

    print(
        "Quality delta:",
        round(
            summary[
                "average_quality_delta"
            ],
            3
        )
    )

    print(
        "Evidence delta:",
        round(
            summary[
                "average_evidence_delta"
            ],
            3
        )
    )

    print()
    print("Classifications:")

    for name, count in sorted(
        classification_counts.items()
    ):
        print(
            f"  {name}: {count}"
        )

    print()
    print(
        f"Saved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()