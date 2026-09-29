import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR / "adaptive_rag_report.json"
)

OUTPUT_FILE = (
    BASE_DIR / "failure_monitoring.json"
)


def classify_result(result):

    # --------------------------------------------------
    # Infrastructure / processing error
    # --------------------------------------------------

    if result.get("status") == "error":
        return "processing_error"

    # --------------------------------------------------
    # Retrieval evidence
    # --------------------------------------------------

    retrieval_recall = result.get(
        "retrieval_evidence_recall",
        0.0
    )

    if retrieval_recall < 1.0:
        return "retrieval_failure"

    # --------------------------------------------------
    # Context evidence
    # --------------------------------------------------

    context_recall = result.get(
        "context_evidence_recall",
        0.0
    )

    if context_recall < 1.0:
        return "context_loss"

    # --------------------------------------------------
    # Explicit abstention
    # --------------------------------------------------

    if result.get("status") == "abstained":
        return "abstention"

    # --------------------------------------------------
    # Generation / judge failures
    # --------------------------------------------------

    judgement = result.get(
        "judgement"
    )

    if judgement is None:
        return "missing_judgement"

    unsupported_claims = judgement.get(
        "unsupported_claims",
        0
    )

    if unsupported_claims > 0:
        return "unsupported_claims"

    groundedness = judgement.get(
        "groundedness",
        0
    )

    if groundedness < 2:
        return "generation_ungrounded"

    relevance = judgement.get(
        "relevance",
        0
    )

    if relevance < 2:
        return "generation_irrelevant"

    completeness = judgement.get(
        "completeness",
        0
    )

    if completeness < 2:
        return "generation_incomplete"

    return "success"


def load_report():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def main():

    report = load_report()

    results = report.get(
        "results",
        []
    )

    if not results:
        print(
            "No question results found."
        )
        return

    monitored_results = []

    counts = {}

    for result in results:

        category = classify_result(
            result
        )

        counts[category] = (
            counts.get(category, 0)
            + 1
        )

        monitored_results.append({
            "question": result.get(
                "question"
            ),

            "status": result.get(
                "status"
            ),

            "category": category,

            "budget": result.get(
                "budget"
            ),

            "retrieval_evidence_recall":
                result.get(
                    "retrieval_evidence_recall",
                    0.0
                ),

            "context_evidence_recall":
                result.get(
                    "context_evidence_recall",
                    0.0
                ),

            "judgement": result.get(
                "judgement"
            ),

            "reason": result.get(
                "reason"
            )
        })

    total = len(results)

    monitoring_report = {
        "summary": {
            "questions": total,

            "successes": counts.get(
                "success",
                0
            ),

            "retrieval_failures":
                counts.get(
                    "retrieval_failure",
                    0
                ),

            "context_losses":
                counts.get(
                    "context_loss",
                    0
                ),

            "abstentions":
                counts.get(
                    "abstention",
                    0
                ),

            "processing_errors":
                counts.get(
                    "processing_error",
                    0
                ),

            "unsupported_claims":
                counts.get(
                    "unsupported_claims",
                    0
                ),

            "ungrounded_answers":
                counts.get(
                    "generation_ungrounded",
                    0
                ),

            "irrelevant_answers":
                counts.get(
                    "generation_irrelevant",
                    0
                ),

            "incomplete_answers":
                counts.get(
                    "generation_incomplete",
                    0
                ),

            "missing_judgements":
                counts.get(
                    "missing_judgement",
                    0
                ),

            "category_counts":
                counts,

            "success_rate": (
                counts.get(
                    "success",
                    0
                ) / total
            ),

            "failure_rate": (
                1
                - (
                    counts.get(
                        "success",
                        0
                    ) / total
                )
            )
        },

        "question_results":
            monitored_results
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            monitoring_report,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 70)
    print("RAG FAILURE MONITOR")
    print("=" * 70)

    print(
        f"Questions: {total}"
    )

    print(
        f"Successes: "
        f"{counts.get('success', 0)}"
    )

    print(
        f"Retrieval failures: "
        f"{counts.get('retrieval_failure', 0)}"
    )

    print(
        f"Context losses: "
        f"{counts.get('context_loss', 0)}"
    )

    print(
        f"Abstentions: "
        f"{counts.get('abstention', 0)}"
    )

    print(
        f"Processing errors: "
        f"{counts.get('processing_error', 0)}"
    )

    print(
        f"Unsupported claims: "
        f"{counts.get('unsupported_claims', 0)}"
    )

    print(
        f"Ungrounded answers: "
        f"{counts.get('generation_ungrounded', 0)}"
    )

    print(
        f"Incomplete answers: "
        f"{counts.get('generation_incomplete', 0)}"
    )

    print(
        f"Irrelevant answers: "
        f"{counts.get('generation_irrelevant', 0)}"
    )

    print()
    print(
        f"Success rate: "
        f"{monitoring_report['summary']['success_rate']:.3f}"
    )

    print(
        f"Failure rate: "
        f"{monitoring_report['summary']['failure_rate']:.3f}"
    )

    print()
    print(
        f"Saved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()