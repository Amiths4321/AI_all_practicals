import json


INPUT_FILE = "context_strategy_benchmark.json"
OUTPUT_FILE = "context_failure_analysis.json"


def classify_failure(
    retrieval_evidence_recall,
    strategy_evidence_recall,
    judgement
):
    """
    Attribute the failure to the earliest stage
    that can explain it.
    """

    # ---------------------------------------------------------
    # Stage 1: Retrieval
    # ---------------------------------------------------------

    if retrieval_evidence_recall < 1.0:
        return "retrieval_failure"

    # ---------------------------------------------------------
    # Stage 2: Context construction
    # ---------------------------------------------------------

    if strategy_evidence_recall < 1.0:
        return "context_loss"

    # ---------------------------------------------------------
    # Stage 3: Generation
    # ---------------------------------------------------------

    if judgement["unsupported_claims"] > 0:
        return "unsupported_claims"

    if judgement["groundedness"] < 2:
        return "generation_ungrounded"

    if judgement["completeness"] < 2:
        return "generation_incomplete"

    if judgement["relevance"] < 2:
        return "generation_irrelevant"

    return "success"


def load_results():
    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def main():

    results = load_results()

    failure_results = []

    for result in results:

        question = result["question"]

        strategies = result["strategies"]

        retrieval_evidence_recall = result.get(
            "retrieval_evidence_recall",
            0.0
        )

        for strategy, data in strategies.items():

            judgement = data.get("judgement")

            if not judgement:
                continue

            strategy_evidence_recall = data.get(
                "strategy_evidence_recall",
                0.0
            )

            failure_type = classify_failure(
                retrieval_evidence_recall=(
                    retrieval_evidence_recall
                ),
                strategy_evidence_recall=(
                    strategy_evidence_recall
                ),
                judgement=judgement
            )

            failure_results.append({
                "question": question,
                "strategy": strategy,

                "retrieval_evidence_recall":
                    retrieval_evidence_recall,

                "strategy_evidence_recall":
                    strategy_evidence_recall,

                "failure_type":
                    failure_type,

                "groundedness":
                    judgement["groundedness"],

                "relevance":
                    judgement["relevance"],

                "completeness":
                    judgement["completeness"],

                "unsupported_claims":
                    judgement["unsupported_claims"],

                "reason":
                    judgement.get("reason", "")
            })

    # ---------------------------------------------------------
    # Aggregate counts
    # ---------------------------------------------------------

    overall_counts = {}

    for item in failure_results:

        failure_type = item["failure_type"]

        overall_counts[failure_type] = (
            overall_counts.get(
                failure_type,
                0
            ) + 1
        )

    # ---------------------------------------------------------
    # Per-strategy counts
    # ---------------------------------------------------------

    strategy_counts = {}

    for item in failure_results:

        strategy = item["strategy"]

        if strategy not in strategy_counts:
            strategy_counts[strategy] = {}

        failure_type = item["failure_type"]

        strategy_counts[strategy][failure_type] = (
            strategy_counts[strategy].get(
                failure_type,
                0
            ) + 1
        )

    report = {
        "total_evaluations": len(
            failure_results
        ),

        "overall_failure_counts":
            overall_counts,

        "strategy_failure_counts":
            strategy_counts,

        "question_results":
            failure_results
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False
        )

    # ---------------------------------------------------------
    # Console report
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print("CONTEXT STRATEGY FAILURE ANALYSIS")
    print("=" * 80)

    print()
    print("Overall:")
    print("-" * 40)

    for failure_type, count in sorted(
        overall_counts.items()
    ):
        print(
            f"{failure_type:25} {count}"
        )

    print()
    print("By strategy:")
    print("-" * 40)

    for strategy, counts in strategy_counts.items():

        print()
        print(strategy)

        for failure_type, count in sorted(
            counts.items()
        ):
            print(
                f"  {failure_type:23} {count}"
            )

    print()
    print(
        f"Saved results to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()