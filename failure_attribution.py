import json


DETAILS_FILE = "experiment_details.json"


def load_results():
    with open(DETAILS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def classify_failure(result):
    answerable = result["answerable"]
    gate_passed = result["gate_passed"]
    evidence_recall = result["evidence_recall"]

    if not answerable:
        if gate_passed:
            return "false_answer"
        return "correct_abstention"

    if evidence_recall == 0:
        return "retrieval_or_evidence_failure"

    if evidence_recall < 1.0:
        return "partial_evidence"

    if not gate_passed:
        return "over_abstention"

    return "retrieval_and_evidence_success"


def main():
    experiments = load_results()

    print("=" * 80)
    print("AUTOMATED FAILURE ATTRIBUTION")
    print("=" * 80)

    counts = {}

    for experiment in experiments:

        config = (
            f"chunk={experiment['chunk_size']}, "
            f"vector_k={experiment['vector_k']}, "
            f"hybrid_k={experiment['hybrid_k']}, "
            f"rerank_k={experiment['rerank_k']}, "
            f"threshold={experiment['evidence_threshold']}"
        )

        for result in experiment["question_results"]:

            category = classify_failure(result)

            counts[category] = counts.get(category, 0) + 1

            if category in {
                "retrieval_or_evidence_failure",
                "partial_evidence",
                "over_abstention",
                "false_answer"
            }:
                print()
                print("-" * 80)
                print(f"Question: {result['question']}")
                print(f"Failure:  {category}")
                print(f"Config:   {config}")
                print(
                    f"Evidence recall: "
                    f"{result['evidence_recall']:.3f}"
                )
                print(
                    f"Gate passed: "
                    f"{result['gate_passed']}"
                )

    print()
    print("=" * 80)
    print("FAILURE COUNTS")
    print("=" * 80)

    total = sum(counts.values())

    for category, count in sorted(counts.items()):
        percentage = count / total if total else 0

        print(
            f"{category:<35}"
            f"{count:>5} "
            f"({percentage:.1%})"
        )


if __name__ == "__main__":
    main()