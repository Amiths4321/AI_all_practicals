import json
from collections import Counter


DETAILS_FILE = "experiment_details.json"


def load_results():
    with open(
        DETAILS_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def classify_failure(result):
    answerable = result["answerable"]
    gate_passed = result["gate_passed"]
    evidence_recall = result["evidence_recall"]

    if answerable and gate_passed:
        if evidence_recall >= 1.0:
            return "supported_answer"
        return "partial_evidence"

    if answerable and not gate_passed:
        return "over_abstention"

    if not answerable and gate_passed:
        return "false_answer"

    if not answerable and not gate_passed:
        return "correct_abstention"

    return "unknown"


def main():
    experiments = load_results()

    overall_counts = Counter()

    for experiment in experiments:
        for result in experiment["question_results"]:
            category =      classify_failure (result)
            overall_counts[category] += 1

    print("=" * 70)
    print("PER-QUESTION FAILURE ANALYSIS")
    print("=" * 70)

    total = sum(overall_counts.values())

    print(f"Total evaluated cases: {total}")
    print()

    for category, count in sorted(overall_counts.items()):
        percentage = count / total if total else 0

        print(
            f"{category:<25} "
            f"{count:>5} "
            f"({percentage:.1%})"
        )


if __name__ == "__main__":
    main()