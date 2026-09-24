from collections import Counter


def summarize_failures(results):

    categories = Counter(
        result["failure_category"]
        for result in results
    )

    print("\nFAILURE ANALYSIS")
    print("=" * 60)

    total = len(results)

    for category, count in categories.most_common():

        percentage = (
            count / total * 100
            if total
            else 0
        )

        print(
            f"{category:30s}"
            f"{count:5d}"
            f" ({percentage:5.1f}%)"
        )


def print_failures(results):

    print("\nDETAILED FAILURES")
    print("=" * 60)

    for result in results:

        if result["failure_category"] == "success":
            continue

        print("\nQuestion:")
        print(result["question"])

        print(
            "Failure:",
            result["failure_category"]
        )

        print(
            "Evidence recall:",
            result["evidence_recall"]
        )

        print(
            "Retrieved:",
            result["retrieved_ids"]
        )

        print("-" * 60)

from failure_report import (
    summarize_failures,
    print_failures
)

summarize_failures(results)
print_failures(results)

def abstention_metrics(results):

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for result in results:

        expected = result["answerable"]

        predicted = result["gate_passed"]

        if expected and predicted:
            tp += 1

        elif not expected and predicted:
            fp += 1

        elif expected and not predicted:
            fn += 1

        elif not expected and not predicted:
            tn += 1

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall
    }  