import json


RESULTS_FILE = "experiment_results.json"


def load_results():
    with open(RESULTS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def classify_experiment(result):
    evidence_recall = result["average_evidence_recall"]
    false_answer_rate = result["false_answer_rate"]

    if false_answer_rate > 0:
        return "false_answers_present"

    if evidence_recall < 1.0:
        return "evidence_missing"

    return "no_detected_failure"


def main():
    results = load_results()

    if not results:
        print("No experiment results found.")
        return

    counts = {}

    for result in results:
        category = classify_experiment(result)
        counts[category] = counts.get(category, 0) + 1

    print("=" * 70)
    print("EXPERIMENT ERROR SUMMARY")
    print("=" * 70)

    total = len(results)

    for category, count in sorted(counts.items()):
        percentage = count / total

        print(
            f"{category:<25} "
            f"{count:>4} "
            f"({percentage:.1%})"
        )

    print()
    print("=" * 70)
    print("THRESHOLD / CONFIGURATION DETAILS")
    print("=" * 70)

    for index, result in enumerate(results, start=1):
        print(
            f"{index:02d}. "
            f"chunk={result['chunk_size']}, "
            f"vector_k={result['vector_k']}, "
            f"hybrid_k={result['hybrid_k']}, "
            f"rerank_k={result['rerank_k']}, "
            f"threshold={result['evidence_threshold']:.2f} | "
            f"evidence={result['average_evidence_recall']:.3f} | "
            f"false={result['false_answer_rate']:.3f}"
        )


if __name__ == "__main__":
    main()