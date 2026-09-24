import json
from collections import defaultdict


RESULTS_FILE = "experiment_results.json"


def load_results():
    with open(RESULTS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def print_summary(results):
    print("=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    print(f"Total experiments: {len(results)}")

    if not results:
        return

    runtimes = [r["runtime_seconds"] for r in results]

    evidence_recalls = [
        r["average_evidence_recall"]
        for r in results
    ]

    false_answer_rates = [
        r["false_answer_rate"]
        for r in results
    ]

    print(
        f"Average evidence recall: "
        f"{sum(evidence_recalls) / len(evidence_recalls):.3f}"
    )

    print(
        f"Average false-answer rate: "
        f"{sum(false_answer_rates) / len(false_answer_rates):.3f}"
    )

    print(
        f"Average runtime: "
        f"{sum(runtimes) / len(runtimes):.3f}s"
    )


def group_by_parameter(results, parameter):
    groups = defaultdict(list)

    for result in results:
        groups[result[parameter]].append(result)

    return groups


def print_parameter_analysis(results, parameter):
    print("\n" + "=" * 70)
    print(f"ANALYSIS BY {parameter}")
    print("=" * 70)

    groups = group_by_parameter(results, parameter)

    for value in sorted(groups):
        group = groups[value]

        evidence = sum(
            r["average_evidence_recall"]
            for r in group
        ) / len(group)

        false_answers = sum(
            r["false_answer_rate"]
            for r in group
        ) / len(group)

        runtime = sum(
            r["runtime_seconds"]
            for r in group
        ) / len(group)

        print(
            f"{parameter}={value:<8} "
            f"evidence_recall={evidence:.3f} "
            f"false_answer_rate={false_answers:.3f} "
            f"runtime={runtime:.3f}s"
        )


def print_configurations(results):
    print("\n" + "=" * 70)
    print("CONFIGURATIONS")
    print("=" * 70)

    for index, result in enumerate(results, start=1):
        print(
            f"{index:02d}. "
            f"chunk={result['chunk_size']}, "
            f"overlap={result['overlap']}, "
            f"vector_k={result['vector_k']}, "
            f"hybrid_k={result['hybrid_k']}, "
            f"rerank_k={result['rerank_k']}, "
            f"threshold={result['evidence_threshold']:.2f} | "
            f"evidence={result['average_evidence_recall']:.3f} | "
            f"false={result['false_answer_rate']:.3f} | "
            f"runtime={result['runtime_seconds']:.3f}s"
        )


def main():
    results = load_results()

    print_summary(results)

    print_parameter_analysis(
        results,
        "chunk_size"
    )

    print_parameter_analysis(
        results,
        "vector_k"
    )

    print_parameter_analysis(
        results,
        "evidence_threshold"
    )

    print_configurations(results)


if __name__ == "__main__":
    main()