import json


DETAILS_FILE = "experiment_details.json"


def load_results():
    with open(
        DETAILS_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def main():
    experiments = load_results()

    total = 0
    source_found = 0
    full_evidence = 0
    partial_evidence = 0
    no_evidence = 0

    for experiment in experiments:
        for result in experiment["question_results"]:

            if not result["answerable"]:
                continue

            total += 1

            expected_sources = set(
                result.get("expected_sources", [])
            )

            retrieved_sources = set(
                result.get("retrieved_sources", [])
            )

            if expected_sources & retrieved_sources:
                source_found += 1

            evidence_recall = result[
                "evidence_recall"
            ]

            if evidence_recall >= 1.0:
                full_evidence += 1
            elif evidence_recall > 0:
                partial_evidence += 1
            else:
                no_evidence += 1

    print("=" * 70)
    print("RETRIEVAL VS EVIDENCE ANALYSIS")
    print("=" * 70)

    print(f"Answerable cases: {total}")

    if total == 0:
        print("No answerable cases found.")
        return

    print()

    print(
        f"Expected source retrieved: "
        f"{source_found}/{total} "
        f"({source_found / total:.3f})"
    )

    print(
        f"Full evidence retrieved: "
        f"{full_evidence}/{total} "
        f"({full_evidence / total:.3f})"
    )

    print(
        f"Partial evidence: "
        f"{partial_evidence}/{total} "
        f"({partial_evidence / total:.3f})"
    )

    print(
        f"No evidence: "
        f"{no_evidence}/{total} "
        f"({no_evidence / total:.3f})"
    )


if __name__ == "__main__":
    main()