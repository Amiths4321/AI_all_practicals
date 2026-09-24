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

    print("=" * 80)
    print("CONTEXT COMPRESSION ANALYSIS")
    print("=" * 80)

    totals = {}

    for experiment in experiments:

        for result in experiment[
            "question_results"
        ]:

            compression = result.get(
                "context_compression"
            )

            if not compression:
                continue

            documents = compression[
                "compressed_documents"
            ]

            if documents not in totals:
                totals[documents] = {
                    "count": 0,
                    "reduction": 0.0,
                    "chars_before": 0,
                    "chars_after": 0
                }

            totals[documents]["count"] += 1

            totals[documents]["reduction"] += (
                compression[
                    "compression_ratio"
                ]
            )

            totals[documents]["chars_before"] += (
                compression[
                    "original_chars"
                ]
            )

            totals[documents]["chars_after"] += (
                compression[
                    "compressed_chars"
                ]
            )

    for documents in sorted(totals):

        data = totals[documents]

        count = data["count"]

        average_reduction = (
            data["reduction"] / count
        )

        print()
        print(
            f"Documents kept: {documents}"
        )

        print(
            f"Average context reduction: "
            f"{average_reduction:.1%}"
        )

        print(
            f"Characters before: "
            f"{data['chars_before']}"
        )

        print(
            f"Characters after: "
            f"{data['chars_after']}"
        )


if __name__ == "__main__":
    main()