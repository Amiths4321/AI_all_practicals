import json
from pathlib import Path


AUDIT_FOLDER = Path(".")
OUTPUT_FILE = Path("audit_metrics.json")


def load_audit_files():

    audits = []

    for file_path in AUDIT_FOLDER.glob("audit_*.json"):

        try:

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(file)

                if isinstance(data, dict):
                    audits.append(data)

        except json.JSONDecodeError:

            print(
                "Skipping invalid file:",
                file_path.name
            )

    return audits


def average(values):

    if not values:
        return 0.0

    return sum(values) / len(values)


def calculate_metrics(audits):

    total = len(audits)

    completed = 0
    manual_review = 0

    retrieval_times = []
    reranking_times = []
    llm_times = []
    total_times = []

    citation_valid = 0
    citation_invalid = 0

    for audit in audits:

        decision = audit.get(
            "decision",
            {}
        )

        review_required = decision.get(
            "review_required",
            False
        )

        if review_required:

            manual_review += 1

        else:

            completed += 1

        performance = audit.get(
            "performance",
            {}
        )

        if performance.get(
            "retrieval_seconds"
        ) is not None:

            retrieval_times.append(
                float(
                    performance[
                        "retrieval_seconds"
                    ]
                )
            )

        if performance.get(
            "reranking_seconds"
        ) is not None:

            reranking_times.append(
                float(
                    performance[
                        "reranking_seconds"
                    ]
                )
            )

        if performance.get(
            "llm_seconds"
        ) is not None:

            llm_times.append(
                float(
                    performance[
                        "llm_seconds"
                    ]
                )
            )

        if performance.get(
            "total_seconds"
        ) is not None:

            total_times.append(
                float(
                    performance[
                        "total_seconds"
                    ]
                )
            )

        citation = audit.get(
            "citation_validation",
            {}
        )

        if citation.get("valid") is True:

            citation_valid += 1

        elif citation:

            citation_invalid += 1

    return {

        "total_audits": total,

        "completed": completed,

        "manual_review": manual_review,

        "performance": {

            "average_retrieval_seconds":
                round(
                    average(
                        retrieval_times
                    ),
                    4
                ),

            "average_reranking_seconds":
                round(
                    average(
                        reranking_times
                    ),
                    4
                ),

            "average_llm_seconds":
                round(
                    average(
                        llm_times
                    ),
                    4
                ),

            "average_total_seconds":
                round(
                    average(
                        total_times
                    ),
                    4
                )
        },

        "citation": {

            "valid": citation_valid,

            "invalid": citation_invalid
        }
    }


if __name__ == "__main__":

    audits = load_audit_files()

    metrics = calculate_metrics(audits)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n==============================")
    print("AUDIT METRICS")
    print("==============================")

    print(
        "Total Audits:",
        metrics["total_audits"]
    )

    print(
        "Completed:",
        metrics["completed"]
    )

    print(
        "Manual Review:",
        metrics["manual_review"]
    )

    print(
        "Avg Retrieval:",
        metrics["performance"][
            "average_retrieval_seconds"
        ],
        "seconds"
    )

    print(
        "Avg Reranking:",
        metrics["performance"][
            "average_reranking_seconds"
        ],
        "seconds"
    )

    print(
        "Avg LLM:",
        metrics["performance"][
            "average_llm_seconds"
        ],
        "seconds"
    )

    print(
        "Avg Total:",
        metrics["performance"][
            "average_total_seconds"
        ],
        "seconds"
    )

    print(
        "Citation Valid:",
        metrics["citation"]["valid"]
    )

    print(
        "Citation Invalid:",
        metrics["citation"]["invalid"]
    )

    print("==============================")

    print(
        "\nSaved:",
        OUTPUT_FILE
    )