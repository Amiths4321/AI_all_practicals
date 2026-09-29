import json
from pathlib import Path
from datetime import datetime
from collections import Counter


REVIEW_FILE = Path("manual_review_queue.json")


def load_queue():

    if not REVIEW_FILE.exists():
        return []

    try:

        with open(
            REVIEW_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            content = file.read().strip()

            if not content:
                return []

            queue = json.loads(content)

            if not isinstance(queue, list):
                return []

            return queue

    except json.JSONDecodeError:

        return []


def calculate_report(queue):

    total = len(queue)

    status_counts = Counter(
        item.get("status")
        for item in queue
    )

    reason_counts = Counter(
        item.get("reason")
        for item in queue
    )

    resolution_times = []

    for item in queue:

        created_at = item.get("timestamp")
        resolved_at = item.get("resolved_at")

        if not created_at or not resolved_at:
            continue

        try:

            created = datetime.fromisoformat(
                created_at
            )

            resolved = datetime.fromisoformat(
                resolved_at
            )

            seconds = (
                resolved - created
            ).total_seconds()

            resolution_times.append(seconds)

        except ValueError:
            continue

    if resolution_times:

        average_resolution_seconds = (
            sum(resolution_times)
            / len(resolution_times)
        )

    else:

        average_resolution_seconds = 0.0

    return {
        "total_reviews": total,

        "pending": status_counts.get(
            "PENDING",
            0
        ),

        "approved": status_counts.get(
            "APPROVED",
            0
        ),

        "rejected": status_counts.get(
            "REJECTED",
            0
        ),

        "review_reasons": dict(
            reason_counts
        ),

        "resolved_reviews": len(
            resolution_times
        ),

        "average_resolution_minutes":
            round(
                average_resolution_seconds / 60,
                2
            )
    }


if __name__ == "__main__":

    queue = load_queue()

    report = calculate_report(queue)

    report["generated_at"] = datetime.now().isoformat()

    output_file = Path("review_operations_report.json")

    with open(
          output_file,
          "w",
          encoding="utf-8"
          ) as file:

          json.dump(
          report,
          file,
          indent=4,
          ensure_ascii=False
          )

    print(
          "\nReport saved:",
          output_file
          )

    print("\n==============================")
    print("REVIEW OPERATIONS REPORT")
    print("==============================")

    print(
        "Total Reviews:",
        report["total_reviews"]
    )

    print(
        "Pending:",
        report["pending"]
    )

    print(
        "Approved:",
        report["approved"]
    )

    print(
        "Rejected:",
        report["rejected"]
    )

    print(
        "Resolved:",
        report["resolved_reviews"]
    )

    print(
        "Average Resolution Time:",
        report["average_resolution_minutes"],
        "minutes"
    )

    print("\nReview Reasons:")

    for reason, count in report[
        "review_reasons"
    ].items():

        print(
            " ",
            reason,
            ":",
            count
        )

    print("==============================")