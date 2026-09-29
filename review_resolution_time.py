import json
from pathlib import Path
from datetime import datetime

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


def calculate_resolution_times(queue):

    resolution_times = []

    for item in queue:

        created_at = item.get("timestamp")
        resolved_at = item.get("resolved_at")

        # Only completed reviews have a resolution time
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

            resolution_times.append({
                "review_id": item.get("review_id"),
                "resolution_seconds": seconds,
                "resolution_minutes": seconds / 60
            })

        except ValueError:

            print(
                "Invalid timestamp for:",
                item.get("review_id")
            )

    return resolution_times


def calculate_average(resolution_times):

    if not resolution_times:
        return 0.0

    total_seconds = sum(
        item["resolution_seconds"]
        for item in resolution_times
    )

    return total_seconds / len(resolution_times)


if __name__ == "__main__":

    queue = load_queue()

    resolution_times = calculate_resolution_times(
        queue
    )

    average_seconds = calculate_average(
        resolution_times
    )

    print("\n==============================")
    print("REVIEW RESOLUTION TIME")
    print("==============================")

    print(
        "Resolved Reviews:",
        len(resolution_times)
    )

    for item in resolution_times:

        print("\nReview ID:",
              item["review_id"])

        print(
            "Resolution Time:",
            round(
                item["resolution_minutes"],
                2
            ),
            "minutes"
        )

    print("\n------------------------------")

    print(
        "Average Resolution Time:",
        round(
            average_seconds / 60,
            2
        ),
        "minutes"
    )

    print("==============================")