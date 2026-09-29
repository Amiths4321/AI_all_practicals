import json
from pathlib import Path

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


def calculate_analytics(queue):

    total = len(queue)

    pending = 0
    approved = 0
    rejected = 0

    for item in queue:

        status = item.get("status")

        if status == "PENDING":
            pending += 1

        elif status == "APPROVED":
            approved += 1

        elif status == "REJECTED":
            rejected += 1

    if total > 0:

        pending_rate = pending / total
        approval_rate = approved / total
        rejection_rate = rejected / total

    else:

        pending_rate = 0.0
        approval_rate = 0.0
        rejection_rate = 0.0

    return {
        "total": total,
        "pending": pending,
        "approved": approved,
        "rejected": rejected,
        "pending_rate": pending_rate,
        "approval_rate": approval_rate,
        "rejection_rate": rejection_rate
    }


if __name__ == "__main__":

    queue = load_queue()

    analytics = calculate_analytics(queue)

    print("\n==============================")
    print("REVIEW ANALYTICS")
    print("==============================")

    print("Total Reviews:",
          analytics["total"])

    print("Pending:",
          analytics["pending"])

    print("Approved:",
          analytics["approved"])

    print("Rejected:",
          analytics["rejected"])

    print(
        "Pending Rate:",
        round(
            analytics["pending_rate"] * 100,
            2
        ),
        "%"
    )

    print(
        "Approval Rate:",
        round(
            analytics["approval_rate"] * 100,
            2
        ),
        "%"
    )

    print(
        "Rejection Rate:",
        round(
            analytics["rejection_rate"] * 100,
            2
        ),
        "%"
    )

    print("==============================")