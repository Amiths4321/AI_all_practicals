import json

from pathlib import Path
from datetime import datetime


REVIEW_FILE = Path("manual_review_queue.json")


# ==============================
# LOAD REVIEW QUEUE
# ==============================

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
                print("Invalid queue format.")
                return []

            return queue

    except json.JSONDecodeError:

        print("Queue file contains invalid JSON.")
        return []


# ==============================
# SAVE REVIEW QUEUE
# ==============================

def save_queue(queue):

    with open(
        REVIEW_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            queue,
            file,
            indent=4,
            ensure_ascii=False
        )


# ==============================
# RESOLVE REVIEW
# ==============================

def resolve_review(
    review_id,
    decision,
    reviewer,
    reviewer_comment=""
):

    if decision not in ["APPROVED", "REJECTED"]:

        print("Invalid decision.")
        return

    queue = load_queue()

    if not queue:

        print("Review queue is empty.")
        return

    resolved_item = None

    for item in queue:

        if item.get("review_id") == review_id:

            if item.get("status") != "PENDING":

                print("Review is already resolved.")
                return

            resolved_at = datetime.now().isoformat()

            item["status"] = decision
            item["reviewer"] = reviewer
            item["reviewer_comment"] = reviewer_comment
            item["resolved_at"] = resolved_at

            item["resolution"] = {

                "decision": decision,

                "reviewer": reviewer,

                "comment": reviewer_comment,

                "resolved_at": resolved_at
            }

            resolved_item = item

            break

    if resolved_item is None:

        print("Review ID not found.")
        return

    save_queue(queue)

    print("\n==============================")
    print("REVIEW RESOLVED")
    print("==============================")

    print(
        "Review ID:",
        resolved_item["review_id"]
    )

    print(
        "Decision:",
        resolved_item["status"]
    )

    print(
        "Reviewer:",
        resolved_item["reviewer"]
    )

    print(
        "Resolved At:",
        resolved_item["resolved_at"]
    )

    print(
        "Comment:",
        resolved_item["reviewer_comment"]
    )


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    review_id = input(
        "Enter Review ID: "
    ).strip()

    decision = input(
        "Enter decision (APPROVED/REJECTED): "
    ).strip().upper()

    reviewer = input(
        "Enter reviewer name: "
    ).strip()

    comment = input(
        "Enter reviewer comment: "
    ).strip()

    resolve_review(
        review_id,
        decision,
        reviewer,
        comment
    )