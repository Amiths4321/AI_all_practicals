import json
from datetime import datetime
import uuid
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
                print("Warning: Invalid queue format.")
                return []

            return queue

    except json.JSONDecodeError:

        print(
            "Warning: Queue file is corrupted."
        )

        return []


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


def add_to_review_queue(
    request_id,
    query,
    product,
    reason,
    answer=None,
    evidence=None
):

    queue = load_queue()

    # Check whether this request is already in the queue
    for item in queue:

        if (
            item.get("request_id") == request_id
            and item.get("status") == "PENDING"
        ):
            print(
                "\nReview already exists for request:",
                request_id
            )
            return item

    review_item = {

        "review_id": "REV-" + str(uuid.uuid4()),

        "request_id": request_id,

        "timestamp":
            datetime.now().isoformat(),

        "query": query,

        "product": product,

        "reason": reason,

        "answer": answer,

        "evidence": evidence or [],

        "status": "PENDING"
    }

    queue.append(review_item)

    save_queue(queue)

    return review_item