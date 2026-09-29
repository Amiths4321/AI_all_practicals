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

            return json.loads(content)

    except json.JSONDecodeError:
        print("Queue file contains invalid JSON.")
        return []


queue = load_queue()

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


print("\n==============================")
print("REVIEW QUEUE SUMMARY")
print("==============================")

print("Total reviews: ", len(queue))
print("Pending:       ", pending)
print("Approved:      ", approved)
print("Rejected:      ", rejected)

print("==============================")