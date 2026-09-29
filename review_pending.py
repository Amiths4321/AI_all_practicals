import json
from pathlib import Path

REVIEW_FILE = Path("manual_review_queue.json")

print("\n==============================")
print("PENDING HUMAN REVIEWS")
print("==============================")

if not REVIEW_FILE.exists():
    print("No review queue found.")
    exit()

with open(REVIEW_FILE, "r", encoding="utf-8") as file:
    queue = json.load(file)

pending = [
    item
    for item in queue
    if item.get("status") == "PENDING"
]

print("Pending reviews:", len(pending))

for item in pending:
    print("\n------------------------------")
    print("Review ID:", item["review_id"])
    print("Request ID:", item["request_id"])
    print("Product:", item["product"])
    print("Reason:", item["reason"])
    print("Query:", item["query"])
    print("Answer:", item["answer"])

    print("\nEvidence:")
    if item["evidence"]:
        for evidence in item["evidence"]:
            print(
                evidence.get("evidence_id"),
                "-",
                evidence.get("document"),
                "-",
                evidence.get("section")
            )
    else:
        print("No evidence available.")