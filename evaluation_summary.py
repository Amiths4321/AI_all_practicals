import json
from pathlib import Path


files = sorted(
    Path(".").glob("batch_evaluation_*.json")
)

if not files:
    print("No batch evaluation report found.")
    exit()


latest_file = files[-1]

with open(
    latest_file,
    "r",
    encoding="utf-8"
) as file:

    report = json.load(file)


tests = report["tests"]

total = len(tests)

retrieval_passed = 0
answer_passed = 0
overall_passed = 0


for test in tests:

    if (
        test["retrieval"]["status"]
        == "PASSED"
    ):
        retrieval_passed += 1

    if test["answer_evaluation"]["correct"]:
        answer_passed += 1

    if test["result"] == "PASSED":
        overall_passed += 1


retrieval_rate = (
    retrieval_passed / total * 100
)

answer_rate = (
    answer_passed / total * 100
)

overall_rate = (
    overall_passed / total * 100
)


print("\n======================================")
print("RAG EVALUATION DASHBOARD")
print("======================================")

print("\nReport:")
print(latest_file.name)

print("\n------------------------------")
print("DATASET")
print("------------------------------")

print("Total tests:", total)

print("\n------------------------------")
print("RETRIEVAL")
print("------------------------------")

print(
    "Retrieval passed:",
    retrieval_passed
)

print(
    "Retrieval pass rate:",
    round(retrieval_rate, 2),
    "%"
)

print("\n------------------------------")
print("ANSWER")
print("------------------------------")

print(
    "Answer passed:",
    answer_passed
)

print(
    "Answer pass rate:",
    round(answer_rate, 2),
    "%"
)

print("\n------------------------------")
print("OVERALL")
print("------------------------------")

print(
    "Overall passed:",
    overall_passed
)

print(
    "Overall pass rate:",
    round(overall_rate, 2),
    "%"
)

print("\n======================================")
print("EVALUATION COMPLETE")
print("======================================")