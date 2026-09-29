import json
from datetime import datetime

from hybrid_retriever import hybrid_retrieve
from reranker import rerank
from failure_analysis import analyze_retrieval_failure
from answer_evaluator import evaluate_answer


TEST_CASES = [
    {
        "question": "What is the minimum income required for a home loan?",
        "product": "home_loan",
        "expected_document": "banking_policy_v2.txt",
        "expected_section": "Eligibility",
        "expected_phrases": ["₹60,000"]
    },
    {
        "question": "What is the minimum age for a home loan?",
        "product": "home_loan",
        "expected_document": "banking_policy_v2.txt",
        "expected_section": "Eligibility",
        "expected_phrases": ["21"]
    },
    {
        "question": "What documents are required for a home loan?",
        "product": "home_loan",
        "expected_document": "banking_policy_v2.txt",
        "expected_section": "Documents",
        "expected_phrases": [
            "identity proof",
            "address proof",
            "income proof"
        ]
    },
    {
        "question": "What is the minimum income required for a personal loan?",
        "product": "personal_loan",
        "expected_document": "personal_loan_policy.txt",
        "expected_section": "Eligibility",
        "expected_phrases": ["₹30,000"]
    },
    {
        "question": "What is the maximum personal loan tenure?",
        "product": "personal_loan",
        "expected_document": "personal_loan_policy.txt",
        "expected_section": "Loan Tenure",
        "expected_phrases": ["5 years"]
    }
]


evaluation_results = []

passed = 0
failed = 0


print("\n==============================")
print("BATCH RAG EVALUATION")
print("==============================")


for number, test in enumerate(TEST_CASES, start=1):

    query = test["question"]
    product = test["product"]

    print(f"\nTEST {number}")
    print("------------------------------")
    print("Question:", query)

    # Retrieval
    candidates = hybrid_retrieve(
        query,
        product
    )

    # Reranking
    results = (
        rerank(query, candidates, top_k=3)
        if candidates
        else []
    )

    # Retrieval failure analysis
    retrieval_analysis = analyze_retrieval_failure(
        candidates,
        results,
        test["expected_document"],
        test["expected_section"]
    )

    # Build a simple evidence context
    context = ""

    for result in results:

        document = result.get(
            "document",
            {}
        )

        context += (
            document.get("text", "")
            + "\n"
        )

    # Simulated answer for evaluation.
    # We are testing the evaluation pipeline here.
    answer = context

    # Answer evaluation
    answer_result = evaluate_answer(
        answer,
        test["expected_phrases"]
    )

    test_passed = (
        retrieval_analysis["status"] == "PASSED"
        and answer_result["correct"]
    )

    if test_passed:
        passed += 1
        status = "PASSED"
    else:
        failed += 1
        status = "FAILED"

    print("Retrieval:", retrieval_analysis["status"])
    print("Answer:", answer_result["correct"])
    print("Result:", status)

    evaluation_results.append({
        "test_number": number,
        "question": query,
        "product": product,
        "retrieval": retrieval_analysis,
        "answer_evaluation": answer_result,
        "result": status
    })


# Final report

report = {
    "timestamp": datetime.now().isoformat(),
    "dataset_size": len(TEST_CASES),
    "summary": {
        "passed": passed,
        "failed": failed,
        "pass_rate": round(
            passed / len(TEST_CASES) * 100,
            2
        )
    },
    "tests": evaluation_results
}


filename = (
    "batch_evaluation_"
    + datetime.now().strftime("%Y%m%d_%H%M%S")
    + ".json"
)


with open(
    filename,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        report,
        file,
        indent=4,
        ensure_ascii=False
    )


print("\n==============================")
print("BATCH SUMMARY")
print("==============================")

print("Total:", len(TEST_CASES))
print("Passed:", passed)
print("Failed:", failed)

print(
    "Pass Rate:",
    report["summary"]["pass_rate"],
    "%"
)

print("\nReport saved:", filename)