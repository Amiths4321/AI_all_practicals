from rag_pipeline import run_rag


TEST_CASES = [
    {
        "question": "What is the minimum income required for a home loan?",
        "product": "home_loan"
    },
    {
        "question": "What is the minimum age for a home loan?",
        "product": "home_loan"
    },
    {
        "question": "What documents are required for a home loan?",
        "product": "home_loan"
    },
    {
        "question": "What is the minimum income required for a personal loan?",
        "product": "personal_loan"
    },
    {
        "question": "What is the maximum personal loan tenure?",
        "product": "personal_loan"
    }
]


print("\n======================================")
print("FINAL END-TO-END RAG TEST")
print("======================================")


passed = 0
review = 0


for number, test in enumerate(TEST_CASES, start=1):

    print(f"\nTEST {number}")
    print("--------------------------------------")

    query = test["question"]
    product = test["product"]

    print("Question:", query)
    print("Product:", product)

    try:

        result = run_rag(
            query,
            product,
            conversation_history=[]
        )

        print("\nAnswer:")
        print(result["answer"])

        print(
            "\nStatus:",
            result.get("status")
        )

        print(
            "Review Required:",
            result.get("review_required")
        )

        print(
            "Confidence:",
            result.get("confidence")
        )

        if result.get("review_required"):

            review += 1
            print("\nRESULT: REVIEW")

        else:

            passed += 1
            print("\nRESULT: PASSED")

    except Exception as e:

        review += 1

        print("\nRESULT: FAILED")
        print("Error:", e)


print("\n======================================")
print("FINAL SUMMARY")
print("======================================")

print("Total tests:", len(TEST_CASES))
print("Passed:", passed)
print("Review/Failed:", review)

print(
    "Success Rate:",
    round(
        passed / len(TEST_CASES) * 100,
        2
    ),
    "%"
)

print("\n======================================")
print("END-TO-END TEST COMPLETE")
print("======================================")