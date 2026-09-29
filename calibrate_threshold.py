from hybrid_retriever import hybrid_retrieve
from reranker import rerank


TEST_CASES = [

    {
        "question": "What is the minimum income required for a home loan?",
        "product": "home_loan",
        "expected_document": "banking_policy_v2.txt",
        "expected_section": "Eligibility"
    },

    {
        "question": "What is the minimum age for a home loan?",
        "product": "home_loan",
        "expected_document": "banking_policy_v2.txt",
        "expected_section": "Eligibility"
    },

    {
        "question": "What documents are required for a home loan?",
        "product": "home_loan",
        "expected_document": "banking_policy_v2.txt",
        "expected_section": "Documents"
    },

    {
        "question": "What is the minimum income required for a personal loan?",
        "product": "personal_loan",
        "expected_document": "personal_loan_policy.txt",
        "expected_section": "Eligibility"
    },

    {
        "question": "What is the maximum personal loan tenure?",
        "product": "personal_loan",
        "expected_document": "personal_loan_policy.txt",
        "expected_section": "Loan Tenure"
    }
]


relevant_scores = []
irrelevant_scores = []


print("\n==============================")
print("RERANKER SCORE CALIBRATION")
print("==============================")


for number, test in enumerate(TEST_CASES, start=1):

    query = test["question"]
    product = test["product"]

    candidates = hybrid_retrieve(
        query,
        product
    )

    if not candidates:
        print(f"\nTEST {number}: NO CANDIDATES")
        continue

    results = rerank(
        query,
        candidates,
        top_k=3
    )

    print(f"\nTEST {number}")
    print("------------------------------")
    print("Question:", query)

    for rank, result in enumerate(results, start=1):

        document = result.get("document", {})

        score = float(
            result.get("reranker_score", 0.0)
        )

        is_expected = (
            document.get("document")
            == test["expected_document"]
            and
            document.get("section")
            == test["expected_section"]
        )

        if is_expected:
            relevant_scores.append(score)
        else:
            irrelevant_scores.append(score)

        print(
            f"Rank {rank} | "
            f"Score: {score:.4f} | "
            f"Relevant: {is_expected}"
        )


print("\n==============================")
print("SCORE SUMMARY")
print("==============================")


print(
    "Relevant scores:",
    [round(score, 4) for score in relevant_scores]
)

print(
    "Irrelevant scores:",
    [round(score, 4) for score in irrelevant_scores]
)


if relevant_scores:

    print(
        "\nMinimum relevant score:",
        round(min(relevant_scores), 4)
    )

    print(
        "Maximum relevant score:",
        round(max(relevant_scores), 4)
    )


if irrelevant_scores:

    print(
        "\nMinimum irrelevant score:",
        round(min(irrelevant_scores), 4)
    )

    print(
        "Maximum irrelevant score:",
        round(max(irrelevant_scores), 4)
    )