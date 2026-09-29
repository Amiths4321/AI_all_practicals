import json

from hybrid_retriever import hybrid_retrieve
from reranker import rerank

with open("evaluation_questions.json", "r", encoding="utf-8") as f:
    questions = json.load(f)


for item in questions:
    question = item["question"]
    product = item["expected_product"]


    print("\n" + "=" * 70)
    print("QUESTION:", question)
    print("PRODUCT :", product)
    print("=" * 70)

    results = hybrid_retrieve(question, product)

    print("\nRetrieved candidates:", len(results))

    for index, result in enumerate(results[:5], start=1):
        print(
            f"{index}. "
            f"Product={result.get('product')} | "
            f"Score={result.get('score')} | "
            f"Text={result.get('text', '')[:150]}"
        )

    reranked = rerank(question, results)

    print("\nAfter reranking:")

    for index, result in enumerate(reranked[:3], start=1):
        print(
            f"{index}. "
            f"Product={result.get('product')} | "
            f"Reranker={result.get('reranker_score')} | "
            f"Text={result.get('text', '')[:150]}"
        )


with open(
    "evaluation_dataset.json",
    "r",
    encoding="utf-8"
) as file:
    dataset = json.load(file)


K = 3

before_hits = 0
after_hits = 0


print("\n==============================")
print("BEFORE vs AFTER RERANKING")
print("==============================")


for item in dataset:

    query = item["question"]
    product = item["product"]

    expected_document = item[
        "expected_document"
    ]

    expected_section = item[
        "expected_section"
    ]


    # --------------------------------
    # Hybrid retrieval
    # --------------------------------

    candidates = hybrid_retrieve(
        query,
        product
    )


    # --------------------------------
    # BEFORE RERANKING
    # --------------------------------

    before_results = candidates[:K]

    before_found = False

    for result in before_results:

        document = result["document"]

        if (
            document.get("document")
            == expected_document
            and
            document.get("section")
            == expected_section
        ):

            before_found = True
            break


    # --------------------------------
    # AFTER RERANKING
    # --------------------------------

    after_results = rerank(
        query,
        candidates,
        top_k=K
    )


    after_found = False

    for result in after_results:

        document = result["document"]

        if (
            document.get("document")
            == expected_document
            and
            document.get("section")
            == expected_section
        ):

            after_found = True
            break


    if before_found:
        before_hits += 1

    if after_found:
        after_hits += 1


    # --------------------------------
    # Display
    # --------------------------------

    print("\n------------------------------")

    print("Question:")
    print(query)

    print(
        "Before Reranking:",
        "PASS" if before_found else "FAIL"
    )

    print(
        "After Reranking:",
        "PASS" if after_found else "FAIL"
    )


# --------------------------------
# Final metrics
# --------------------------------

before_recall = (
    before_hits / len(dataset)
)

after_recall = (
    after_hits / len(dataset)
)


print("\n==============================")
print("FINAL COMPARISON")
print("==============================")


print(
    f"Recall@{K} Before Reranking: "
    f"{before_recall:.2%}"
)

print(
    f"Recall@{K} After Reranking:  "
    f"{after_recall:.2%}"
)


print(
    f"Improvement: "
    f"{(after_recall - before_recall):.2%}"
)