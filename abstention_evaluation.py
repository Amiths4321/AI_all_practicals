import json

from ingestion import create_chroma_collection
from retrieval import retrieve_documents
from hybrid_search import HybridRetriever
from reranking import DocumentReranker


# --------------------------------------------------
# 1. LOAD CHROMA
# --------------------------------------------------

chroma_client, collection = (
    create_chroma_collection()
)


# --------------------------------------------------
# 2. LOAD COMPLETE CORPUS
# --------------------------------------------------

all_data = collection.get(
    include=[
        "documents",
        "metadatas"
    ]
)


all_documents = []

for document_id, document, metadata in zip(
    all_data["ids"],
    all_data["documents"],
    all_data["metadatas"]
):

    all_documents.append(
        {
            "id": document_id,
            "document": document,
            "metadata": metadata or {}
        }
    )


# --------------------------------------------------
# 3. CREATE RETRIEVERS
# --------------------------------------------------

hybrid_retriever = HybridRetriever(
    all_documents
)

reranker = DocumentReranker()


# --------------------------------------------------
# 4. LOAD DATASET
# --------------------------------------------------

with open(
    "abstention_eval.json",
    "r",
    encoding="utf-8"
) as file:

    evaluation_data = json.load(file)


# --------------------------------------------------
# 5. EVIDENCE DECISION
# --------------------------------------------------

def has_sufficient_evidence(
    reranked_documents,
    minimum_score=0.0,
    minimum_margin=0.0
):

    if not reranked_documents:
        return False

    top_score = (
        reranked_documents[0][
            "rerank_score"
        ]
    )

    if top_score < minimum_score:
        return False

    if len(reranked_documents) >= 2:

        second_score = (
            reranked_documents[1][
                "rerank_score"
            ]
        )

        margin = (
            top_score
            - second_score
        )

        if margin < minimum_margin:
            return False

    return True


# --------------------------------------------------
# 6. CONFUSION-MATRIX COUNTERS
# --------------------------------------------------

true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0


# --------------------------------------------------
# 7. RUN EVALUATION
# --------------------------------------------------

for item in evaluation_data:

    question = item["question"]

    expected_answerable = (
        item["answerable"]
    )

    retrieval = retrieve_documents(
        collection=collection,
        question=question,
        n_results=10
    )

    vector_results = []

    for document_id, document, metadata in zip(
        retrieval["ids"],
        retrieval["documents"],
        retrieval["metadatas"]
    ):

        vector_results.append(
            {
                "id": document_id,
                "document": document,
                "metadata": metadata or {}
            }
        )


    hybrid_results = (
        hybrid_retriever.search(
            question=question,
            vector_results=vector_results,
            top_k=10
        )
    )


    reranked = reranker.rerank(
        question=question,
        documents=hybrid_results,
        top_n=3
    )


    predicted_answerable = (
        has_sufficient_evidence(
            reranked_documents=reranked,
            minimum_score=2.0,
            minimum_margin=0.0
        )
    )


    # ----------------------------------------------
    # CONFUSION MATRIX
    # ----------------------------------------------

    if (
        expected_answerable
        and predicted_answerable
    ):

        true_positive += 1

    elif (
        not expected_answerable
        and not predicted_answerable
    ):

        true_negative += 1

    elif (
        not expected_answerable
        and predicted_answerable
    ):

        false_positive += 1

    elif (
        expected_answerable
        and not predicted_answerable
    ):

        false_negative += 1


    # ----------------------------------------------
    # DISPLAY
    # ----------------------------------------------

    print("\n" + "=" * 60)

    print(
        "QUESTION:",
        question
    )

    print(
        "EXPECTED ANSWERABLE:",
        expected_answerable
    )

    print(
        "PREDICTED ANSWERABLE:",
        predicted_answerable
    )

    if reranked:

        print(
            "TOP SCORE:",
            round(
                reranked[0]["rerank_score"],
                4
            )
        )

    else:

        print(
            "TOP SCORE: none"
        )


# --------------------------------------------------
# 8. SUMMARY
# --------------------------------------------------

total = len(
    evaluation_data
)

print("\n" + "=" * 60)

print("ABSTENTION EVALUATION")

print("=" * 60)

print(
    "Total questions:",
    total
)

print(
    "True Positive:",
    true_positive
)

print(
    "True Negative:",
    true_negative
)

print(
    "False Positive:",
    false_positive
)

print(
    "False Negative:",
    false_negative
)