import json

from ingestion import create_chroma_collection
from retrieval import retrieve_documents
from hybrid_search import HybridRetriever
from reranking import DocumentReranker

from evaluation import (
    evaluate_ranked_results
)


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
# 4. LOAD EVALUATION DATA
# --------------------------------------------------

with open(
    "rag_eval.json",
    "r",
    encoding="utf-8"
) as file:

    evaluation_data = json.load(file)


hybrid_metrics = []
reranked_metrics = []


# --------------------------------------------------
# 5. EVALUATE
# --------------------------------------------------

for item in evaluation_data:

    question = item["question"]

    relevant_ids = item[
        "relevant_ids"
    ]


    # ----------------------------------------------
    # VECTOR RETRIEVAL
    # ----------------------------------------------

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


    # ----------------------------------------------
    # HYBRID
    # ----------------------------------------------

    hybrid_results = (
        hybrid_retriever.search(
            question=question,
            vector_results=vector_results,
            top_k=10
        )
    )


    hybrid_ids = [
        item["id"]
        for item in hybrid_results
    ]


    hybrid_eval = (
        evaluate_ranked_results(
            retrieved_ids=hybrid_ids,
            relevant_ids=relevant_ids,
            k=3
        )
    )


    # ----------------------------------------------
    # RERANK
    # ----------------------------------------------

    reranked = reranker.rerank(
        question=question,
        documents=hybrid_results,
        top_n=3
    )


    reranked_ids = [
        item["id"]
        for item in reranked
    ]


    reranked_eval = (
        evaluate_ranked_results(
            retrieved_ids=reranked_ids,
            relevant_ids=relevant_ids,
            k=3
        )
    )


    hybrid_metrics.append(
        hybrid_eval
    )

    reranked_metrics.append(
        reranked_eval
    )


    # ----------------------------------------------
    # DISPLAY
    # ----------------------------------------------

    print("\n" + "=" * 60)

    print(
        "QUESTION:",
        question
    )

    print(
        "\nRELEVANT:",
        relevant_ids
    )

    print(
        "\nHYBRID:",
        hybrid_ids
    )

    print(
        hybrid_eval
    )

    print(
        "\nRERANKED:",
        reranked_ids
    )

    print(
        reranked_eval
    )


# --------------------------------------------------
# 6. AVERAGING
# --------------------------------------------------

def average(
    metrics,
    key
):

    if not metrics:
        return 0.0

    return sum(
        item[key]
        for item in metrics
    ) / len(metrics)


# --------------------------------------------------
# 7. FINAL COMPARISON
# --------------------------------------------------

print("\n" + "=" * 60)

print(
    "HYBRID vs RERANKED"
)

print("=" * 60)


for metric in [
    "precision",
    "recall",
    "f1",
    "mrr"
]:

    hybrid_average = average(
        hybrid_metrics,
        metric
    )

    reranked_average = average(
        reranked_metrics,
        metric
    )

    print(
        f"\n{metric.upper()}"
    )

    print(
        f"Hybrid:   "
        f"{hybrid_average:.3f}"
    )

    print(
        f"Reranked: "
        f"{reranked_average:.3f}"
    )