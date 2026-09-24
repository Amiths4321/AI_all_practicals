import json

from ingestion import create_chroma_collection
from retrieval import retrieve_documents
from hybrid_search import hybrid_search
from evaluation import evaluate_retrieval


# --------------------------------------------------
# 1. LOAD DATABASE
# --------------------------------------------------

chroma_client, collection = (
    create_chroma_collection()
)


# --------------------------------------------------
# 2. LOAD EVALUATION DATA
# --------------------------------------------------

with open(
    "rag_eval.json",
    "r",
    encoding="utf-8"
) as file:

    evaluation_data = json.load(file)


# --------------------------------------------------
# 3. STORAGE FOR METRICS
# --------------------------------------------------

vector_metrics = []
hybrid_metrics = []


# --------------------------------------------------
# 4. EVALUATE EACH QUESTION
# --------------------------------------------------

for item in evaluation_data:

    question = item["question"]

    relevant_ids = item[
        "relevant_ids"
    ]

    retrieval = retrieve_documents(
        collection=collection,
        question=question,
        n_results=5
    )

    ids = retrieval["ids"]

    documents = retrieval[
        "documents"
    ]


    # ----------------------------------------------
    # VECTOR RESULTS
    # ----------------------------------------------

    vector_results = []

    for document_id, document in zip(
        ids,
        documents
    ):

        vector_results.append(
            {
                "id": document_id,
                "document": document
            }
        )


    vector_eval = evaluate_retrieval(
        retrieved_ids=ids,
        relevant_ids=relevant_ids,
        k=5
    )

    vector_metrics.append(
        vector_eval
    )


    # ----------------------------------------------
    # HYBRID RESULTS
    # ----------------------------------------------

    candidates = vector_results

    hybrid_results = hybrid_search(
        question=question,
        vector_results=vector_results,
        candidates=candidates,
        top_k=5
    )

    hybrid_ids = [
        item["id"]
        for item in hybrid_results
    ]

    hybrid_eval = evaluate_retrieval(
        retrieved_ids=hybrid_ids,
        relevant_ids=relevant_ids,
        k=5
    )

    hybrid_metrics.append(
        hybrid_eval
    )


    # ----------------------------------------------
    # DISPLAY QUESTION
    # ----------------------------------------------

    print("\n" + "=" * 60)

    print(
        "QUESTION:",
        question
    )

    print(
        "\nRELEVANT IDS:",
        relevant_ids
    )

    print(
        "\nVECTOR IDS:",
        ids
    )

    print(
        "VECTOR:",
        vector_eval
    )

    print(
        "\nHYBRID IDS:",
        hybrid_ids
    )

    print(
        "HYBRID:",
        hybrid_eval
    )


# --------------------------------------------------
# 5. AVERAGE METRICS
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


print("\n" + "=" * 60)

print(
    "VECTOR vs HYBRID"
)

print("=" * 60)


metric_names = [
    "precision",
    "recall",
    "f1",
    "reciprocal_rank"
]


for metric in metric_names:

    vector_average = average(
        vector_metrics,
        metric
    )

    hybrid_average = average(
        hybrid_metrics,
        metric
    )

    print(
        f"\n{metric}:"
    )

    print(
        f"Vector: {vector_average:.3f}"
    )

    print(
        f"Hybrid: {hybrid_average:.3f}"
    )