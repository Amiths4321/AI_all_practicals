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
# 4. LOAD EVALUATION DATA
# --------------------------------------------------

with open(
    "abstention_eval.json",
    "r",
    encoding="utf-8"
) as file:

    evaluation_data = json.load(file)


# --------------------------------------------------
# 5. PRECOMPUTE TOP SCORES
# --------------------------------------------------

results = []

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

    if reranked:

        top_score = (
            reranked[0]["rerank_score"]
        )

    else:

        top_score = float("-inf")

    results.append(
        {
            "question": question,
            "expected_answerable":
                expected_answerable,
            "top_score": top_score
        }
    )


# --------------------------------------------------
# 6. EVALUATE A THRESHOLD
# --------------------------------------------------

def evaluate_threshold(
    results,
    threshold
):

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for item in results:

        actual = (
            item["expected_answerable"]
        )

        predicted = (
            item["top_score"] >= threshold
        )

        if actual and predicted:

            tp += 1

        elif not actual and not predicted:

            tn += 1

        elif not actual and predicted:

            fp += 1

        elif actual and not predicted:

            fn += 1

    total = (
        tp + tn + fp + fn
    )

    accuracy = (
        (tp + tn) / total
        if total
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp
        else 0.0
    )

    false_positive_rate = (
        fp / (fp + tn)
        if fp + tn
        else 0.0
    )

    return {
        "threshold": threshold,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate":
            false_positive_rate
    }


# --------------------------------------------------
# 7. TEST MULTIPLE THRESHOLDS
# --------------------------------------------------

thresholds = [
    0.0,
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
    10.0
]


print("\n" + "=" * 80)
print("THRESHOLD CALIBRATION")
print("=" * 80)

print(
    "\nThreshold | "
    "TP | TN | FP | FN | "
    "Precision | Recall | FPR"
)

print("-" * 80)


for threshold in thresholds:

    metrics = evaluate_threshold(
        results,
        threshold
    )

    print(
        f"{metrics['threshold']:9.1f} | "
        f"{metrics['tp']:2d} | "
        f"{metrics['tn']:2d} | "
        f"{metrics['fp']:2d} | "
        f"{metrics['fn']:2d} | "
        f"{metrics['precision']:.3f}     | "
        f"{metrics['recall']:.3f}  | "
        f"{metrics['false_positive_rate']:.3f}"
    )