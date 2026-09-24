import json
import os
import shutil

import chromadb

from chunking import chunk_text
from hybrid_search import HybridRetriever
from reranking import DocumentReranker


CHUNK_SIZE = 500
OVERLAP = 100

RETRIEVAL_K = 10
RERANK_K = 5

DB_PATH = "./abstention_benchmark_chroma"


THRESHOLDS = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
]


def load_documents():
    with open(
        "documents.json",
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def load_evaluation_data():
    with open(
        "abstention_eval.json",
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def build_chunks(documents):
    chunks = []

    for document in documents:
        source = document["source"]
        text = document["text"]

        document_chunks = chunk_text(
            text,
            chunk_size=CHUNK_SIZE,
            overlap=OVERLAP
        )

        for chunk_index, chunk in enumerate(
            document_chunks
        ):
            chunks.append({
                "id": f"{source}_chunk_{chunk_index}",
                "document": chunk,
                "metadata": {
                    "source": source,
                    "chunk_index": chunk_index
                }
            })

    return chunks


def create_collection(chunks):
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    client = chromadb.PersistentClient(
        path=DB_PATH
    )

    collection = client.get_or_create_collection(
        name="documents"
    )

    collection.add(
        ids=[
            item["id"]
            for item in chunks
        ],
        documents=[
            item["document"]
            for item in chunks
        ],
        metadatas=[
            item["metadata"]
            for item in chunks
        ]
    )

    return collection


def vector_search(
    collection,
    question,
    top_k
):
    available = collection.count()

    if available == 0:
          return []

    actual_k = min(top_k, available)

    available = collection.count()

    if available == 0:
        return []

    actual_k = min(top_k, available)

    results = collection.query(
        query_texts=[question],
        n_results=actual_k
    )

    output = []

    for doc_id, document, metadata in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0]
    ):
        output.append({
            "id": doc_id,
            "document": document,
            "metadata": metadata or {}
        })

    return output


def retrieve_and_rerank(
    question,
    collection,
    hybrid_retriever,
    reranker
):
    vector_results = vector_search(
        collection,
        question,
        RETRIEVAL_K
    )

    hybrid_results = hybrid_retriever.search(
        question=question,
        vector_results=vector_results,
        top_k=RETRIEVAL_K
    )

    reranked_results = reranker.rerank(
        question=question,
        documents=hybrid_results,
        top_n=RERANK_K
    )

    return reranked_results


def evaluate_threshold(
    evaluation_data,
    scored_results,
    threshold
):
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    answered = 0
    correct_answers = 0

    for item in evaluation_data:
        question = item["question"]

        expected_answerable = item["answerable"]

        result = scored_results[question]

        top_score = result["top_score"]

        system_answered = (
            top_score >= threshold
        )

        if system_answered:
            answered += 1

        # --------------------------------------------
        # Classification
        # --------------------------------------------

        if expected_answerable and system_answered:
            true_positive += 1

        elif not expected_answerable and system_answered:
            false_positive += 1

        elif not expected_answerable and not system_answered:
            true_negative += 1

        elif expected_answerable and not system_answered:
            false_negative += 1

        # --------------------------------------------
        # Answered-question retrieval correctness
        # --------------------------------------------

        if system_answered:
            relevant_sources = set(
                item.get(
                    "relevant_sources",
                    []
                )
            )

            retrieved_sources = {
                result_item["metadata"].get("source")
                for result_item
                in result["documents"]
            }

            if (
                relevant_sources
                and
                retrieved_sources.intersection(
                    relevant_sources
                )
            ):
                correct_answers += 1

    total = len(evaluation_data)

    coverage = (
        answered / total
        if total
        else 0.0
    )

    abstention_rate = (
        1 - coverage
    )

    false_answer_rate = (
        false_positive / total
        if total
        else 0.0
    )

    answer_precision = (
        correct_answers / answered
        if answered
        else 0.0
    )

    return {
        "threshold": threshold,
        "tp": true_positive,
        "fp": false_positive,
        "tn": true_negative,
        "fn": false_negative,
        "coverage": coverage,
        "abstention_rate": abstention_rate,
        "false_answer_rate": false_answer_rate,
        "answer_precision": answer_precision
    }

def has_sufficient_evidence(
    reranked_documents,
    minimum_score=0.0,
    minimum_margin=0.0
):
    if not reranked_documents:
        return False

    top_score = reranked_documents[0]["rerank_score"]

    if top_score < minimum_score:
        return False

    if len(reranked_documents) >= 2:
        second_score = reranked_documents[1]["rerank_score"]

        margin = (
            top_score
            - second_score
        )

        if margin < minimum_margin:
            return False

    return True

def main():
    documents = load_documents()
    evaluation_data = load_evaluation_data()

    chunks = build_chunks(documents)

    print(
        f"Total chunks: {len(chunks)}"
    )

    collection = create_collection(
        chunks
    )

    hybrid_retriever = HybridRetriever(
        chunks
    )

    reranker = DocumentReranker()

    # -------------------------------------------------
    # Precompute scores
    # -------------------------------------------------

    scored_results = {}

    print()
    print("Calculating reranker scores...")

    for item in evaluation_data:
        question = item["question"]

        documents = retrieve_and_rerank(
            question,
            collection,
            hybrid_retriever,
            reranker
        )

        if documents:
            top_score = documents[0][
                "rerank_score"
            ]
        else:
            top_score = float("-inf")

        scored_results[question] = {
            "top_score": top_score,
            "documents": documents
        }

        print(
            f"{question}"
        )
        print(
            f"  top score: {top_score:.4f}"
        )

    # -------------------------------------------------
    # Threshold benchmark
    # -------------------------------------------------

    print()
    print("=" * 95)
    print("ABSTENTION THRESHOLD BENCHMARK")
    print("=" * 95)

    print(
        f"{'Threshold':>10} "
        f"{'Coverage':>10} "
        f"{'Abstain':>10} "
        f"{'FP':>6} "
        f"{'FN':>6} "
        f"{'FalseAns':>10} "
        f"{'AnsPrec':>10}"
    )

    print("-" * 95)

    for threshold in THRESHOLDS:

        metrics = evaluate_threshold(
            evaluation_data,
            scored_results,
            threshold
        )

        print(
            f"{metrics['threshold']:>10.2f} "
            f"{metrics['coverage']:>10.3f} "
            f"{metrics['abstention_rate']:>10.3f} "
            f"{metrics['fp']:>6} "
            f"{metrics['fn']:>6} "
            f"{metrics['false_answer_rate']:>10.3f} "
            f"{metrics['answer_precision']:>10.3f}"
        )

def evidence_margin(documents):
    if not documents:
        return 0.0
    
    # Use .get() with a default of 0.0
    top_score = documents[0].get("rerank_score", 0.0)
    
    # Check if there is a second document to compare against
    if len(documents) > 1:
        second_score = documents[1].get("rerank_score", 0.0)
    else:
        second_score = 0.0
        
    return top_score - second_score

# 1. Load the documents first so the variable exists
with open("documents.json", "r", encoding="utf-8") as file:
  documents = json.load(file)

# 2. Now this line will work because 'documents' is defined
margin = evidence_margin(documents)

if __name__ == "__main__":
    main()