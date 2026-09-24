import json

from rag_benchmark import (
    load_json,
    build_chunks as base_build_chunks,  # unused locally — see note below
    create_collection,
    aggregate_metrics
)

from hybrid_search import HybridRetriever
from reranking import DocumentReranker
from semantic_evidence import SemanticEvidenceEvaluator
import os

os.environ["ANONYMIZED_TELEMETRY"] = "false"

import chromadb
from chromadb.config import Settings

DOCUMENTS_FILE = "documents.json"
EVALUATION_FILE = "evaluation_data.json"

DB_PATH = "./benchmark_chroma"

client = chromadb.PersistentClient(
    path=DB_PATH,
    settings=Settings(
        anonymized_telemetry=False
    )
)


CHUNK_CONFIGS = [
    (200, 40),
    (300, 60),
    (500, 100),
    (800, 160)
]


# ============================================================
# CHUNKING
# ============================================================

def chunk_text(
    text,
    chunk_size=500,
    overlap=100
):
    """
    Sliding-window chunker over raw characters.

    ASSUMPTION: chunk_size / overlap are character counts, not
    tokens or words. Verify this matches how the rest of the
    pipeline (e.g. evidence scoring, embeddings) expects chunks
    to be sized — swap for a word/token splitter if not.
    """

    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunks.append(text[start:end])

        if end >= text_length:
            break

        start = end - overlap

    return chunks


def build_chunks(
    documents,
    chunk_size,
    overlap
):
    chunks = []
    counter = 0

    for document in documents:
        source = document["source"]
        text = document["text"]

        text_chunks = chunk_text(
            text,
            chunk_size=chunk_size,
            overlap=overlap
        )

        for chunk in text_chunks:
            chunks.append({
                "id": f"chunk_{counter}",
                "document": chunk,
                "metadata": {
                    "source": source
                }
            })

            counter += 1

    return chunks


# ============================================================
# RETRIEVAL / RERANKING
# ============================================================

def vector_search(
    collection,
    question,
    top_k
):
    """
    NOTE: assumes `collection` exposes a Chroma-style `.query(...)`
    API. This wasn't defined anywhere in the original file — verify
    against whatever `create_collection` actually returns.
    """

    raw_results = collection.query(
        query_texts=[question],
        n_results=top_k
    )

    documents = raw_results["documents"][0]
    metadatas = raw_results["metadatas"][0]

    return [
        {"document": document, "metadata": metadata}
        for document, metadata in zip(documents, metadatas)
    ]


def retrieve(
    question,
    collection,
    hybrid_retriever,
    vector_k,
    hybrid_k
):
    vector_results = vector_search(
        collection,
        question,
        top_k=vector_k
    )

    hybrid_results = hybrid_retriever.search(
        question=question,
        vector_results=vector_results,
        top_k=hybrid_k
    )

    return hybrid_results


def rerank_documents(
    question,
    documents,
    reranker,
    rerank_k
):
    return reranker.rerank(
        question=question,
        documents=documents,
        top_n=rerank_k
    )


# ============================================================
# EVIDENCE GATE
# ============================================================

def evidence_gate(
    evidence_result,
    reranked_documents,
    evidence_threshold
):
    if not reranked_documents:
        return False

    if evidence_result["evidence_recall"] < 1.0:
        return False

    top_score = reranked_documents[0]["rerank_score"]

    return top_score >= evidence_threshold


# ============================================================
# PER-QUESTION EVALUATION
#
# NOTE: this function was called in the original file but never
# defined or imported. Reconstructed here from the retrieve /
# rerank_documents / evidence_gate helpers that were already
# present but unused — please check this matches what you intended,
# especially the shape of the returned dict (`aggregate_metrics`
# needs to know what keys to expect).
# ============================================================

def evaluate_question(
    case,
    collection,
    hybrid_retriever,
    reranker,
    evidence_evaluator,
    vector_k,
    hybrid_k,
    rerank_k,
    evidence_threshold
):
    question = case["question"]

    retrieved_documents = retrieve(
        question,
        collection,
        hybrid_retriever,
        vector_k,
        hybrid_k
    )
    hybrid_results = retrieve(
          question=question,
          collection=collection,
          hybrid_retriever=hybrid_retriever,
          vector_k=vector_k,
          hybrid_k=hybrid_k
          )      
    reranked_documents = rerank_documents(
        question,
        retrieved_documents,
        reranker,
        rerank_k
    )

    evidence_result = evidence_evaluator.evaluate(
        required_evidence=case.get("required_evidence", []),
        retrieved_documents=reranked_documents
    )

    gate_passed = evidence_gate(
          evidence_result=evidence_result,
          reranked_documents=reranked_documents,
          evidence_threshold=evidence_threshold
          )

    return {
        "question": question,
        "evidence_recall": evidence_result["evidence_recall"],
        "gate_passed": gate_passed
    }


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

def run_experiment(
    documents,
    evaluation_data,
    chunk_size,
    overlap
):

    chunks = build_chunks(
        documents,
        chunk_size=chunk_size,
        overlap=overlap
    )

    collection = create_collection(
        chunks
    )

    hybrid_retriever = HybridRetriever(
        documents=chunks
    )

    reranker = DocumentReranker()

    evidence_evaluator = (
        SemanticEvidenceEvaluator()
    )

    results = []

    for case in evaluation_data:

        result = evaluate_question(
            case=case,
            collection=collection,
            hybrid_retriever=hybrid_retriever,
            reranker=reranker,
            evidence_evaluator=evidence_evaluator
        )

        results.append(result)

    metrics = aggregate_metrics(results)

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():
    documents = load_json("documents.json")
    evaluation_data = load_json("evaluation_data.json")

    all_results = []

    for chunk_size, overlap in CHUNK_CONFIGS:
        print(f"\nTesting chunk={chunk_size}, overlap={overlap}")

        metrics = run_experiment(
            documents=documents,
            evaluation_data=evaluation_data,
            chunk_size=chunk_size,
            overlap=overlap
        )

        row = {
            "chunk_size": chunk_size,
            "overlap": overlap,
            **metrics
        }

        all_results.append(row)

    for row in all_results:
        print(row)

    



    with open(
        "chunk_sweep_results.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(all_results, file, indent=2)
if __name__ == "__main__":
    main()
