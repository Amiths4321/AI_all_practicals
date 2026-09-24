import json
import os
from datetime import datetime
from context_compression import (
    compress_context,
    compression_stats
)

import chromadb
from chromadb.config import Settings

from hybrid_search import HybridRetriever
from reranking import DocumentReranker
from semantic_evidence import SemanticEvidenceEvaluator
from generation import generate_answer, build_context
from rag_judge import judge_answer
from failure_analysis import classify_failure

os.environ["ANONYMIZED_TELEMETRY"] = "false"


DOCUMENTS_FILE = "documents.json"
EVALUATION_FILE = "evaluation_data.json"

DB_PATH = "./benchmark_chroma"

CHUNK_SIZE = 500
OVERLAP = 100

VECTOR_K = 10
HYBRID_K = 10
RERANK_K = 5

EVIDENCE_THRESHOLD = 0.60


def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()

    if not words:
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def build_chunks(
    documents,
    chunk_size=500,
    overlap=100
):
    chunks = []
    counter = 0
def build_chunks(
    documents,
    chunk_size=500,
    overlap=100
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


def create_collection(chunks):
    client = chromadb.PersistentClient(
        path=DB_PATH,
        settings=Settings(
            anonymized_telemetry=False
        )
    )

    try:
        client.delete_collection(name="documents")
    except Exception:
        pass

    collection = client.get_or_create_collection(name="documents")

    if chunks:
        collection.add(
            ids=[item["id"] for item in chunks],
            documents=[item["document"] for item in chunks],
            metadatas=[item["metadata"] for item in chunks]
        )

    return collection


def vector_search(collection, question, top_k=10):
    count = collection.count()

    if count == 0:
        return []

    actual_k = min(top_k, count)

    result = collection.query(
        query_texts=[question],
        n_results=actual_k
    )

    documents = result["documents"][0]
    ids = result["ids"][0]
    metadatas = result["metadatas"][0]

    results = []

    for document_id, document, metadata in zip(ids, documents, metadatas):
        results.append({
            "id": document_id,
            "document": document,
            "metadata": metadata or {}
        })

    return results


def retrieve(
    question,
    collection,
    hybrid_retriever,
    vector_k,
    hybrid_k
):
    vector_results = vector_search(
        collection=collection,
        question=question,
        top_k=vector_k
    )

    keyword_results = hybrid_retriever.keyword_search(
        question=question,
        top_k=hybrid_k
    )

    hybrid_results = hybrid_retriever.reciprocal_rank_fusion(
        vector_results=vector_results,
        keyword_results=keyword_results,
        top_k=hybrid_k
    )

    return {
        "vector_results": vector_results,
        "keyword_results": keyword_results,
        "hybrid_results": hybrid_results
    }


def rerank_documents(question, documents, reranker, rerank_k):
    return reranker.rerank(
        question=question,
        documents=documents,
        # BUG FIX: was hardcoded to the module-level RERANK_K constant,
        # so callers passing a different rerank_k (e.g. a sweep script)
        # were silently ignored.
        top_n=rerank_k
    )


def evaluate_evidence(
    required_evidence,
    reranked_documents,
    evidence_evaluator
):
    return evidence_evaluator.evaluate(
        required_evidence=required_evidence,
        retrieved_documents=reranked_documents
    )


def evidence_gate(
    evidence_result,
    reranked_documents,
    evidence_threshold
):
    if not reranked_documents:
        return False

    # Required evidence must be fully present.
    if evidence_result["evidence_recall"] < 1.0:
        return False

    top_score = reranked_documents[0]["rerank_score"]

    if top_score < evidence_threshold:
        return False

    return True

    # BUG FIX: was hardcoded to the module-level EVIDENCE_THRESHOLD
    # constant, so callers passing a different evidence_threshold
    # (e.g. a threshold sweep) had zero actual effect on the gate.
    return top_score >= evidence_threshold


def evaluate_question(
    case,
    collection,
    hybrid_retriever,
    reranker,
    evidence_evaluator,
    vector_k=VECTOR_K,
    hybrid_k=HYBRID_K,
    rerank_k=RERANK_K,
    evidence_threshold=EVIDENCE_THRESHOLD
):
    question = case["question"]

    retrieval_trace = retrieve(
        question=question,
        collection=collection,
        hybrid_retriever=hybrid_retriever,
        vector_k=vector_k,
        hybrid_k=hybrid_k
    )

    vector_results = retrieval_trace["vector_results"]
    keyword_results = retrieval_trace["keyword_results"]
    hybrid_results = retrieval_trace["hybrid_results"]

    reranked_documents = rerank_documents(
        question=question,
        documents=hybrid_results,
        reranker=reranker,
        rerank_k=rerank_k
    )

    compressed_documents = compress_context(
        reranked_documents,
        max_documents=3
    )
    evidence_result = evidence_evaluator.evaluate(
        required_evidence=case.get("required_evidence", []),
        retrieved_documents=reranked_documents
    )

    compression = compression_stats(
        reranked_documents,
        compressed_documents
    )

    gate_passed = evidence_gate(
        evidence_result=evidence_result,
        reranked_documents=reranked_documents,
        evidence_threshold=evidence_threshold
    )

    retrieved_sources = {
        item.get("metadata", {}).get("source")
        for item in reranked_documents
        if item.get("metadata", {}).get("source")
    }

    expected_sources = set(case.get("relevant_sources", []))

    if expected_sources:
        source_overlap = (
            len(retrieved_sources & expected_sources)
            / len(expected_sources)
        )
    else:
        source_overlap = 1.0

    # --------------------------------------------------------
    # RECONSTRUCTED SECTION — generate_answer / build_context /
    # judge_answer / classify_failure were imported at the top of
    # the file but never called anywhere, and main() expects
    # result["answer"] and result["failure_category"], neither of
    # which this function used to return. Wired here based on the
    # shape aggregate_metrics() expects from "judgement"
    # (groundedness / relevance / completeness / unsupported_claims).
    # VERIFY the actual signatures in generation.py, rag_judge.py,
    # and failure_analysis.py match what's called below.
    # --------------------------------------------------------

    if gate_passed:
        context = build_context(reranked_documents)
        answer = generate_answer(
            question=question,
            documents=reranked_documents
        )
        judgement = judge_answer(
            question=question,
            answer=answer,
            context=context
        )
    else:
        context = None
        answer = None
        judgement = None

    return {
        "question": question,
        "answerable": case["answerable"],
        "expected_sources": case.get(
            "relevant_sources",
            []
        ),

        "vector_results": vector_results,
        "keyword_results": keyword_results,
        "hybrid_results": hybrid_results,
        "reranked_results": reranked_documents,
        "context_compression": compression,
        "evidence_recall": evidence_result["evidence_recall"],
        "evidence_matches": evidence_result["matches"],

        "gate_passed": gate_passed,

       

        "retrieved_sources": list(retrieved_sources),
        "source_overlap": source_overlap,
        "context_compression": compression,
        "judgement": None
    }

    # classify_failure reads its inputs off the assembled result dict
    # (answerable / gate_passed / evidence_recall), so it has to run
    # after result is built, not before.
    result["failure_category"] = classify_failure(result)

    return result


def aggregate_metrics(results):
    total = len(results)

    if total == 0:
        return {}

    answered = sum(result["gate_passed"] for result in results)

    correct_abstentions = sum(
        not result["answerable"] and not result["gate_passed"]
        for result in results
    )

    false_answers = sum(
        not result["answerable"] and result["gate_passed"]
        for result in results
    )

    evidence_recalls = [result["evidence_recall"] for result in results]
    source_overlaps = [result["source_overlap"] for result in results]

    metrics = {
        "questions": total,
        "answered": answered,
        "answer_rate": answered / total,
        "abstention_rate": 1 - answered / total,
        "correct_abstentions": correct_abstentions,
        "false_answers": false_answers,
        "false_answer_rate": false_answers / total,
        "average_evidence_recall": sum(evidence_recalls) / len(evidence_recalls),
        "average_source_overlap": sum(source_overlaps) / len(source_overlaps)
    }

    judged_results = [
        result
        for result in results
        if result["judgement"] is not None
    ]

    if judged_results:
        metrics["average_groundedness"] = (
            sum(result["judgement"]["groundedness"] for result in judged_results)
            / len(judged_results)
        )

        metrics["average_relevance"] = (
            sum(result["judgement"]["relevance"] for result in judged_results)
            / len(judged_results)
        )

        metrics["average_completeness"] = (
            sum(result["judgement"]["completeness"] for result in judged_results)
            / len(judged_results)
        )

        metrics["average_unsupported_claims"] = (
            sum(result["judgement"]["unsupported_claims"] for result in judged_results)
            / len(judged_results)
        )

    return metrics


def load_json(filename):
    # Check if file is missing or completely empty
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        print(
            f"Warning: '{filename}' is missing or empty. Initializing with an "
            "empty list []."
        )
        with open(filename, "w", encoding="utf-8") as file:
            json.dump([], file)

    # Try loading the file
    with open(filename, "r", encoding="utf-8") as file:
        try:
            return json.load(file)
        except json.JSONDecodeError:
            print(
                f"Warning: '{filename}' contains invalid JSON formatting. "
                "Resetting to empty list."
            )
            # Reset the corrupted file to a valid empty list
            with open(filename, "w", encoding="utf-8") as file_write:
                json.dump([], file_write)
            return []


def main():
    print("=" * 70)
    print("AUTOMATED RAG BENCHMARK")
    print("=" * 70)

    documents = load_json(DOCUMENTS_FILE)
    evaluation_data = load_json(EVALUATION_FILE)

    print(f"Documents: {len(documents)}")

    # -------------------------
    # Build corpus
    # -------------------------

    chunks = build_chunks(
        documents,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP
    )

    print(f"Chunks: {len(chunks)}")

    # -------------------------
    # Create vector DB
    # -------------------------

    collection = create_collection(chunks)

    print(f"Vector DB documents: {collection.count()}")

    # -------------------------
    # Initialize retriever
    # -------------------------

    hybrid_retriever = HybridRetriever(documents=chunks)

    # -------------------------
    # Initialize models
    # -------------------------

    reranker = DocumentReranker()
    evidence_evaluator = SemanticEvidenceEvaluator()

    # -------------------------
    # Evaluate
    # -------------------------

    results = []

    for index, case in enumerate(evaluation_data, start=1):
        print("\n" + "-" * 70)
        print(f"Question {index}/{len(evaluation_data)}")
        print(case["question"])

        result = evaluate_question(
            case=case,
            collection=collection,
            hybrid_retriever=hybrid_retriever,
            reranker=reranker,
            evidence_evaluator=evidence_evaluator,
            vector_k=VECTOR_K,
            hybrid_k=HYBRID_K,
            rerank_k=RERANK_K,
            evidence_threshold=EVIDENCE_THRESHOLD
        )

        results.append(result)

        print("Evidence recall:", result["evidence_recall"])
        print("Gate passed:", result["gate_passed"])
        print("Failure:", result["failure_category"])

        if result["answer"]:
            print("\nAnswer:")
            print(result["answer"])

    # -------------------------
    # Aggregate
    # -------------------------

    metrics = aggregate_metrics(results)

    report = {
        "timestamp": datetime.now().isoformat(),

        "configuration": {
            "chunk_size": CHUNK_SIZE,
            "overlap": OVERLAP,
            "vector_k": VECTOR_K,
            "hybrid_k": HYBRID_K,
            "rerank_k": RERANK_K,
            "evidence_threshold": EVIDENCE_THRESHOLD
        },

        "metrics": metrics,
        "results": results
    }

    with open(
        "benchmark_report.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\n")
    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()