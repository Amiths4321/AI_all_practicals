import json
import time

from rag_generation_benchmark import (
    load_json,
    build_chunks,
    create_collection,
    retrieve_documents,
    source_overlap
)

from hybrid_search import HybridRetriever
from reranking import DocumentReranker
from evidence_eval import evaluate_question
from semantic_evidence import SemanticEvidenceEvaluator


# ============================================================
# CONFIGURATION
# ============================================================

TOP_K = 5

DOCUMENT_FILE = "documents.json"
EVALUATION_FILE = "generation_eval.json"
OUTPUT_FILE = "rag_evaluation_report.json"


# ============================================================
# METRICS
# ============================================================

def precision_at_k(retrieved_sources, relevant_sources, k):
    retrieved = retrieved_sources[:k]

    if not retrieved:
        return 0.0

    relevant = sum(1 for source in retrieved if source in relevant_sources)

    return relevant / len(retrieved)


def recall_at_k(retrieved_sources, relevant_sources, k):
    if not relevant_sources:
        return 1.0

    retrieved = set(retrieved_sources[:k])
    relevant = set(relevant_sources)

    return len(retrieved.intersection(relevant)) / len(relevant)


def reciprocal_rank(retrieved_sources, relevant_sources):
    relevant_sources = set(relevant_sources)

    for rank, source in enumerate(retrieved_sources, start=1):
        if source in relevant_sources:
            return 1 / rank

    return 0.0


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():
    print("=" * 80)
    print("RAG EVALUATION")
    print("=" * 80)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    documents = load_json(DOCUMENT_FILE)
    evaluation_data = load_json(EVALUATION_FILE)

    # --------------------------------------------------------
    # Build corpus
    # --------------------------------------------------------

    chunks = build_chunks(documents)

    print(f"Documents: {len(documents)}")
    print(f"Chunks:    {len(chunks)}")

    # --------------------------------------------------------
    # Create retrieval system
    # --------------------------------------------------------

    collection = create_collection(chunks)
    hybrid_retriever = HybridRetriever(chunks)
    reranker = DocumentReranker()
    semantic_evaluator = SemanticEvidenceEvaluator()

    # --------------------------------------------------------
    # Per-question results
    # --------------------------------------------------------

    question_results = []

    precisions = []
    recalls = []
    reciprocal_ranks = []

    answerable_correct = 0
    answerable_total = 0

    start_time = time.time()

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    for index, item in enumerate(evaluation_data, start=1):
        question = item["question"]
        relevant_sources = set(item.get("relevant_sources", []))
        answerable = item.get("answerable", True)

        retrieved = retrieve_documents(
            question,
            collection,
            hybrid_retriever,
            reranker
        )

        semantic_evidence = semantic_evaluator.evaluate(
            required_evidence=item.get("required_evidence", []),
            retrieved_documents=retrieved
        )

        evidence_metrics = evaluate_question(item, retrieved)

        retrieved_sources = [
            document["metadata"].get("source")
            for document in retrieved
        ]

        precision = precision_at_k(retrieved_sources, relevant_sources, TOP_K)
        recall = recall_at_k(retrieved_sources, relevant_sources, TOP_K)
        rr = reciprocal_rank(retrieved_sources, relevant_sources)
        evidence_found = source_overlap(retrieved, relevant_sources)

        if answerable:
            answerable_total += 1
            if evidence_found:
                answerable_correct += 1

        precisions.append(precision)
        recalls.append(recall)
        reciprocal_ranks.append(rr)

        question_result = {
            "question": question,
            "answerable": answerable,
            "relevant_sources": list(relevant_sources),
            "retrieved_sources": retrieved_sources,

            "precision_at_k": precision,
            "recall_at_k": recall,
            "reciprocal_rank": rr,

            "source_precision": evidence_metrics["source_precision"],
            "source_recall": evidence_metrics["source_recall"],
            "evidence_recall": evidence_metrics["evidence_recall"],
            "semantic_evidence_recall": semantic_evidence["evidence_recall"],

            "evidence_found": evidence_found
        }

        question_results.append(question_result)

        # ----------------------------------------------------
        # Progress output
        # ----------------------------------------------------

        print()
        print(f"[{index}/{len(evaluation_data)}] {question}")

        print(
            "  Semantic evidence recall:",
            f"{semantic_evidence['evidence_recall']:.3f}"
        )

        for match in semantic_evidence["matches"]:
            print()
            print("  Required evidence:", match["evidence"])
            print("  Best source:      ", match["source"])
            print("  Similarity:       ", f"{match['score']:.4f}")

        print(f"  Precision@{TOP_K}: {precision:.3f}")
        print(f"  Recall@{TOP_K}:    {recall:.3f}")
        print(f"  RR:                {rr:.3f}")
        print(f"  Evidence:          {evidence_found}")

    # --------------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------------

    elapsed = time.time() - start_time
    total = len(evaluation_data)

    average_evidence_recall = (
        sum(item["semantic_evidence_recall"] for item in question_results)
        / len(question_results)
        if question_results
        else 0.0
    )

    report = {
        "experiment": {
            "chunk_size": 500,
            "overlap": 100,
            "retrieval": "hybrid_rrf",
            "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "generation_model": "qwen2.5vl:latest"
        },
        "configuration": {
            "top_k": TOP_K,
            "documents": DOCUMENT_FILE,
            "evaluation": EVALUATION_FILE
        },

        "corpus": {
            "documents": len(documents),
            "chunks": len(chunks)
        },

        "retrieval": {
            "precision_at_k": sum(precisions) / total if total else 0.0,
            "recall_at_k": sum(recalls) / total if total else 0.0,
            "mrr": sum(reciprocal_ranks) / total if total else 0.0
        },

        "evidence": {
            "answerable_questions": answerable_total,
            "answerable_with_evidence": answerable_correct,
            "evidence_coverage": (
                answerable_correct / answerable_total
                if answerable_total
                else 0.0
            ),
            "average_evidence_recall": average_evidence_recall
        },

        "runtime": {
            "seconds": elapsed
        },

        "questions": question_results
    }

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print()
    print("=" * 80)
    print("FINAL RAG REPORT")
    print("=" * 80)

    print(f"Precision@{TOP_K}:        {report['retrieval']['precision_at_k']:.3f}")
    print(f"Recall@{TOP_K}:           {report['retrieval']['recall_at_k']:.3f}")
    print(f"MRR:                     {report['retrieval']['mrr']:.3f}")
    print(f"Evidence coverage:       {report['evidence']['evidence_coverage']:.3f}")
    print(f"Avg semantic ev. recall: {report['evidence']['average_evidence_recall']:.3f}")
    print(f"Runtime:                 {elapsed:.2f}s")
    print()
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()