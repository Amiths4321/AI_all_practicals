import json

from rag_benchmark import (
    load_json,
    build_chunks,
    create_collection,
    vector_search
)

from hybrid_search import HybridRetriever
from reranking import DocumentReranker
from semantic_evidence import SemanticEvidenceEvaluator

from context_compression import (
    compress_context,
    compression_stats
)

from evidence_compression import (
    EvidenceAwareCompressor
)


CHUNK_SIZE = 500
OVERLAP = 100

VECTOR_K = 10
HYBRID_K = 10
RERANK_K = 10

MAX_DOCUMENTS = 3
EVIDENCE_THRESHOLD = 0.60


def evaluate_evidence(
    required_evidence,
    documents,
    evaluator
):
    result = evaluator.evaluate(
        required_evidence=required_evidence,
        retrieved_documents=documents
    )

    return result["evidence_recall"]


def main():

    documents = load_json(
        "documents.json"
    )

    evaluation_data = load_json(
        "evaluation_data.json"
    )

    chunks = build_chunks(
        documents,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP
    )

    collection = create_collection(
        chunks
    )

    hybrid_retriever = HybridRetriever(
        documents=chunks
    )

    reranker = DocumentReranker()

    evaluator = SemanticEvidenceEvaluator()

    evidence_compressor = (
        EvidenceAwareCompressor(
            evaluator=evaluator,
            threshold=EVIDENCE_THRESHOLD
        )
    )

    top_n_results = []
    evidence_aware_results = []

    print("=" * 80)
    print("CONTEXT COMPRESSION COMPARISON")
    print("=" * 80)

    for case in evaluation_data:

        required_evidence = case.get(
            "required_evidence",
            []
        )

        if not required_evidence:
            continue

        question = case["question"]

        vector_results = vector_search(
            collection=collection,
            question=question,
            top_k=VECTOR_K
        )

        hybrid_results = hybrid_retriever.search(
            question=question,
            vector_results=vector_results,
            top_k=HYBRID_K
        )

        reranked_results = reranker.rerank(
            question=question,
            documents=hybrid_results,
            top_n=RERANK_K
        )

        # ------------------------------------------------
        # Strategy 1: Top-N
        # ------------------------------------------------

        top_n = compress_context(
            reranked_results,
            max_documents=MAX_DOCUMENTS
        )

        top_n_evidence = evaluate_evidence(
            required_evidence,
            top_n,
            evaluator
        )

        top_n_stats = compression_stats(
            reranked_results,
            top_n
        )

        top_n_results.append({
            "question": question,
            "evidence_recall": top_n_evidence,
            **top_n_stats
        })

        # ------------------------------------------------
        # Strategy 2: Evidence-aware
        # ------------------------------------------------

        evidence_aware = (
            evidence_compressor.compress(
                required_evidence=required_evidence,
                reranked_documents=reranked_results,
                max_documents=MAX_DOCUMENTS
            )
        )

        evidence_aware_evidence = (
            evaluate_evidence(
                required_evidence,
                evidence_aware,
                evaluator
            )
        )

        evidence_aware_stats = (
            compression_stats(
                reranked_results,
                evidence_aware
            )
        )

        evidence_aware_results.append({
            "question": question,
            "evidence_recall": (
                evidence_aware_evidence
            ),
            **evidence_aware_stats
        })

        print()
        print("-" * 80)
        print(question)

        print(
            f"Top-N evidence recall: "
            f"{top_n_evidence:.3f}"
        )

        print(
            f"Evidence-aware recall: "
            f"{evidence_aware_evidence:.3f}"
        )

        print(
            f"Top-N compression: "
            f"{top_n_stats['compression_ratio']:.1%}"
        )

        print(
            f"Evidence-aware compression: "
            f"{evidence_aware_stats['compression_ratio']:.1%}"
        )

    # ----------------------------------------------------
    # Aggregate
    # ----------------------------------------------------

    if not top_n_results:
        print("\nNo multi-evidence questions found.")
        return

    avg_top_n_recall = (
        sum(
            item["evidence_recall"]
            for item in top_n_results
        )
        / len(top_n_results)
    )

    avg_evidence_recall = (
        sum(
            item["evidence_recall"]
            for item in evidence_aware_results
        )
        / len(evidence_aware_results)
    )

    avg_top_n_compression = (
        sum(
            item["compression_ratio"]
            for item in top_n_results
        )
        / len(top_n_results)
    )

    avg_evidence_compression = (
        sum(
            item["compression_ratio"]
            for item in evidence_aware_results
        )
        / len(evidence_aware_results)
    )

    print()
    print("=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)

    print(
        f"Questions evaluated: "
        f"{len(top_n_results)}"
    )

    print()

    print(
        f"Top-N evidence recall: "
        f"{avg_top_n_recall:.3f}"
    )

    print(
        f"Evidence-aware recall: "
        f"{avg_evidence_recall:.3f}"
    )

    print()

    print(
        f"Top-N context reduction: "
        f"{avg_top_n_compression:.1%}"
    )

    print(
        f"Evidence-aware reduction: "
        f"{avg_evidence_compression:.1%}"
    )

    with open(
        "compression_comparison.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            {
                "top_n": top_n_results,
                "evidence_aware": (
                    evidence_aware_results
                )
            },
            file,
            indent=2
        )

    print()
    print(
        "Saved compression_comparison.json"
    )


if __name__ == "__main__":
    main()