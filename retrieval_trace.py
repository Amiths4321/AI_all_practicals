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


CHUNK_SIZE = 500
OVERLAP = 100

VECTOR_K = 10
HYBRID_K = 10
RERANK_K = 5

EVIDENCE_THRESHOLD = 0.60


def print_documents(title, documents):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    if not documents:
        print("No documents returned.")
        return

    for rank, item in enumerate(documents, start=1):
        metadata = item.get("metadata") or {}
        source = metadata.get("source", "unknown")

        print(f"\n[{rank}]")
        print(f"ID: {item.get('id')}")
        print(f"Source: {source}")

        if "score" in item:
            print(f"Score: {item['score']:.4f}")

        if "rrf_score" in item:
            print(f"RRF score: {item['rrf_score']:.6f}")

        if "rerank_score" in item:
            print(f"Rerank score: {item['rerank_score']:.4f}")

        print("Document:")
        print(item["document"][:500])


def main():

    documents = load_json("documents.json")

    evaluation_data = load_json("evaluation_data.json")

    print("Available questions:")

    for index, case in enumerate(evaluation_data, start=1):
        print(f"{index}. {case['question']}")

    question_number = int(
        input("\nEnter question number: ")
    )

    case = evaluation_data[question_number - 1]

    question = case["question"]

    print("\n" + "=" * 80)
    print("RETRIEVAL TRACE")
    print("=" * 80)

    print(f"\nQuestion:\n{question}")

    chunks = build_chunks(
        documents,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP
    )

    collection = create_collection(chunks)

    hybrid_retriever = HybridRetriever(
        documents=chunks
    )

    reranker = DocumentReranker()

    evidence_evaluator = SemanticEvidenceEvaluator()

    # --------------------------------------------------
    # 1. Vector retrieval
    # --------------------------------------------------

    vector_results = vector_search(
        collection=collection,
        question=question,
        top_k=VECTOR_K
    )

    print_documents(
        "1. VECTOR SEARCH",
        vector_results
    )

    # --------------------------------------------------
    # 2. BM25
    # --------------------------------------------------

    keyword_results = hybrid_retriever.keyword_search(
        question=question,
        top_k=HYBRID_K
    )

    print_documents(
        "2. BM25 KEYWORD SEARCH",
        keyword_results
    )

    # --------------------------------------------------
    # 3. Hybrid RRF
    # --------------------------------------------------

    hybrid_results = hybrid_retriever.search(
        question=question,
        vector_results=vector_results,
        top_k=HYBRID_K
    )

    print_documents(
        "3. HYBRID RRF",
        hybrid_results
    )

    # --------------------------------------------------
    # 4. Cross-encoder reranking
    # --------------------------------------------------

    reranked_results = reranker.rerank(
        question=question,
        documents=hybrid_results,
        top_n=RERANK_K
    )

    print_documents(
        "4. CROSS-ENCODER RERANKING",
        reranked_results
    )

    # --------------------------------------------------
    # 5. Semantic evidence
    # --------------------------------------------------

    evidence_result = evidence_evaluator.evaluate(
        required_evidence=case.get(
            "required_evidence",
            []
        ),
        retrieved_documents=reranked_results
    )

    print("\n" + "=" * 80)
    print("5. SEMANTIC EVIDENCE")
    print("=" * 80)

    print(
        f"Evidence recall: "
        f"{evidence_result['evidence_recall']:.3f}"
    )

    for match in evidence_result["matches"]:

        print("\nRequired evidence:")
        print(match["evidence"])

        print(
            f"Similarity: "
            f"{match['score']:.4f}"
        )

        print(
            f"Document ID: "
            f"{match['document_id']}"
        )

        print(
            f"Source: "
            f"{match['source']}"
        )

    # --------------------------------------------------
    # 6. Evidence gate
    # --------------------------------------------------

    gate_passed = (
        evidence_result["evidence_recall"] >=
        EVIDENCE_THRESHOLD
    )

    print("\n" + "=" * 80)
    print("6. EVIDENCE GATE")
    print("=" * 80)

    print(
        f"Threshold: {EVIDENCE_THRESHOLD:.2f}"
    )

    print(
        f"Evidence recall: "
        f"{evidence_result['evidence_recall']:.3f}"
    )

    print(
        f"Gate passed: {gate_passed}"
    )


if __name__ == "__main__":
    main()