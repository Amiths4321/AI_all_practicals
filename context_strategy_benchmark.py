import json
import time

import chromadb
from chromadb.config import Settings

from chunking import chunk_text
from hybrid_search import HybridRetriever
from reranking import DocumentReranker
from semantic_evidence import SemanticEvidenceEvaluator
from context_builder import ContextBuilder
from generation import generate_answer
from rag_judge import judge_answer


DOCUMENT_FILE = "documents.json"
EVALUATION_FILE = "evaluation_data.json"
OUTPUT_FILE = "context_strategy_benchmark.json"

CHUNK_SIZE = 500
OVERLAP = 100

VECTOR_K = 10
HYBRID_K = 10
RERANK_K = 10

MAX_DOCUMENTS = 3

STRATEGIES = [
    "original",
    "top_n",
    "evidence_aware",
    "evidence_first",
    "evidence_last",
    "sentence"
]


def load_documents():
    with open(
        DOCUMENT_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def load_evaluation_data():
    with open(
        EVALUATION_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def build_chunks(documents):

    chunks = []

    for document in documents:

        source = document["source"]

        document_chunks = chunk_text(
            document["text"],
            chunk_size=CHUNK_SIZE,
            overlap=OVERLAP
        )

        for index, chunk in enumerate(document_chunks):

            chunks.append({
                "id": f"{source}_{index}",
                "document": chunk,
                "metadata": {
                    "source": source,
                    "chunk_index": index
                }
            })

    return chunks


def create_collection(chunks):

    client = chromadb.Client(
        Settings(
            anonymized_telemetry=False
        )
    )

    collection = client.create_collection(
        name=f"context_strategy_{int(time.time())}"
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

    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )

    documents = []

    for index in range(
        len(results["ids"][0])
    ):

        documents.append({
            "id": results["ids"][0][index],
            "document": results["documents"][0][index],
            "metadata": results["metadatas"][0][index],
            "score": float(
                results["distances"][0][index]
            )
        })

    return documents


def retrieve_and_rerank(
    question,
    collection,
    hybrid_retriever,
    reranker
):
    """
    Complete retrieval pipeline:

    vector search
        ↓
    BM25 keyword search
        ↓
    RRF hybrid retrieval
        ↓
    cross-encoder reranking
    """

    vector_results = vector_search(
        collection=collection,
        question=question,
        top_k=VECTOR_K
    )

    keyword_results = hybrid_retriever.keyword_search(
        question=question,
        top_k=HYBRID_K
    )

    hybrid_results = hybrid_retriever.reciprocal_rank_fusion(
        vector_results=vector_results,
        keyword_results=keyword_results,
        top_k=HYBRID_K
    )

    reranked_results = reranker.rerank(
        question=question,
        documents=hybrid_results,
        top_n=RERANK_K
    )

    return {
        "vector_results": vector_results,
        "keyword_results": keyword_results,
        "hybrid_results": hybrid_results,
        "reranked_results": reranked_results
    }


def build_context(documents):

    parts = []

    for index, item in enumerate(
        documents,
        start=1
    ):

        metadata = (
            item.get("metadata", {})
            or {}
        )

        source = metadata.get(
            "source",
            "unknown"
        )

        parts.append(
            f"[SOURCE {index}: {source}]\n"
            f"{item['document']}"
        )

    return "\n\n".join(parts)


def main():

    documents = load_documents()

    evaluation_data = load_evaluation_data()

    chunks = build_chunks(
        documents
    )

    collection = create_collection(
        chunks
    )

    hybrid_retriever = HybridRetriever(
        documents=chunks
    )

    reranker = DocumentReranker()

    context_builder = ContextBuilder()

    evidence_evaluator = SemanticEvidenceEvaluator()

    all_results = []

    for case in evaluation_data:

        if not case.get(
            "answerable",
            False
        ):
            continue

        question = case["question"]

        required_evidence = case.get(
            "required_evidence",
            []
        )

        print()
        print("=" * 90)
        print(question)
        print("=" * 90)

        # ---------------------------------------------------------
        # Retrieval + reranking
        # ---------------------------------------------------------

        retrieval = retrieve_and_rerank(
            question=question,
            collection=collection,
            hybrid_retriever=hybrid_retriever,
            reranker=reranker
        )

        reranked_results = retrieval["reranked_results"]

        # ---------------------------------------------------------
        # Retrieval-level evidence evaluation
        # ---------------------------------------------------------

        evidence_result = evidence_evaluator.evaluate(
            required_evidence=required_evidence,
            retrieved_documents=reranked_results
        )

        strategy_results = {}

        # ---------------------------------------------------------
        # Compare context strategies
        # ---------------------------------------------------------

        for strategy in STRATEGIES:

            selected_documents = context_builder.build(
                strategy=strategy,
                question=question,
                reranked_documents=reranked_results,
                required_evidence=required_evidence,
                max_documents=MAX_DOCUMENTS
            )

            # -----------------------------------------------------
            # Strategy-specific evidence evaluation
            # -----------------------------------------------------

            strategy_evidence = evidence_evaluator.evaluate(
                required_evidence=required_evidence,
                retrieved_documents=selected_documents
            )

            # -----------------------------------------------------
            # Build context
            # -----------------------------------------------------

            context = build_context(
                selected_documents
            )

            # -----------------------------------------------------
            # Generate answer
            # -----------------------------------------------------

            start_time = time.perf_counter()

            answer = generate_answer(
                question,
                selected_documents
            )

            latency = (
                time.perf_counter()
                - start_time
            )

            # -----------------------------------------------------
            # Judge answer
            # -----------------------------------------------------

            judgement = judge_answer(
                question,
                context,
                answer
            )

            # -----------------------------------------------------
            # Save strategy result
            # -----------------------------------------------------

            strategy_results[strategy] = {
                "documents_used": len(
                    selected_documents
                ),

                "context_characters": len(
                    context
                ),

                "generation_latency_seconds": latency,

                "answer": answer,

                "judgement": judgement,

                # Evidence found during retrieval/reranking
                "retrieval_evidence_recall":
                    evidence_result[
                        "evidence_recall"
                    ],

                # Evidence preserved after
                # applying this context strategy
                "strategy_evidence_recall":
                    strategy_evidence[
                        "evidence_recall"
                    ],

                "evidence_matches":
                    strategy_evidence[
                        "matches"
                    ]
            }

            print(
                f"{strategy:18} | "
                f"docs={len(selected_documents):2} | "
                f"context={len(context):5} | "
                f"latency={latency:.2f}s | "
                f"ground={judgement['groundedness']} | "
                f"complete={judgement['completeness']} | "
                f"evidence={strategy_evidence['evidence_recall']:.2f}"
            )

        # ---------------------------------------------------------
        # Save question-level results
        # ---------------------------------------------------------

        all_results.append({
            "question": question,

            "required_evidence":
                required_evidence,

            # Retrieval-level evidence recall
            "retrieval_evidence_recall":
                evidence_result[
                    "evidence_recall"
                ],

            "strategies":
                strategy_results
        })

    # -------------------------------------------------------------
    # Save benchmark
    # -------------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_results,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print(
        f"Saved results to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()