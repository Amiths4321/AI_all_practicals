import json
import time

from rag_benchmark import (
    load_json,
    build_chunks,
    create_collection,
    vector_search
)

from hybrid_search import HybridRetriever
from reranking import DocumentReranker
from semantic_evidence import SemanticEvidenceEvaluator

from context_compression import compress_context
from evidence_compression import EvidenceAwareCompressor

from generation import generate_answer
from rag_judge import judge_answer


CHUNK_SIZE = 500
OVERLAP = 100

VECTOR_K = 10
HYBRID_K = 10
RERANK_K = 10

COMPRESSED_K = 3
EVIDENCE_THRESHOLD = 0.60


def build_context(documents):
    parts = []

    for index, item in enumerate(
        documents,
        start=1
    ):
        source = (
            item.get("metadata", {})
            .get("source", "unknown")
        )

        parts.append(
            f"[SOURCE {index}: {source}]\n"
            f"{item['document']}"
        )

    return "\n\n".join(parts)


def generate_with_context(
    question,
    documents
):
    context = build_context(documents)

    start = time.perf_counter()

    answer = generate_answer(
        question=question,
        documents=documents
    )

    latency = (
        time.perf_counter()
        - start
    )

    return {
        "answer": answer,
        "context": context,
        "latency_seconds": latency,
        "context_characters": len(context)
    }


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

    evidence_evaluator = (
        SemanticEvidenceEvaluator()
    )

    evidence_compressor = (
        EvidenceAwareCompressor(
            evaluator=evidence_evaluator,
            threshold=EVIDENCE_THRESHOLD
        )
    )

    all_results = []

    for case in evaluation_data:

        if not case["answerable"]:
            continue

        question = case["question"]

        print()
        print("=" * 80)
        print(question)
        print("=" * 80)

        vector_results = vector_search(
            collection=collection,
            question=question,
            top_k=VECTOR_K
        )

        hybrid_results = (
            hybrid_retriever.search(
                question=question,
                vector_results=vector_results,
                top_k=HYBRID_K
            )
        )

        reranked_results = reranker.rerank(
            question=question,
            documents=hybrid_results,
            top_n=RERANK_K
        )

        top_n_documents = compress_context(
            reranked_results,
            max_documents=COMPRESSED_K
        )

        evidence_documents = (
            evidence_compressor.compress(
                required_evidence=case.get(
                    "required_evidence",
                    []
                ),
                reranked_documents=reranked_results,
                max_documents=COMPRESSED_K
            )
        )

        strategies = {
            "full": reranked_results,
            "top_n": top_n_documents,
            "evidence_aware": evidence_documents
        }

        question_results = {
            "question": question,
            "strategies": {}
        }

        for strategy_name, selected_documents in (
            strategies.items()
        ):

            generation = generate_with_context(
                question,
                selected_documents
            )

            judgement = judge_answer(
                question=question,
                context=generation["context"],
                answer=generation["answer"]
            )

            question_results[
                "strategies"
            ][strategy_name] = {
                "answer": generation["answer"],
                "latency_seconds": (
                    generation[
                        "latency_seconds"
                    ]
                ),
                "context_characters": (
                    generation[
                        "context_characters"
                    ]
                ),
                "judgement": judgement
            }

            print()
            print(
                f"--- {strategy_name} ---"
            )

            print(
                "Context characters:",
                generation[
                    "context_characters"
                ]
            )

            print(
                "Latency:",
                f"{generation['latency_seconds']:.2f}s"
            )

            print(
                "Groundedness:",
                judgement["groundedness"]
            )

            print(
                "Completeness:",
                judgement["completeness"]
            )

        all_results.append(
            question_results
        )

    with open(
        "compression_generation_results.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            all_results,
            file,
            indent=2
        )

    print()
    print("=" * 80)
    print(
        "Saved compression_generation_results.json"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()