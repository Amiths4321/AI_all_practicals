import json
import time

from rag_benchmark import (
    load_json,
    build_chunks,
    create_collection,
    evaluate_question,
    aggregate_metrics
)

from hybrid_search import HybridRetriever
from reranking import DocumentReranker
from semantic_evidence import SemanticEvidenceEvaluator


CHUNK_CONFIGS = [
    (300, 60),
    (500, 100),
    (800, 160)
]

VECTOR_K_VALUES = [5, 10, 20]

HYBRID_K_VALUES = [10]

RERANK_K_VALUES = [5]

EVIDENCE_THRESHOLDS = [
    0.50,
    0.60,
    0.70
]


def run_experiment(
    documents,
    evaluation_data,
    chunk_size,
    overlap,
    vector_k,
    hybrid_k,
    rerank_k,
    evidence_threshold,
    reranker,
    evidence_evaluator
):
    start_time = time.perf_counter()

    chunks = build_chunks(
        documents,
        chunk_size=chunk_size,
        overlap=overlap
    )

    collection = create_collection(chunks)
    hybrid_retriever = HybridRetriever(documents=chunks)

    results = []

    for case in evaluation_data:
        result = evaluate_question(
            case=case,
            collection=collection,
            hybrid_retriever=hybrid_retriever,
            reranker=reranker,
            evidence_evaluator=evidence_evaluator,
            vector_k=vector_k,
            hybrid_k=hybrid_k,
            rerank_k=rerank_k,
            evidence_threshold=evidence_threshold
        )

        results.append(result)

    metrics = aggregate_metrics(results)

    elapsed = time.perf_counter() - start_time

    return {
        "experiment": {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "vector_k": vector_k,
            "hybrid_k": hybrid_k,
            "rerank_k": rerank_k,
            "evidence_threshold": evidence_threshold
        },
        "runtime_seconds": elapsed,
        "metrics": metrics,
        "question_results": results
    }


def main():
    documents = load_json("documents.json")
    evaluation_data = load_json("evaluation_data.json")

    print("Documents:", len(documents))
    print("Evaluation questions:", len(evaluation_data))

    if not evaluation_data:
        raise ValueError(
            "evaluation_data.json contains 0 evaluation questions. "
            "Add evaluation cases before running the experiment sweep."
        )

    experiment_results = []
    experiment_details = []

    # Load models ONCE, reused across every experiment in the sweep.
    reranker = DocumentReranker()
    evidence_evaluator = SemanticEvidenceEvaluator()

    experiment_number = 0

    total_experiments = (
        len(CHUNK_CONFIGS)
        * len(VECTOR_K_VALUES)
        * len(HYBRID_K_VALUES)
        * len(RERANK_K_VALUES)
        * len(EVIDENCE_THRESHOLDS)
    )

    print(f"Total experiments: {total_experiments}")

    for chunk_size, overlap in CHUNK_CONFIGS:
        for vector_k in VECTOR_K_VALUES:
            for hybrid_k in HYBRID_K_VALUES:
                for rerank_k in RERANK_K_VALUES:
                    for threshold in EVIDENCE_THRESHOLDS:
                        experiment_number += 1

                        print("\n" + "=" * 70)
                        print(f"Experiment {experiment_number}/{total_experiments}")
                        print(
                            f"chunk={chunk_size}, "
                            f"overlap={overlap}, "
                            f"vector_k={vector_k}, "
                            f"hybrid_k={hybrid_k}, "
                            f"rerank_k={rerank_k}, "
                            f"threshold={threshold}"
                        )

                        result = run_experiment(
                            documents=documents,
                            evaluation_data=evaluation_data,
                            chunk_size=chunk_size,
                            overlap=overlap,
                            vector_k=vector_k,
                            hybrid_k=hybrid_k,
                            rerank_k=rerank_k,
                            evidence_threshold=threshold,
                            reranker=reranker,
                            evidence_evaluator=evidence_evaluator
                        )

                        experiment_results.append({
                            **result["experiment"],
                            "runtime_seconds": result["runtime_seconds"],
                            **result["metrics"]
                        })

                        experiment_details.append({
                            **result["experiment"],
                            "runtime_seconds": result["runtime_seconds"],
                            "question_results": result["question_results"]
                        })

                        print(
                            "Evidence recall:",
                            result["metrics"]["average_evidence_recall"]
                        )

                        print(
                            "False answer rate:",
                            result["metrics"]["false_answer_rate"]
                        )

                        print(
                            "Runtime:",
                            f"{result['runtime_seconds']:.2f}s"
                        )

    with open(
        "experiment_results.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(experiment_results, file, indent=2)

    with open(
        "experiment_details.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(experiment_details, file, indent=2)

    print("\n" + "=" * 70)
    print(f"Saved {len(experiment_results)} experiments.")
    print("Saved detailed question results to experiment_details.json")


if __name__ == "__main__":
    main()