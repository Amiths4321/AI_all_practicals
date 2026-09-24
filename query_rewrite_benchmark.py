import json

from rag_benchmark import (
    load_json,
    build_chunks,
    create_collection,
    vector_search
)

from hybrid_search import HybridRetriever

from query_rewriter import rewrite_query


CHUNK_SIZE = 500
OVERLAP = 100
TOP_K = 10


def source_match(results, expected_sources):

    expected_sources = set(expected_sources)

    for item in results:

        metadata = item.get("metadata") or {}

        source = metadata.get("source")

        if source in expected_sources:
            return True

    return False


def main():

    documents = load_json("documents.json")
    evaluation_data = load_json(
        "evaluation_data.json"
    )

    chunks = build_chunks(
        documents,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP
    )

    collection = create_collection(chunks)

    hybrid_retriever = HybridRetriever(
        documents=chunks
    )

    original_hits = 0
    rewritten_hits = 0
    total = 0

    print("=" * 80)
    print("ORIGINAL VS REWRITTEN QUERY RETRIEVAL")
    print("=" * 80)

    for case in evaluation_data:

        if not case["answerable"]:
            continue

        expected_sources = case.get(
            "relevant_sources",
            []
        )

        if not expected_sources:
            continue

        total += 1

        question = case["question"]

        rewritten = rewrite_query(question)

        original_results = hybrid_retriever.search(
            question=question,
            vector_results=vector_search(
                collection=collection,
                question=question,
                top_k=TOP_K
            ),
            top_k=TOP_K
        )

        rewritten_results = hybrid_retriever.search(
            question=rewritten,
            vector_results=vector_search(
                collection=collection,
                question=rewritten,
                top_k=TOP_K
            ),
            top_k=TOP_K
        )

        original_hit = source_match(
            original_results,
            expected_sources
        )

        rewritten_hit = source_match(
            rewritten_results,
            expected_sources
        )

        original_hits += original_hit
        rewritten_hits += rewritten_hit

        print()
        print("-" * 80)
        print(f"Question: {question}")
        print(f"Rewritten: {rewritten}")
        print(f"Original hit:   {original_hit}")
        print(f"Rewritten hit:  {rewritten_hit}")

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)

    if total:

        print(
            f"Original retrieval:  "
            f"{original_hits}/{total} "
            f"({original_hits / total:.3f})"
        )

        print(
            f"Rewritten retrieval: "
            f"{rewritten_hits}/{total} "
            f"({rewritten_hits / total:.3f})"
        )

        print(
            f"Difference: "
            f"{(rewritten_hits - original_hits) / total:+.3f}"
        )


if __name__ == "__main__":
    main()