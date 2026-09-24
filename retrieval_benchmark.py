import json
import os
import shutil

import chromadb

from chunking import chunk_text
from hybrid_search import HybridRetriever


CHUNK_CONFIGS = [
    {"chunk_size": 200, "overlap": 40},
    {"chunk_size": 300, "overlap": 60},
    {"chunk_size": 500, "overlap": 100},
    {"chunk_size": 800, "overlap": 160},
]

TOP_K = 5
DB_ROOT = "./retrieval_benchmark_chroma"


def load_documents():
    with open("documents.json", "r", encoding="utf-8") as file:
        return json.load(file)


def load_evaluation_data():
    with open("rag_eval.json", "r", encoding="utf-8") as file:
        return json.load(file)


def build_chunks(documents, chunk_size, overlap):
    chunks = []

    for document in documents:
        source = document["source"]
        text = document["text"]

        document_chunks = chunk_text(
            text,
            chunk_size=chunk_size,
            overlap=overlap
        )

        for chunk_index, chunk in enumerate(document_chunks):
            chunks.append({
                "id": f"{source}_chunk_{chunk_index}",
                "document": chunk,
                "metadata": {
                    "source": source,
                    "chunk_index": chunk_index
                }
            })

    return chunks


def create_collection(chunks, collection_name):
    path = os.path.join(DB_ROOT, collection_name)

    client = chromadb.PersistentClient(path=path)

    collection = client.get_or_create_collection(
        name="documents"
    )

    if collection.count() > 0:
        collection.delete(where={})

    collection.add(
        ids=[item["id"] for item in chunks],
        documents=[item["document"] for item in chunks],
        metadatas=[item["metadata"] for item in chunks]
    )

    return collection


def vector_search(collection, question, top_k):
    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )

    ids = results["ids"][0]

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    results = []

    for doc_id, document, metadata in zip(
        ids,
        documents,
        metadatas
    ):
        results.append({
            "id": doc_id,
            "document": document,
            "metadata": metadata or {}
        })

    return results


def evaluate(retrieved_sources, relevant_sources, k):
    retrieved = retrieved_sources[:k]

    if not retrieved:
        precision = 0.0
    else:
        relevant_count = sum(
            1
            for source in retrieved
            if source in relevant_sources
        )

        precision = relevant_count / len(retrieved)

    if not relevant_sources:
        recall = 1.0
    else:
        relevant_count = sum(
            1
            for source in retrieved
            if source in relevant_sources
        )

        recall = relevant_count / len(relevant_sources)

    reciprocal_rank = 0.0

    for rank, source in enumerate(
        retrieved_sources,
        start=1
    ):
        if source in relevant_sources:
            reciprocal_rank = 1 / rank
            break

    return precision, recall, reciprocal_rank


def evaluate_method(
    questions,
    search_function,
    k
):
    precisions = []
    recalls = []
    reciprocal_ranks = []

    for item in questions:
        question = item["question"]

        relevant_sources = set(item.get("relevant_sources", []))

        results = search_function(
            question,
            k
        )

        retrieved_sources = [
            result["metadata"].get("source")
            for result in results
        ]

        precision, recall, rr = evaluate(
            retrieved_sources,
            relevant_sources,
            k
        )

        precisions.append(precision)
        recalls.append(recall)
        reciprocal_ranks.append(rr)

    return {
        "precision": sum(precisions) / len(precisions),
        "recall": sum(recalls) / len(recalls),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks)
    }


def main():
    documents = load_documents()
    questions = load_evaluation_data()

    if os.path.exists(DB_ROOT):
        shutil.rmtree(DB_ROOT)

    os.makedirs(DB_ROOT)

    all_results = []

    for config in CHUNK_CONFIGS:

        chunk_size = config["chunk_size"]
        overlap = config["overlap"]

        print()
        print("=" * 70)
        print(
            f"CHUNK SIZE: {chunk_size} | "
            f"OVERLAP: {overlap}"
        )
        print("=" * 70)

        chunks = build_chunks(
            documents,
            chunk_size,
            overlap
        )

        print(f"Chunks: {len(chunks)}")

        collection = create_collection(
            chunks,
            f"chunks_{chunk_size}_{overlap}"
        )

        # ------------------------------------------------
        # VECTOR SEARCH
        # ------------------------------------------------

        vector_metrics = evaluate_method(
            questions,
            lambda question, k:
                vector_search(
                    collection,
                    question,
                    k
                ),
            TOP_K
        )

        print()
        print("Vector Search")
        print(
            f"Precision@{TOP_K}: "
            f"{vector_metrics['precision']:.3f}"
        )
        print(
            f"Recall@{TOP_K}:    "
            f"{vector_metrics['recall']:.3f}"
        )
        print(
            f"MRR:               "
            f"{vector_metrics['mrr']:.3f}"
        )

        # ------------------------------------------------
        # BM25 + HYBRID
        # ------------------------------------------------

        hybrid_retriever = HybridRetriever(
            chunks
        )

        def hybrid_search(question, k):
            vector_results = vector_search(
                collection,
                question,
                k
            )

            return hybrid_retriever.search(
                question=question,
                vector_results=vector_results,
                top_k=k
            )

        # BM25 only
        def bm25_search(question, k):
            return hybrid_retriever.keyword_search(
                question=question,
                top_k=k
            )

        bm25_metrics = evaluate_method(
            questions,
            bm25_search,
            TOP_K
        )

        hybrid_metrics = evaluate_method(
            questions,
            hybrid_search,
            TOP_K
        )

        print()
        print("BM25")
        print(
            f"Precision@{TOP_K}: "
            f"{bm25_metrics['precision']:.3f}"
        )
        print(
            f"Recall@{TOP_K}:    "
            f"{bm25_metrics['recall']:.3f}"
        )
        print(
            f"MRR:               "
            f"{bm25_metrics['mrr']:.3f}"
        )

        print()
        print("Hybrid RRF")
        print(
            f"Precision@{TOP_K}: "
            f"{hybrid_metrics['precision']:.3f}"
        )
        print(
            f"Recall@{TOP_K}:    "
            f"{hybrid_metrics['recall']:.3f}"
        )
        print(
            f"MRR:               "
            f"{hybrid_metrics['mrr']:.3f}"
        )

        all_results.extend([
            {
                "chunk_size": chunk_size,
                "overlap": overlap,
                "method": "vector",
                **vector_metrics
            },
            {
                "chunk_size": chunk_size,
                "overlap": overlap,
                "method": "bm25",
                **bm25_metrics
            },
            {
                "chunk_size": chunk_size,
                "overlap": overlap,
                "method": "hybrid",
                **hybrid_metrics
            }
        ])

    # ----------------------------------------------------
    # FINAL SUMMARY
    # ----------------------------------------------------

    print()
    print()
    print("=" * 90)
    print("FINAL RETRIEVAL BENCHMARK")
    print("=" * 90)

    print(
        f"{'Chunk':>7} "
        f"{'Overlap':>8} "
        f"{'Method':>10} "
        f"{'P@5':>8} "
        f"{'R@5':>8} "
        f"{'MRR':>8}"
    )

    print("-" * 90)

    for result in all_results:
        print(
            f"{result['chunk_size']:>7} "
            f"{result['overlap']:>8} "
            f"{result['method']:>10} "
            f"{result['precision']:>8.3f} "
            f"{result['recall']:>8.3f} "
            f"{result['mrr']:>8.3f}"
        )

question = evaluation_data[0]["question"]

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
    top_n=FINAL_K
)

inspect_question(
    question,
    hybrid_results,
    reranked_results
)

if __name__ == "__main__":
    main()