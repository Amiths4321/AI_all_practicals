import json
import os
import shutil

import chromadb

from chunking import chunk_text
from hybrid_search import HybridRetriever
from reranking import DocumentReranker


CHUNK_SIZE = 500
OVERLAP = 100

RETRIEVAL_K = 10
FINAL_K = 5

DB_PATH = "./reranker_benchmark_chroma"


def load_documents():
    with open("documents.json", "r", encoding="utf-8") as file:
        return json.load(file)


def load_evaluation_data():
    with open("rag_eval.json", "r", encoding="utf-8") as file:
        return json.load(file)


def build_chunks(documents):
    chunks = []

    for document in documents:
        source = document["source"]
        text = document["text"]

        document_chunks = chunk_text(
            text,
            chunk_size=CHUNK_SIZE,
            overlap=OVERLAP
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


def create_collection(chunks):
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    client = chromadb.PersistentClient(
        path=DB_PATH
    )

    collection = client.get_or_create_collection(
        name="documents"
    )

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

    output = []

    for doc_id, document, metadata in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0]
    ):
        output.append({
            "id": doc_id,
            "document": document,
            "metadata": metadata or {}
        })

    return output


def precision_at_k(retrieved, relevant_sources, k):
    retrieved = retrieved[:k]

    if not retrieved:
        return 0.0

    relevant = sum(
        1
        for item in retrieved
        if item["metadata"].get("source") in relevant_sources
    )

    return relevant / len(retrieved)


def recall_at_k(retrieved, relevant_sources, k):
    if not relevant_sources:
        return 1.0

    retrieved = retrieved[:k]

    retrieved_sources = {
        item["metadata"].get("source")
        for item in retrieved
    }

    found = len(
        retrieved_sources.intersection(relevant_sources)
    )

    return found / len(relevant_sources)


def reciprocal_rank(retrieved, relevant_sources):
    for rank, item in enumerate(
        retrieved,
        start=1
    ):
        source = item["metadata"].get("source")

        if source in relevant_sources:
            return 1 / rank

    return 0.0


def evaluate(
    evaluation_data,
    search_function,
    k
):
    precisions = []
    recalls = []
    reciprocal_ranks = []

    for item in evaluation_data:
        question = item["question"]

        relevant_sources = set(
            item.get("relevant_sources", [])
        )

        retrieved = search_function(question)

        precisions.append(
            precision_at_k(
                retrieved,
                relevant_sources,
                k
            )
        )

        recalls.append(
            recall_at_k(
                retrieved,
                relevant_sources,
                k
            )
        )

        reciprocal_ranks.append(
            reciprocal_rank(
                retrieved,
                relevant_sources
            )
        )

    return {
        "precision": sum(precisions) / len(precisions),
        "recall": sum(recalls) / len(recalls),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks)
    }


def main():
    documents = load_documents()
    evaluation_data = load_evaluation_data()

    chunks = build_chunks(documents)

    print(f"Total chunks: {len(chunks)}")

    collection = create_collection(chunks)

    hybrid_retriever = HybridRetriever(chunks)

    reranker = DocumentReranker()

    # --------------------------------------------------
    # HYBRID WITHOUT RERANKING
    # --------------------------------------------------

    def hybrid_only(question):
        vector_results = vector_search(
            collection,
            question,
            RETRIEVAL_K
        )

        return hybrid_retriever.search(
            question=question,
            vector_results=vector_results,
            top_k=FINAL_K
        )

    # --------------------------------------------------
    # HYBRID + RERANKING
    # --------------------------------------------------

    def hybrid_reranked(question):
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

        return reranked_results

    print()
    print("=" * 70)
    print("HYBRID WITHOUT RERANKING")
    print("=" * 70)

    baseline = evaluate(
        evaluation_data,
        hybrid_only,
        FINAL_K
    )

    print(
        f"Precision@{FINAL_K}: "
        f"{baseline['precision']:.3f}"
    )

    print(
        f"Recall@{FINAL_K}:    "
        f"{baseline['recall']:.3f}"
    )

    print(
        f"MRR:               "
        f"{baseline['mrr']:.3f}"
    )

    print()
    print("=" * 70)
    print("HYBRID + CROSS-ENCODER")
    print("=" * 70)

    reranked = evaluate(
        evaluation_data,
        hybrid_reranked,
        FINAL_K
    )

    print(
        f"Precision@{FINAL_K}: "
        f"{reranked['precision']:.3f}"
    )

    print(
        f"Recall@{FINAL_K}:    "
        f"{reranked['recall']:.3f}"
    )

    print(
        f"MRR:               "
        f"{reranked['mrr']:.3f}"
    )

    print()
    print("=" * 70)
    print("DELTA")
    print("=" * 70)

    print(
        f"Precision delta: "
        f"{reranked['precision'] - baseline['precision']:+.3f}"
    )

    print(
        f"Recall delta:    "
        f"{reranked['recall'] - baseline['recall']:+.3f}"
    )

    print(
        f"MRR delta:       "
        f"{reranked['mrr'] - baseline['mrr']:+.3f}"
    )

def inspect_question(
    question,
    hybrid_results,
    reranked_results
):
    print()
    print("=" * 80)
    print("QUESTION")
    print(question)

    print()
    print("BEFORE RERANKING")

    for rank, item in enumerate(
        hybrid_results,
        start=1
    ):
        print(
            f"{rank}. "
            f"{item['id']} | "
            f"{item['metadata'].get('source')}"
        )

    print()
    print("AFTER RERANKING")

    for rank, item in enumerate(
        reranked_results,
        start=1
    ):
        print(
            f"{rank}. "
            f"{item['id']} | "
            f"{item['metadata'].get('source')} | "
            f"score={item['rerank_score']:.4f}"
        )

if __name__ == "__main__":
    main()