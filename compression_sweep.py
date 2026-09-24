import json
import time

from chunking import chunk_text
from semantic_evidence import SemanticEvidenceEvaluator
from context_compression import compress_context
from generation import generate_answer
from rag_judge import judge_answer

from chromadb.config import Settings
import chromadb

from hybrid_search import HybridRetriever
from reranking import DocumentReranker


DOCUMENT_FILE = "documents.json"
EVALUATION_FILE = "evaluation_data.json"

CHUNK_SIZE = 500
OVERLAP = 100

VECTOR_K = 10
HYBRID_K = 10
RERANK_K = 10

COMPRESSION_VALUES = [1, 2, 3, 5, 10]

OUTPUT_FILE = "compression_sweep.json"


def load_documents():
    with open(DOCUMENT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_evaluation_data():
    with open(EVALUATION_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


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
        Settings(anonymized_telemetry=False)
    )

    collection = client.create_collection(
        name=f"compression_sweep_{int(time.time())}"
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

    documents = []

    for i in range(len(results["ids"][0])):
        documents.append({
            "id": results["ids"][0][i],
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": float(results["distances"][0][i])
        })

    return documents


def retrieve(
    question,
    collection,
    hybrid_retriever,
    vector_k,
    hybrid_k
):
    vector_results = vector_search(
        collection,
        question,
        vector_k
    )

    keyword_results = hybrid_retriever.keyword_search(
        question,
        top_k=hybrid_k
    )

    hybrid_results = hybrid_retriever.reciprocal_rank_fusion(
        vector_results,
        keyword_results,
        top_k=hybrid_k
    )

    return hybrid_results


def rerank_documents(
    question,
    documents,
    reranker,
    rerank_k
):
    return reranker.rerank(
        question=question,
        documents=documents,
        top_n=rerank_k
    )


def build_context(documents):
    parts = []

    for index, item in enumerate(documents, start=1):
        source = item["metadata"].get(
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

    chunks = build_chunks(documents)

    collection = create_collection(chunks)

    hybrid_retriever = HybridRetriever(
        documents=chunks
    )

    reranker = DocumentReranker()

    results = []

    for case in evaluation_data:

        if not case.get("answerable", False):
            continue

        question = case["question"]

        print()
        print("=" * 70)
        print(question)
        print("=" * 70)

        hybrid_results = retrieve(
            question=question,
            collection=collection,
            hybrid_retriever=hybrid_retriever,
            vector_k=VECTOR_K,
            hybrid_k=HYBRID_K
        )

        reranked = rerank_documents(
            question=question,
            documents=hybrid_results,
            reranker=reranker,
            rerank_k=RERANK_K
        )

        for max_documents in COMPRESSION_VALUES:

            compressed = compress_context(
                reranked_documents=reranked,
                max_documents=max_documents
            )

            context = build_context(compressed)

            start = time.perf_counter()

            answer = generate_answer(
                question=question,
                documents=compressed
            )

            latency = time.perf_counter() - start

            judgement = judge_answer(
                question=question,
                context=context,
                answer=answer
            )

            result = {
                "question": question,
                "max_documents": max_documents,
                "documents_used": len(compressed),
                "context_characters": len(context),
                "latency_seconds": latency,
                "answer": answer,
                "judgement": judgement
            }

            results.append(result)

            print(
                f"max_documents={max_documents} | "
                f"context={len(context)} chars | "
                f"latency={latency:.2f}s | "
                f"groundedness={judgement['groundedness']} | "
                f"completeness={judgement['completeness']}"
            )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            results,
            f,
            indent=2
        )

    print()
    print(f"Saved results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()