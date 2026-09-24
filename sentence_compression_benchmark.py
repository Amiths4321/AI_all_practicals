import json
import time

from chunking import chunk_text
from context_compression import compress_context
from evidence_compression import EvidenceAwareCompressor
from sentence_compression import SentenceCompressor
from generation import generate_answer
from rag_judge import judge_answer

from hybrid_search import HybridRetriever
from reranking import DocumentReranker

import chromadb
from chromadb.config import Settings


DOCUMENT_FILE = "documents.json"
EVALUATION_FILE = "evaluation_data.json"

OUTPUT_FILE = "sentence_compression_benchmark.json"

CHUNK_SIZE = 500
OVERLAP = 100

VECTOR_K = 10
HYBRID_K = 10
RERANK_K = 10

TOP_N = 3
EVIDENCE_THRESHOLD = 0.60

SENTENCE_THRESHOLD = 0.45


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

        for index, chunk in enumerate(
            document_chunks
        ):

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
        name=f"sentence_benchmark_{int(time.time())}"
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


def retrieve(
    question,
    collection,
    hybrid_retriever
):

    vector_results = vector_search(
        collection=collection,
        question=question,
        top_k=VECTOR_K
    )

    keyword_results = (
        hybrid_retriever.keyword_search(
            question=question,
            top_k=HYBRID_K
        )
    )

    hybrid_results = (
        hybrid_retriever.reciprocal_rank_fusion(
            vector_results=vector_results,
            keyword_results=keyword_results,
            top_k=HYBRID_K
        )
    )

    return hybrid_results


def build_context(documents):

    parts = []

    for index, item in enumerate(
        documents,
        start=1
    ):

        metadata = item.get(
            "metadata",
            {}
        ) or {}

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

    evaluation_data = (
        load_evaluation_data()
    )

    chunks = build_chunks(
        documents
    )

    collection = create_collection(
        chunks
    )

    hybrid_retriever = (
        HybridRetriever(
            documents=chunks
        )
    )

    reranker = DocumentReranker()

    evidence_compressor = (
        EvidenceAwareCompressor(
            threshold=EVIDENCE_THRESHOLD
        )
    )

    sentence_compressor = (
        SentenceCompressor(
            threshold=SENTENCE_THRESHOLD
        )
    )

    results = []

    for case in evaluation_data:

        if not case.get(
            "answerable",
            False
        ):
            continue

        question = case["question"]

        print()
        print("=" * 80)
        print(question)
        print("=" * 80)

        hybrid_results = retrieve(
            question=question,
            collection=collection,
            hybrid_retriever=hybrid_retriever
        )

        reranked = reranker.rerank(
            question=question,
            documents=hybrid_results,
            top_n=RERANK_K
        )

        # --------------------------------------------------
        # Strategy 1: Top-N
        # --------------------------------------------------

        top_n_documents = compress_context(
            reranked_documents=reranked,
            max_documents=TOP_N
        )

        # --------------------------------------------------
        # Strategy 2: Evidence-aware
        # --------------------------------------------------

        required_evidence = case.get(
            "required_evidence",
            []
        )

        evidence_documents = (
            evidence_compressor.compress(
                required_evidence=required_evidence,
                reranked_documents=reranked,
                max_documents=TOP_N
            )
        )

        # --------------------------------------------------
        # Strategy 3: Sentence-level
        # --------------------------------------------------

        sentence_items = (
            sentence_compressor.compress(
                question=question,
                documents=reranked
            )
        )

        sentence_documents = (
            sentence_compressor
            .reconstruct_documents(
                sentence_items
            )
        )

        strategies = {
            "top_n": top_n_documents,
            "evidence_aware": evidence_documents,
            "sentence": sentence_documents
        }

        question_results = {}

        for strategy, selected_documents in (
            strategies.items()
        ):

            context = build_context(
                selected_documents
            )

            start = time.perf_counter()

            answer = generate_answer(
                question=question,
                documents=selected_documents
            )

            latency = (
                time.perf_counter()
                - start
            )

            judgement = judge_answer(
                question=question,
                context=context,
                answer=answer
            )

            question_results[strategy] = {
                "answer": answer,
                "documents_used": len(
                    selected_documents
                ),
                "context_characters": len(
                    context
                ),
                "latency_seconds": latency,
                "judgement": judgement
            }

            print(
                f"{strategy:16} | "
                f"docs={len(selected_documents):2} | "
                f"context={len(context):5} | "
                f"latency={latency:.2f}s | "
                f"ground={judgement['groundedness']} | "
                f"complete={judgement['completeness']}"
            )

        results.append({
            "question": question,
            "strategies": question_results
        })

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
    print(
        f"Saved results to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()