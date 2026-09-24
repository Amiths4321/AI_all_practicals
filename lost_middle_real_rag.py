import json
import time

import chromadb
from chromadb.config import Settings

from chunking import chunk_text
from hybrid_search import HybridRetriever
from reranking import DocumentReranker
from generation import generate_answer
from rag_judge import judge_answer


DOCUMENT_FILE = "documents.json"
EVALUATION_FILE = "evaluation_data.json"
OUTPUT_FILE = "lost_middle_real_rag.json"

CHUNK_SIZE = 500
OVERLAP = 100

VECTOR_K = 10
HYBRID_K = 10
RERANK_K = 10

POSITIONS = [
    "beginning",
    "quarter",
    "middle",
    "three_quarters",
    "end"
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
        name=f"lost_middle_{int(time.time())}"
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

    reranked = reranker.rerank(
        question=question,
        documents=hybrid_results,
        top_n=RERANK_K
    )

    return reranked


def find_relevant_index(
    documents,
    relevant_sources
):

    relevant_sources = set(
        relevant_sources
    )

    for index, document in enumerate(
        documents
    ):

        metadata = (
            document.get("metadata", {})
            or {}
        )

        source = metadata.get(
            "source"
        )

        if source in relevant_sources:
            return index

    return None


def position_index(
    total_documents,
    position
):

    if position == "beginning":
        return 0

    if position == "quarter":
        return total_documents // 4

    if position == "middle":
        return total_documents // 2

    if position == "three_quarters":
        return (
            3 * total_documents // 4
        )

    if position == "end":
        return total_documents - 1

    raise ValueError(
        f"Unknown position: {position}"
    )


def move_relevant_document(
    documents,
    relevant_index,
    position
):

    documents = documents.copy()

    relevant_document = documents.pop(
        relevant_index
    )

    target_index = position_index(
        len(documents) + 1,
        position
    )

    documents.insert(
        target_index,
        relevant_document
    )

    return documents


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

    results = []

    for case in evaluation_data:

        if not case.get(
            "answerable",
            False
        ):
            continue

        question = case["question"]

        relevant_sources = case.get(
            "relevant_sources",
            []
        )

        print()
        print("=" * 80)
        print(question)
        print("=" * 80)

        reranked = retrieve_and_rerank(
            question=question,
            collection=collection,
            hybrid_retriever=hybrid_retriever,
            reranker=reranker
        )

        relevant_index = find_relevant_index(
            documents=reranked,
            relevant_sources=relevant_sources
        )

        if relevant_index is None:

            print(
                "Relevant source was not found "
                "in the reranked results."
            )

            continue

        print(
            f"Relevant chunk originally at "
            f"index {relevant_index}"
        )

        question_results = {}

        for position in POSITIONS:

            ordered_documents = (
                move_relevant_document(
                    documents=reranked,
                    relevant_index=relevant_index,
                    position=position
                )
            )

            actual_index = next(
                index
                for index, item
                in enumerate(
                    ordered_documents
                )
                if item["id"] ==
                reranked[relevant_index]["id"]
            )

            context = build_context(
                ordered_documents
            )

            start = time.perf_counter()

            answer = generate_answer(
                question=question,
                documents=ordered_documents
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

            question_results[position] = {
                "relevant_index": actual_index,
                "context_characters": len(
                    context
                ),
                "latency_seconds": latency,
                "answer": answer,
                "judgement": judgement
            }

            print(
                f"{position:16} | "
                f"index={actual_index:2} | "
                f"ground={judgement['groundedness']} | "
                f"complete={judgement['completeness']} | "
                f"relevance={judgement['relevance']}"
            )

        results.append({
            "question": question,
            "relevant_sources": relevant_sources,
            "results": question_results
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