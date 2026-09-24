import json
import time

import chromadb
from chromadb.config import Settings

from chunking import chunk_text
from hybrid_search import HybridRetriever
from reranking import DocumentReranker
from semantic_evidence import SemanticEvidenceEvaluator
from generation import generate_answer
from rag_judge import judge_answer


DOCUMENT_FILE = "documents.json"
EVALUATION_FILE = "evaluation_data.json"
OUTPUT_FILE = "context_ordering_results.json"

CHUNK_SIZE = 500
OVERLAP = 100

VECTOR_K = 10
HYBRID_K = 10
RERANK_K = 10

EVIDENCE_THRESHOLD = 0.60


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
        name=f"context_order_{int(time.time())}"
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
        collection,
        question,
        VECTOR_K
    )

    keyword_results = (
        hybrid_retriever.keyword_search(
            question,
            top_k=HYBRID_K
        )
    )

    hybrid_results = (
        hybrid_retriever.reciprocal_rank_fusion(
            vector_results,
            keyword_results,
            top_k=HYBRID_K
        )
    )

    return reranker.rerank(
        question=question,
        documents=hybrid_results,
        top_n=RERANK_K
    )


def find_evidence_chunk(
    question,
    documents,
    required_evidence,
    evaluator
):

    if not required_evidence:
        return None

    best_document = None
    best_score = -1.0

    for document in documents:

        best_document_score = 0.0

        for evidence in required_evidence:

            score = evaluator.score(
                evidence=evidence,
                document=document["document"]
            )

            best_document_score = max(
                best_document_score,
                score
            )

        if best_document_score > best_score:

            best_score = best_document_score
            best_document = document

    if (
        best_document is None
        or best_score < EVIDENCE_THRESHOLD
    ):
        return None

    return best_document


def order_original(documents):
    return documents.copy()


def order_evidence_first(
    documents,
    evidence_id
):

    evidence = None
    others = []

    for document in documents:

        if document["id"] == evidence_id:
            evidence = document
        else:
            others.append(document)

    if evidence is None:
        return documents.copy()

    return [evidence] + others


def order_evidence_last(
    documents,
    evidence_id
):

    evidence = None
    others = []

    for document in documents:

        if document["id"] == evidence_id:
            evidence = document
        else:
            others.append(document)

    if evidence is None:
        return documents.copy()

    return others + [evidence]


def order_alternating(
    documents,
    evidence_id
):

    documents = documents.copy()

    evidence = None

    for document in documents:

        if document["id"] == evidence_id:
            evidence = document
            break

    if evidence is None:
        return documents

    remaining = [
        document
        for document in documents
        if document["id"] != evidence_id
    ]

    result = []

    left = 0
    right = len(remaining) - 1

    while left <= right:

        result.append(
            remaining[left]
        )
        left += 1

        if left <= right:

            result.append(
                remaining[right]
            )
            right -= 1

    # Put the strongest evidence first.
    return [evidence] + result


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

    evidence_evaluator = (
        SemanticEvidenceEvaluator()
    )

    results = []

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
        print("=" * 80)
        print(question)
        print("=" * 80)

        reranked = retrieve_and_rerank(
            question=question,
            collection=collection,
            hybrid_retriever=hybrid_retriever,
            reranker=reranker
        )

        evidence_document = find_evidence_chunk(
            question=question,
            documents=reranked,
            required_evidence=required_evidence,
            evaluator=evidence_evaluator
        )

        if evidence_document is None:

            print(
                "No evidence-bearing chunk "
                "found above threshold."
            )

            continue

        evidence_id = evidence_document["id"]

        strategies = {
            "original": order_original(
                reranked
            ),

            "evidence_first": order_evidence_first(
                reranked,
                evidence_id
            ),

            "evidence_last": order_evidence_last(
                reranked,
                evidence_id
            ),

            "alternating": order_alternating(
                reranked,
                evidence_id
            )
        }

        question_results = {}

        for strategy, ordered_documents in (
            strategies.items()
        ):

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

            evidence_index = next(
                index
                for index, document
                in enumerate(
                    ordered_documents
                )
                if document["id"] == evidence_id
            )

            question_results[strategy] = {
                "evidence_index": evidence_index,
                "context_characters": len(
                    context
                ),
                "latency_seconds": latency,
                "answer": answer,
                "judgement": judgement
            }

            print(
                f"{strategy:18} | "
                f"evidence_index={evidence_index:2} | "
                f"ground={judgement['groundedness']} | "
                f"complete={judgement['completeness']} | "
                f"relevant={judgement['relevance']}"
            )

        results.append({
            "question": question,
            "evidence_id": evidence_id,
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