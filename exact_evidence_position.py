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
OUTPUT_FILE = "exact_evidence_position.json"

CHUNK_SIZE = 500
OVERLAP = 100

VECTOR_K = 10
HYBRID_K = 10
RERANK_K = 10

EVIDENCE_THRESHOLD = 0.60

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
        name=f"exact_evidence_{int(time.time())}"
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

    return reranker.rerank(
        question=question,
        documents=hybrid_results,
        top_n=RERANK_K
    )


def find_exact_evidence_chunk(
    required_evidence,
    documents,
    evaluator
):

    if not required_evidence:
        return None

    best_document = None
    best_score = -1.0

    for document in documents:

        document_score = 0.0

        for evidence in required_evidence:

            score = evaluator.score(
                evidence=evidence,
                document=document["document"]
            )

            document_score = max(
                document_score,
                score
            )

        if document_score > best_score:

            best_score = document_score
            best_document = document

    if (
        best_document is None
        or best_score < EVIDENCE_THRESHOLD
    ):
        return None

    return {
        "document": best_document,
        "score": best_score
    }


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


def move_evidence_chunk(
    documents,
    evidence_id,
    position
):

    remaining = []
    evidence_document = None

    for item in documents:

        if item["id"] == evidence_id:
            evidence_document = item
        else:
            remaining.append(item)

    if evidence_document is None:
        raise ValueError(
            f"Evidence document {evidence_id} "
            "was not found."
        )

    target_index = position_index(
        len(remaining) + 1,
        position
    )

    remaining.insert(
        target_index,
        evidence_document
    )

    return remaining

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

    chunks = build_chunks(documents)

    collection = create_collection(
        chunks
    )

    hybrid_retriever = HybridRetriever(
        documents=chunks
    )

    reranker = DocumentReranker()

    evaluator = SemanticEvidenceEvaluator()

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

        evidence_match = (
            find_exact_evidence_chunk(
                required_evidence=required_evidence,
                documents=reranked,
                evaluator=evaluator
            )
        )

        if evidence_match is None:

            print(
                "Exact evidence chunk not found "
                "above the threshold."
            )

            continue

        evidence_document = (
            evidence_match["document"]
        )

        evidence_id = evidence_document["id"]

        print(
            f"Evidence chunk: {evidence_id}"
        )

        print(
            f"Evidence similarity: "
            f"{evidence_match['score']:.3f}"
        )

        question_results = {}

        for position in POSITIONS:

            ordered_documents = (
                move_evidence_chunk(
                    documents=reranked,
                    evidence_id=evidence_id,
                    position=position
                )
            )

            actual_index = next(
                index
                for index, item
                in enumerate(
                    ordered_documents
                )
                if item["id"] == evidence_id
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
                "evidence_id": evidence_id,
                "evidence_similarity": (
                    evidence_match["score"]
                ),
                "evidence_index": actual_index,
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
                f"unsupported={judgement['unsupported_claims']}"
            )

        results.append({
            "question": question,
            "required_evidence": required_evidence,
            "evidence_id": evidence_id,
            "evidence_similarity": (
                evidence_match["score"]
            ),
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