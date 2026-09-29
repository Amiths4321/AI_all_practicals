import json
import time
from pathlib import Path

from chunking import chunk_text
from hybrid_search import HybridRetriever
from reranking import DocumentReranker
from semantic_evidence import SemanticEvidenceEvaluator
from context_builder import ContextBuilder
from generation import generate_answer
from rag_judge import judge_answer


BASE_DIR = Path(__file__).resolve().parent

DOCUMENT_FILES = [
    BASE_DIR / "documents" / "leave_policy.txt",
    BASE_DIR / "documents" / "employee_handbook.txt"
]

EVALUATION_FILE = (
    BASE_DIR / "evaluation_data.json"
)

OUTPUT_FILE = (
    BASE_DIR / "performance_profile.json"
)

CHUNK_SIZE = 500
OVERLAP = 100

VECTOR_K = 10
HYBRID_K = 10
RERANK_K = 10

BUDGET = 3
EVIDENCE_THRESHOLD = 0.60


def load_documents():

    documents = []

    for filename in DOCUMENT_FILES:

        with open(
            filename,
            "r",
            encoding="utf-8"
        ) as file:

            text = file.read()

        documents.append({
            "id": filename.name,
            "document": text,
            "metadata": {
                "source": str(filename)
            }
        })

    return documents


def load_evaluation_data():

    with open(
        EVALUATION_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def build_chunks(documents):

    chunks = []

    for document in documents:

        document_chunks = chunk_text(
            document["document"],
            chunk_size=CHUNK_SIZE,
            overlap=OVERLAP
        )

        for index, chunk in enumerate(
            document_chunks
        ):

            chunks.append({
                "id": (
                    f"{document['id']}"
                    f"_chunk_{index}"
                ),

                "document": chunk,

                "metadata": {
                    "source": str(
                        document["metadata"]["source"]
                    ),

                    "chunk_index": index
                }
            })

    return chunks


def create_collection(chunks):

    import chromadb

    client = chromadb.Client()

    collection = client.create_collection(
        name=(
            f"profile_"
            f"{int(time.time() * 1000)}"
        )
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
    question
):

    result = collection.query(
        query_texts=[question],
        n_results=VECTOR_K
    )

    results = []

    for document_id, document, metadata in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0]
    ):

        results.append({
            "id": document_id,
            "document": document,
            "metadata": metadata or {}
        })

    return results


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


def profile_question(
    case,
    collection,
    hybrid_retriever,
    reranker,
    evidence_evaluator,
    context_builder
):

    question = case["question"]

    required_evidence = case.get(
        "required_evidence",
        []
    )

    timings = {}

    question_start = time.perf_counter()

    # --------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------

    start = time.perf_counter()

    vector_results = vector_search(
        collection=collection,
        question=question
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

    timings["retrieval_seconds"] = (
        time.perf_counter() - start
    )

    # --------------------------------------------------
    # RERANKING
    # --------------------------------------------------

    start = time.perf_counter()

    reranked_documents = (
        reranker.rerank(
            question=question,
            documents=hybrid_results,
            top_n=RERANK_K
        )
    )

    timings["reranking_seconds"] = (
        time.perf_counter() - start
    )

    # --------------------------------------------------
    # EVIDENCE EVALUATION
    # --------------------------------------------------

    start = time.perf_counter()

    evidence_result = (
        evidence_evaluator.evaluate(
            required_evidence=required_evidence,
            retrieved_documents=reranked_documents
        )
    )

    timings["evidence_seconds"] = (
        time.perf_counter() - start
    )

    # --------------------------------------------------
    # CONTEXT BUILDING
    # --------------------------------------------------

    start = time.perf_counter()

    selected_documents = (
        context_builder.build(
            strategy="evidence_first",
            question=question,
            reranked_documents=reranked_documents,
            required_evidence=required_evidence,
            max_documents=BUDGET
        )
    )

    context = build_context(
        selected_documents
    )

    timings["context_seconds"] = (
        time.perf_counter() - start
    )

    # --------------------------------------------------
    # GENERATION
    # --------------------------------------------------

    start = time.perf_counter()

    answer = generate_answer(
        question,
        selected_documents
    )

    timings["generation_seconds"] = (
        time.perf_counter() - start
    )

    # --------------------------------------------------
    # JUDGING
    # --------------------------------------------------

    start = time.perf_counter()

    judgement = judge_answer(
        question=question,
        context=context,
        answer=answer
    )

    timings["judge_seconds"] = (
        time.perf_counter() - start
    )

    # --------------------------------------------------
    # TOTAL
    # --------------------------------------------------

    timings["total_seconds"] = (
        time.perf_counter()
        - question_start
    )

    return {
        "question": question,
        "timings": timings,
        "answer": answer,
        "judgement": judgement,
        "documents_used": len(
            selected_documents
        ),
        "context_characters": len(
            context
        ),
        "evidence_recall":
            evidence_result["evidence_recall"]
    }


def main():

    print("=" * 70)
    print("RAG PERFORMANCE PROFILER")
    print("=" * 70)

    documents = load_documents()

    evaluation_data = (
        load_evaluation_data()
    )

    chunks = build_chunks(
        documents
    )

    print(
        f"Documents: {len(documents)}"
    )

    print(
        f"Chunks: {len(chunks)}"
    )

    print(
        f"Questions: {len(evaluation_data)}"
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

    context_builder = ContextBuilder(
        evidence_threshold=EVIDENCE_THRESHOLD
    )

    results = []

    for index, case in enumerate(
        evaluation_data,
        start=1
    ):

        print()
        print(
            f"[{index}/{len(evaluation_data)}] "
            f"{case['question']}"
        )

        result = profile_question(
            case=case,
            collection=collection,
            hybrid_retriever=hybrid_retriever,
            reranker=reranker,
            evidence_evaluator=evidence_evaluator,
            context_builder=context_builder
        )

        results.append(result)

        print(
            "Total:",
            round(
                result["timings"]["total_seconds"],
                3
            ),
            "seconds"
        )

    # --------------------------------------------------
    # AGGREGATE
    # --------------------------------------------------

    timing_keys = [
        "retrieval_seconds",
        "reranking_seconds",
        "evidence_seconds",
        "context_seconds",
        "generation_seconds",
        "judge_seconds",
        "total_seconds"
    ]

    averages = {}

    for key in timing_keys:

        values = [
            result["timings"][key]
            for result in results
        ]

        averages[key] = (
            sum(values) / len(values)
            if values
            else 0.0
        )

    report = {
        "configuration": {
            "chunk_size": CHUNK_SIZE,
            "overlap": OVERLAP,
            "vector_k": VECTOR_K,
            "hybrid_k": HYBRID_K,
            "rerank_k": RERANK_K,
            "budget": BUDGET
        },

        "averages": averages,

        "question_results": results
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=2
        )

    print()
    print("=" * 70)
    print("AVERAGE LATENCY")
    print("=" * 70)

    for key, value in averages.items():

        print(
            f"{key}: "
            f"{value:.3f}s"
        )

    print()
    print(
        f"Saved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()