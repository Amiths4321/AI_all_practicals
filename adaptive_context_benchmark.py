import json
import time

from chunking import chunk_text
from hybrid_search import HybridRetriever
from reranking import DocumentReranker
from semantic_evidence import SemanticEvidenceEvaluator
from context_builder import ContextBuilder
from adaptive_context import AdaptiveContextPolicy
from generation import generate_answer
from rag_judge import judge_answer
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

DOCUMENT_FILES = [
    BASE_DIR / "documents" / "leave_policy.txt",
    BASE_DIR / "documents" / "employee_handbook.txt"
]

EVALUATION_FILE = BASE_DIR / "evaluation_data.json"

CHUNK_SIZE = 500
OVERLAP = 100

VECTOR_K = 10
HYBRID_K = 10
RERANK_K = 10

FIXED_BUDGET = 3


def load_documents():

    documents = []

    for filename in DOCUMENT_FILES:

        filename = Path(filename)

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

        for index, chunk in enumerate(document_chunks):

            chunks.append({
                "id": f"{document['id']}_chunk_{index}",
                "document": chunk,
                "metadata": {
                    "source": document["id"],
                    "chunk_index": index
                }
            })

    return chunks


def create_collection(chunks):

    import chromadb

    client = chromadb.Client()

    collection = client.create_collection(
        name=f"adaptive_context_{int(time.time() * 1000)}"
    )

    collection.add(
        ids=[item["id"] for item in chunks],
        documents=[item["document"] for item in chunks],
        metadatas=[item["metadata"] for item in chunks]
    )

    return collection


def vector_search(
    collection,
    question,
    top_k
):

    result = collection.query(
        query_texts=[question],
        n_results=top_k
    )

    documents = result["documents"][0]
    ids = result["ids"][0]
    metadatas = result["metadatas"][0]

    results = []

    for document_id, document, metadata in zip(
        ids,
        documents,
        metadatas
    ):

        results.append({
            "id": document_id,
            "document": document,
            "metadata": metadata or {}
        })

    return results


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

    keyword_results = hybrid_retriever.keyword_search(
        question=question,
        top_k=HYBRID_K
    )

    hybrid_results = hybrid_retriever.reciprocal_rank_fusion(
        vector_results=vector_results,
        keyword_results=keyword_results,
        top_k=HYBRID_K
    )

    reranked_results = reranker.rerank(
        question=question,
        documents=hybrid_results,
        top_n=RERANK_K
    )

    return reranked_results


def build_context(documents):

    parts = []

    for index, item in enumerate(
        documents,
        start=1
    ):

        source = item.get(
            "metadata",
            {}
        ).get(
            "source",
            "unknown"
        )

        parts.append(
            f"[SOURCE {index}: {source}]\n"
            f"{item['document']}"
        )

    return "\n\n".join(parts)


def evaluate_strategy(
    question,
    documents
):

    if not documents:

        return {
            "answer": None,
            "latency_seconds": 0.0,
            "context_characters": 0,
            "judgement": None
        }

    context = build_context(documents)

    start = time.perf_counter()

    answer = generate_answer(
        question,
        documents
    )

    latency = time.perf_counter() - start

    judgement = judge_answer(
        question=question,
        context=context,
        answer=answer
    )

    return {
        "answer": answer,
        "latency_seconds": latency,
        "context_characters": len(context),
        "judgement": judgement
    }


def main():

    documents = load_documents()

    evaluation_data = load_evaluation_data()

    chunks = build_chunks(documents)

    collection = create_collection(chunks)

    hybrid_retriever = HybridRetriever(
        documents=chunks
    )

    reranker = DocumentReranker()

    evidence_evaluator = SemanticEvidenceEvaluator()

    context_builder = ContextBuilder(
        evidence_threshold=0.60
    )

    adaptive_policy = AdaptiveContextPolicy(
        context_builder=context_builder,
        evidence_evaluator=evidence_evaluator,
        budgets=(1, 2, 3, 5),
        evidence_threshold=0.60
    )

    results = []

    for case in evaluation_data:

        question = case["question"]

        required_evidence = case.get(
            "required_evidence",
            []
        )

        answerable = case["answerable"]

        print()
        print("=" * 70)
        print(question)

        reranked_results = retrieve_and_rerank(
            question=question,
            collection=collection,
            hybrid_retriever=hybrid_retriever,
            reranker=reranker
        )

        # --------------------------------------------------
        # FIXED CONTEXT
        # --------------------------------------------------

        fixed_documents = context_builder.build(
            strategy="top_n",
            question=question,
            reranked_documents=reranked_results,
            required_evidence=required_evidence,
            max_documents=FIXED_BUDGET
        )

        fixed_result = evaluate_strategy(
            question=question,
            documents=fixed_documents
        )

        # --------------------------------------------------
        # ADAPTIVE CONTEXT
        # --------------------------------------------------

        adaptive_result = adaptive_policy.build_context(
            question=question,
            required_evidence=required_evidence,
            reranked_documents=reranked_results
        )

        adaptive_documents = adaptive_result["documents"]

        if adaptive_documents:

            adaptive_generation = evaluate_strategy(
                question=question,
                documents=adaptive_documents
            )

        else:

            adaptive_generation = {
                "answer": (
                    "I don't have enough information "
                    "in the provided documents."
                ),
                "latency_seconds": 0.0,
                "context_characters": 0,
                "judgement": None
            }

        question_result = {
            "question": question,
            "answerable": answerable,

            "fixed": {
                "budget": len(fixed_documents),
                **fixed_result
            },

            "adaptive": {
                "budget": len(adaptive_documents),
                "decision": adaptive_result["decision"],
                **adaptive_generation
            }
        }

        results.append(question_result)

        print(
            "Fixed budget:",
            len(fixed_documents)
        )

        print(
            "Adaptive budget:",
            len(adaptive_documents)
        )

        print(
            "Adaptive reason:",
            adaptive_result["decision"]["reason"]
        )

    output = {
        "config": {
            "chunk_size": CHUNK_SIZE,
            "overlap": OVERLAP,
            "vector_k": VECTOR_K,
            "hybrid_k": HYBRID_K,
            "rerank_k": RERANK_K,
            "fixed_budget": FIXED_BUDGET
        },
        "results": results
    }

    with open(
        "adaptive_context_benchmark.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 70)
    print("Saved adaptive_context_benchmark.json")


if __name__ == "__main__":
    main()