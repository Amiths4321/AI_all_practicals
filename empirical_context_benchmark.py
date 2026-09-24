import json
import time
from pathlib import Path

from chunking import chunk_text
from hybrid_search import HybridRetriever
from reranking import DocumentReranker
from semantic_evidence import SemanticEvidenceEvaluator
from context_builder import ContextBuilder
from empirical_context_controller import (
    EmpiricalContextController
)
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

POLICY_FILE = (
    BASE_DIR / "empirical_context_policy.json"
)

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
            f"empirical_context_"
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

    keyword_results = (
        hybrid_retriever.keyword_search(
            question=question,
            top_k=HYBRID_K
        )
    )

    hybrid_results = (
        hybrid_retriever
        .reciprocal_rank_fusion(
            vector_results=vector_results,
            keyword_results=keyword_results,
            top_k=HYBRID_K
        )
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


def generate_and_judge(
    question,
    documents
):

    if not documents:

        return {
            "answer": (
                "I don't have enough information "
                "in the provided documents."
            ),
            "latency_seconds": 0.0,
            "context_characters": 0,
            "judgement": None
        }

    context = build_context(
        documents
    )

    start = time.perf_counter()

    answer = generate_answer(
        question,
        documents
    )

    latency = (
        time.perf_counter() - start
    )

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

    evidence_evaluator = (
        SemanticEvidenceEvaluator()
    )

    context_builder = ContextBuilder(
        evidence_threshold=0.60
    )

    controller = (
        EmpiricalContextController(
            policy_file=POLICY_FILE,
            fallback_budget=FIXED_BUDGET
        )
    )

    results = []

    for case in evaluation_data:

        question = case["question"]

        required_evidence = case.get(
            "required_evidence",
            []
        )

        print()
        print("=" * 70)
        print(question)

        reranked_results = (
            retrieve_and_rerank(
                question=question,
                collection=collection,
                hybrid_retriever=hybrid_retriever,
                reranker=reranker
            )
        )

        # --------------------------------------------------
        # RETRIEVAL EVIDENCE
        # --------------------------------------------------

        evidence_result = (
            evidence_evaluator.evaluate(
                required_evidence=required_evidence,
                retrieved_documents=reranked_results
            )
        )

        retrieval_evidence_recall = (
            evidence_result["evidence_recall"]
        )

        # --------------------------------------------------
        # FIXED BUDGET
        # --------------------------------------------------

        fixed_documents = (
            context_builder.build(
                strategy="top_n",
                question=question,
                reranked_documents=reranked_results,
                required_evidence=required_evidence,
                max_documents=FIXED_BUDGET
            )
        )

        fixed_generation = (
            generate_and_judge(
                question=question,
                documents=fixed_documents
            )
        )

        # --------------------------------------------------
        # EMPIRICAL POLICY
        # --------------------------------------------------

        decision = controller.choose_budget(
            evidence_recall=(
                retrieval_evidence_recall
            ),
            reranked_documents=(
                reranked_results
            )
        )

        empirical_budget = decision[
            "budget"
        ]

        if empirical_budget > 0:

            empirical_documents = (
                context_builder.build(
                    strategy="evidence_first",
                    question=question,
                    reranked_documents=(
                        reranked_results
                    ),
                    required_evidence=(
                        required_evidence
                    ),
                    max_documents=(
                        empirical_budget
                    )
                )
            )

        else:

            empirical_documents = []

        empirical_generation = (
            generate_and_judge(
                question=question,
                documents=empirical_documents
            )
        )

        # --------------------------------------------------
        # STRATEGY-SPECIFIC EVIDENCE
        # --------------------------------------------------

        fixed_evidence = (
            evidence_evaluator.evaluate(
                required_evidence=(
                    required_evidence
                ),
                retrieved_documents=(
                    fixed_documents
                )
            )
        )

        empirical_evidence = (
            evidence_evaluator.evaluate(
                required_evidence=(
                    required_evidence
                ),
                retrieved_documents=(
                    empirical_documents
                )
            )
        )

        result = {
            "question": question,

            "retrieval_evidence_recall":
                retrieval_evidence_recall,

            "fixed": {
                "budget": len(
                    fixed_documents
                ),
                "evidence_recall":
                    fixed_evidence[
                        "evidence_recall"
                    ],
                **fixed_generation
            },

            "empirical": {
                "budget": len(
                    empirical_documents
                ),
                "decision": decision,
                "evidence_recall":
                    empirical_evidence[
                        "evidence_recall"
                    ],
                **empirical_generation
            }
        }

        results.append(result)

        print(
            "Retrieval evidence recall:",
            retrieval_evidence_recall
        )

        print(
            "Fixed budget:",
            len(fixed_documents)
        )

        print(
            "Empirical budget:",
            len(empirical_documents)
        )

        print(
            "Policy reason:",
            decision["reason"]
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

    output_file = (
        BASE_DIR /
        "empirical_context_benchmark.json"
    )

    with open(
        output_file,
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
    print(
        "Saved:",
        output_file
    )


if __name__ == "__main__":
    main()