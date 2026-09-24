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

QUESTIONS_FILE = (
    BASE_DIR / "unseen_questions.json"
)

POLICY_FILE = (
    BASE_DIR / "empirical_context_policy.json"
)

BUDGETS = [1, 2, 3, 5, 10]

CHUNK_SIZE = 500
OVERLAP = 100

VECTOR_K = 10
HYBRID_K = 10
RERANK_K = 10

EVIDENCE_THRESHOLD = 0.60


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


def load_questions():

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def load_policy():

    with open(
        POLICY_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file).get(
            "policy",
            {}
        )


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
            f"unseen_compare_"
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
        "context_characters":
            len(context),
        "judgement": judgement
    }


def quality_index(judgement):

    if judgement is None:
        return 0.0

    return (
        judgement["groundedness"]
        + judgement["relevance"]
        + judgement["completeness"]
    ) / 6.0


def choose_adaptive_budget(
    question,
    evidence_recall,
    reranked_documents,
    policy
):

    if not reranked_documents:

        return {
            "budget": 0,
            "reason": "no_documents"
        }

    if evidence_recall < 1.0:

        return {
            "budget": 0,
            "reason": "insufficient_evidence"
        }

    # Exact lookup should not normally happen for
    # unseen questions.

    if question in policy:

        budget = policy[
            question
        ].get("budget")

        if budget is not None:

            return {
                "budget": min(
                    budget,
                    len(reranked_documents)
                ),
                "reason": "exact_policy_match"
            }

    valid = [
        item
        for item in policy.values()
        if isinstance(item, dict)
        and item.get("budget") is not None
    ]

    if valid:

        selected = min(
            valid,
            key=lambda item: item["budget"]
        )

        return {
            "budget": min(
                selected["budget"],
                len(reranked_documents)
            ),
            "reason":
                "global_empirical_policy"
        }

    return {
        "budget": min(
            3,
            len(reranked_documents)
        ),
        "reason": "fallback"
    }


def evaluate_budget(
    question,
    required_evidence,
    reranked_documents,
    budget,
    context_builder,
    evidence_evaluator
):

    selected_documents = (
        context_builder.build(
            strategy="evidence_first",
            question=question,
            reranked_documents=(
                reranked_documents
            ),
            required_evidence=(
                required_evidence
            ),
            max_documents=budget
        )
    )

    evidence = evidence_evaluator.evaluate(
        required_evidence=required_evidence,
        retrieved_documents=selected_documents
    )

    generation = generate_and_judge(
        question=question,
        documents=selected_documents
    )

    return {
        "budget": budget,
        "documents_used":
            len(selected_documents),
        "context_characters":
            generation["context_characters"],
        "evidence_recall":
            evidence["evidence_recall"],
        "latency_seconds":
            generation["latency_seconds"],
        "quality_index":
            quality_index(
                generation["judgement"]
            ),
        "judgement":
            generation["judgement"],
        "answer":
            generation["answer"]
    }


def main():

    documents = load_documents()

    questions = load_questions()

    policy = load_policy()

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

    context_builder = ContextBuilder(
        evidence_threshold=EVIDENCE_THRESHOLD
    )

    all_results = []

    for case in questions:

        question = case["question"]

        required_evidence = case.get(
            "required_evidence",
            []
        )

        print()
        print("=" * 70)
        print(question)

        reranked_documents = (
            retrieve_and_rerank(
                question=question,
                collection=collection,
                hybrid_retriever=(
                    hybrid_retriever
                ),
                reranker=reranker
            )
        )

        retrieval_evidence = (
            evidence_evaluator.evaluate(
                required_evidence=(
                    required_evidence
                ),
                retrieved_documents=(
                    reranked_documents
                )
            )
        )

        retrieval_recall = (
            retrieval_evidence[
                "evidence_recall"
            ]
        )

        fixed_results = {}

        for budget in BUDGETS:

            print(
                f"  Fixed budget {budget}..."
            )

            fixed_results[str(budget)] = (
                evaluate_budget(
                    question=question,
                    required_evidence=(
                        required_evidence
                    ),
                    reranked_documents=(
                        reranked_documents
                    ),
                    budget=budget,
                    context_builder=(
                        context_builder
                    ),
                    evidence_evaluator=(
                        evidence_evaluator
                    )
                )
            )

        adaptive_decision = (
            choose_adaptive_budget(
                question=question,
                evidence_recall=(
                    retrieval_recall
                ),
                reranked_documents=(
                    reranked_documents
                ),
                policy=policy
            )
        )

        adaptive_budget = (
            adaptive_decision["budget"]
        )

        if adaptive_budget > 0:

            adaptive_result = (
                evaluate_budget(
                    question=question,
                    required_evidence=(
                        required_evidence
                    ),
                    reranked_documents=(
                        reranked_documents
                    ),
                    budget=adaptive_budget,
                    context_builder=(
                        context_builder
                    ),
                    evidence_evaluator=(
                        evidence_evaluator
                    )
                )
            )

        else:

            adaptive_result = {
                "budget": 0,
                "documents_used": 0,
                "context_characters": 0,
                "evidence_recall": 0.0,
                "latency_seconds": 0.0,
                "quality_index": 0.0,
                "judgement": None,
                "answer": (
                    "I don't have enough information "
                    "in the provided documents."
                )
            }

        all_results.append({
            "question": question,
            "required_evidence":
                required_evidence,
            "retrieval_evidence_recall":
                retrieval_recall,
            "fixed": fixed_results,
            "adaptive": {
                "decision":
                    adaptive_decision,
                "result":
                    adaptive_result
            }
        })

        print(
            "  Retrieval evidence:",
            retrieval_recall
        )

        print(
            "  Adaptive budget:",
            adaptive_budget
        )

    output = {
        "budgets": BUDGETS,
        "questions": all_results
    }

    with open(
        "unseen_budget_comparison.json",
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
        "Saved unseen_budget_comparison.json"
    )


if __name__ == "__main__":
    main()