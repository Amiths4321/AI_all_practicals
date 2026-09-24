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

POLICY_FILE = (
    BASE_DIR / "empirical_context_policy.json"
)

CHUNK_SIZE = 500
OVERLAP = 100

VECTOR_K = 10
HYBRID_K = 10
RERANK_K = 10

DEFAULT_BUDGET = 3
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


def load_evaluation_data():

    with open(
        EVALUATION_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def load_policy():

    if not POLICY_FILE.exists():
        return {}

    with open(
        POLICY_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    return data.get(
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
            f"adaptive_rag_"
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

    return {
        "vector_results":
            vector_results,
        "keyword_results":
            keyword_results,
        "hybrid_results":
            hybrid_results
    }


def rerank(
    question,
    hybrid_results,
    reranker
):

    return reranker.rerank(
        question=question,
        documents=hybrid_results,
        top_n=RERANK_K
    )


def evaluate_evidence(
    required_evidence,
    documents,
    evidence_evaluator
):

    return evidence_evaluator.evaluate(
        required_evidence=required_evidence,
        retrieved_documents=documents
    )


def choose_budget(
    question,
    evidence_result,
    reranked_documents,
    policy,
    default_budget=3
):
    # No evidence means we should not generate an answer.
    if not reranked_documents:
        return {
            "budget": 0,
            "reason": "no_retrieved_documents"
        }

    if evidence_result["evidence_recall"] < 1.0:
        return {
            "budget": 0,
            "reason": "insufficient_evidence"
        }

    # ---------------------------------------------------------
    # 1. Try an exact question-specific policy
    # ---------------------------------------------------------
    if isinstance(policy, list):
        for item in policy:
            if not isinstance(item, dict):
                continue

            if item.get("question") == question:
                budget = item.get("budget")

                if budget is not None:
                    return {
                        "budget": int(budget),
                        "reason": "question_specific_policy"
                    }

    elif isinstance(policy, dict):
        # Handle dictionary-shaped policies too.
        question_policy = policy.get(question)

        if isinstance(question_policy, dict):
            budget = question_policy.get("budget")

            if budget is not None:
                return {
                    "budget": int(budget),
                    "reason": "question_specific_policy"
                }

    # ---------------------------------------------------------
    # 2. Fall back to global empirical policy
    # ---------------------------------------------------------
    if isinstance(policy, list):
        budgets = []

        for item in policy:
            if not isinstance(item, dict):
                continue

            budget = item.get("budget")

            if budget is not None:
                budgets.append(int(budget))

        if budgets:
            return {
                "budget": min(budgets),
                "reason": "global_empirical_policy"
            }

    elif isinstance(policy, dict):
        # Possible dictionary format:
        # {"budget": 3}
        if policy.get("budget") is not None:
            return {
                "budget": int(policy["budget"]),
                "reason": "global_empirical_policy"
            }

        # Possible format:
        # {"recommended_budget": 3}
        if policy.get("recommended_budget") is not None:
            return {
                "budget": int(policy["recommended_budget"]),
                "reason": "global_empirical_policy"
            }

    # ---------------------------------------------------------
    # 3. Final fallback
    # ---------------------------------------------------------
    return {
        "budget": default_budget,
        "reason": "default_budget"
    }


def build_adaptive_context(
    question,
    required_evidence,
    reranked_documents,
    budget,
    context_builder
):

    if budget <= 0:
        return []

    return context_builder.build(
        strategy="evidence_first",
        question=question,
        reranked_documents=reranked_documents,
        required_evidence=required_evidence,
        max_documents=budget
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


def generate(
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
            "context_characters": 0
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

    return {
        "answer": answer,
        "latency_seconds": latency,
        "context_characters":
            len(context)
    }


def process_question(
    case,
    collection,
    hybrid_retriever,
    reranker,
    evidence_evaluator,
    context_builder,
    policy
):

    question = case["question"]

    required_evidence = case.get(
        "required_evidence",
        []
    )

    answerable = case.get(
        "answerable",
        True
    )

    # --------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------

    retrieval = retrieve(
        question=question,
        collection=collection,
        hybrid_retriever=hybrid_retriever
    )

    # --------------------------------------------------
    # RERANK
    # --------------------------------------------------

    reranked_documents = rerank(
        question=question,
        hybrid_results=retrieval[
            "hybrid_results"
        ],
        reranker=reranker
    )

    # --------------------------------------------------
    # RETRIEVAL-LEVEL EVIDENCE
    # --------------------------------------------------

    evidence_result = evaluate_evidence(
        required_evidence=required_evidence,
        documents=reranked_documents,
        evidence_evaluator=evidence_evaluator
    )

    retrieval_evidence_recall = (
        evidence_result[
            "evidence_recall"
        ]
    )

    # --------------------------------------------------
    # ADAPTIVE POLICY
    # --------------------------------------------------

    decision = choose_budget(
        question=question,
        reranked_documents=(
            reranked_documents
        ),
        evidence_recall=(
            retrieval_evidence_recall
        ),
        policy=policy
    )

    budget = decision["budget"]

    # --------------------------------------------------
    # CONTEXT
    # --------------------------------------------------

    selected_documents = (
        build_adaptive_context(
            question=question,
            required_evidence=(
                required_evidence
            ),
            reranked_documents=(
                reranked_documents
            ),
            budget=budget,
            context_builder=context_builder
        )
    )

    # --------------------------------------------------
    # STRATEGY-LEVEL EVIDENCE
    # --------------------------------------------------

    context_evidence = (
        evaluate_evidence(
            required_evidence=(
                required_evidence
            ),
            documents=selected_documents,
            evidence_evaluator=evidence_evaluator
        )
    )

    context_evidence_recall = (
        context_evidence[
            "evidence_recall"
        ]
    )

    # --------------------------------------------------
    # FINAL EVIDENCE GATE
    # --------------------------------------------------

    if (
        not selected_documents
        or context_evidence_recall < 1.0
    ):

        return {
            "question": question,
            "answerable": answerable,

            "status": "abstained",

            "reason": (
                "insufficient_context_evidence"
            ),

            "budget": budget,

            "decision": decision,

            "retrieval_evidence_recall":
                retrieval_evidence_recall,

            "context_evidence_recall":
                context_evidence_recall,

            "answer": (
                "I don't have enough information "
                "in the provided documents."
            ),

            "judgement": None,

            "retrieval": retrieval,

            "reranked_documents":
                reranked_documents,

            "selected_documents":
                selected_documents
        }

    # --------------------------------------------------
    # GENERATION
    # --------------------------------------------------

    generation = generate(
        question=question,
        documents=selected_documents
    )

    # --------------------------------------------------
    # JUDGE
    # --------------------------------------------------

    context = build_context(
        selected_documents
    )

    judgement = judge_answer(
        question=question,
        context=context,
        answer=generation["answer"]
    )

    return {
        "question": question,
        "answerable": answerable,

        "status": "answered",

        "reason": (
            "sufficient_evidence"
        ),

        "budget": budget,

        "decision": decision,

        "retrieval_evidence_recall":
            retrieval_evidence_recall,

        "context_evidence_recall":
            context_evidence_recall,

        "context_characters":
            generation[
                "context_characters"
            ],

        "latency_seconds":
            generation[
                "latency_seconds"
            ],

        "answer":
            generation["answer"],

        "judgement":
            judgement,

        "retrieval": retrieval,

        "reranked_documents":
            reranked_documents,

        "selected_documents":
            selected_documents
    }


def main():

    print("=" * 70)
    print("ADAPTIVE RAG")
    print("=" * 70)

    documents = load_documents()

    evaluation_data = (
        load_evaluation_data()
    )

    policy = load_policy()

    print(
        f"Documents: {len(documents)}"
    )

    print(
        f"Questions: {len(evaluation_data)}"
    )

    print(
        f"Policy entries: {len(policy)}"
    )

    chunks = build_chunks(
        documents
    )

    print(
        f"Chunks: {len(chunks)}"
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
            f"[{index}/{len(evaluation_data)}]"
        )

        print(
            case["question"]
        )

        result = process_question(
            case=case,
            collection=collection,
            hybrid_retriever=(
                hybrid_retriever
            ),
            reranker=reranker,
            evidence_evaluator=(
                evidence_evaluator
            ),
            context_builder=(
                context_builder
            ),
            policy=policy
        )

        results.append(result)

        print(
            "Status:",
            result["status"]
        )

        print(
            "Budget:",
            result["budget"]
        )

        print(
            "Retrieval evidence:",
            result[
                "retrieval_evidence_recall"
            ]
        )

        print(
            "Context evidence:",
            result[
                "context_evidence_recall"
            ]
        )

        if result["status"] == "answered":

            print(
                "Latency:",
                round(
                    result["latency_seconds"],
                    3
                ),
                "seconds"
            )

    answered = sum(
        result["status"] == "answered"
        for result in results
    )

    abstained = sum(
        result["status"] == "abstained"
        for result in results
    )

    report = {
        "config": {
            "chunk_size": CHUNK_SIZE,
            "overlap": OVERLAP,
            "vector_k": VECTOR_K,
            "hybrid_k": HYBRID_K,
            "rerank_k": RERANK_K,
            "default_budget":
                DEFAULT_BUDGET,
            "evidence_threshold":
                EVIDENCE_THRESHOLD
        },

        "summary": {
            "questions":
                len(results),
            "answered":
                answered,
            "abstained":
                abstained,
            "answer_rate": (
                answered / len(results)
                if results
                else 0.0
            ),
            "abstention_rate": (
                abstained / len(results)
                if results
                else 0.0
            )
        },

        "results": results
    }

    output_file = (
        BASE_DIR /
        "adaptive_rag_report.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        "Questions:",
        len(results)
    )

    print(
        "Answered:",
        answered
    )

    print(
        "Abstained:",
        abstained
    )

    print(
        "Answer rate:",
        round(
            report["summary"]["answer_rate"],
            3
        )
    )

    print(
        "Abstention rate:",
        round(
            report["summary"]["abstention_rate"],
            3
        )
    )

    print()
    print(
        "Saved:",
        output_file
    )


if __name__ == "__main__":
    main()