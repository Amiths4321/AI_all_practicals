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

CHUNK_SIZE = 500
OVERLAP = 100

VECTOR_K = 10
HYBRID_K = 10
RERANK_K = 10

FALLBACK_BUDGET = 3

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
            f"unseen_questions_"
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


def choose_unseen_budget(
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

    # Exact question match should normally not
    # happen for the held-out set.

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
                "reason": (
                    "unexpected_exact_match"
                )
            }

    # --------------------------------------------------
    # Generalization fallback
    #
    # Use the global empirical policy rather than
    # memorizing the unseen question.
    # --------------------------------------------------

    policy_rows = [
        row
        for row in policy.values()
        if isinstance(row, dict)
        and row.get("budget") is not None
    ]

    if policy_rows:

        # Smallest historically successful budget.
        selected = min(
            policy_rows,
            key=lambda row: row["budget"]
        )

        return {
            "budget": min(
                selected["budget"],
                len(reranked_documents)
            ),
            "reason":
                "global_empirical_fallback"
        }

    return {
        "budget": min(
            FALLBACK_BUDGET,
            len(reranked_documents)
        ),
        "reason":
            "default_fallback"
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

    results = []

    for case in questions:

        question = case["question"]

        required_evidence = case.get(
            "required_evidence",
            []
        )

        answerable = case.get(
            "answerable",
            True
        )

        print()
        print("=" * 70)
        print(question)

        reranked_results = (
            retrieve_and_rerank(
                question=question,
                collection=collection,
                hybrid_retriever=(
                    hybrid_retriever
                ),
                reranker=reranker
            )
        )

        # --------------------------------------------------
        # Retrieval-level evidence
        # --------------------------------------------------

        retrieval_evidence = (
            evidence_evaluator.evaluate(
                required_evidence=(
                    required_evidence
                ),
                retrieved_documents=(
                    reranked_results
                )
            )
        )

        retrieval_recall = (
            retrieval_evidence[
                "evidence_recall"
            ]
        )

        # --------------------------------------------------
        # Adaptive decision
        # --------------------------------------------------

        decision = choose_unseen_budget(
            question=question,
            evidence_recall=retrieval_recall,
            reranked_documents=(
                reranked_results
            ),
            policy=policy
        )

        budget = decision["budget"]

        if budget > 0:

            selected_documents = (
                context_builder.build(
                    strategy="evidence_first",
                    question=question,
                    reranked_documents=(
                        reranked_results
                    ),
                    required_evidence=(
                        required_evidence
                    ),
                    max_documents=budget
                )
            )

        else:

            selected_documents = []

        # --------------------------------------------------
        # Strategy-level evidence
        # --------------------------------------------------

        strategy_evidence = (
            evidence_evaluator.evaluate(
                required_evidence=(
                    required_evidence
                ),
                retrieved_documents=(
                    selected_documents
                )
            )
        )

        # --------------------------------------------------
        # Generate
        # --------------------------------------------------

        generation = generate_and_judge(
            question=question,
            documents=selected_documents
        )

        result = {
            "question": question,
            "answerable": answerable,

            "required_evidence":
                required_evidence,

            "retrieval_evidence_recall":
                retrieval_recall,

            "strategy_evidence_recall":
                strategy_evidence[
                    "evidence_recall"
                ],

            "budget":
                budget,

            "decision":
                decision,

            "context_characters":
                generation[
                    "context_characters"
                ],

            "latency_seconds":
                generation[
                    "latency_seconds"
                ],

            "answer":
                generation[
                    "answer"
                ],

            "judgement":
                generation[
                    "judgement"
                ]
        }

        results.append(result)

        print(
            "Retrieval evidence:",
            retrieval_recall
        )

        print(
            "Adaptive budget:",
            budget
        )

        print(
            "Strategy evidence:",
            strategy_evidence[
                "evidence_recall"
            ]
        )

        print(
            "Decision:",
            decision["reason"]
        )

    output = {
        "questions": len(results),
        "results": results
    }

    with open(
        "unseen_question_results.json",
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
        "Saved unseen_question_results.json"
    )


if __name__ == "__main__":
    main()