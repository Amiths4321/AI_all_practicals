import json
import time
from pathlib import Path
import hashlib
import chromadb
from config import load_config
from hybrid_search import HybridRetriever
from reranking import DocumentReranker
# Add this import near the top of adaptive_rag.py
from chunking import chunk_text
from semantic_evidence import SemanticEvidenceEvaluator
from cached_generation import cached_generate_answer
from cached_judge import cached_judge_answer
from context_builder import ContextBuilder
BASE_DIR = Path(__file__).resolve().parent

# --- Load configuration values ---
config = load_config() # Or define them directly below if config.py works differently

INDEX_VERSION = config.get("INDEX_VERSION", "v1")
EVIDENCE_THRESHOLD = config.get("EVIDENCE_THRESHOLD", 0.7)
CHUNK_SIZE = config.get("CHUNK_SIZE", 500)
OVERLAP = config.get("OVERLAP", 50)
VECTOR_K = config.get("VECTOR_K", 5)
HYBRID_K = config.get("HYBRID_K", 5)
RERANK_K = config.get("RERANK_K", 3)
DEFAULT_BUDGET = config.get("DEFAULT_BUDGET", 3)

# File Paths
INDEX_DIR = BASE_DIR / "chroma_db"
INDEX_STATE_FILE = INDEX_DIR / "active_index.json"
INDEX_HISTORY_FILE = INDEX_DIR / "index_history.json"
INDEX_MANIFEST = INDEX_DIR / "manifest.json"
EVALUATION_FILE = BASE_DIR / "evaluation.json"
POLICY_FILE = BASE_DIR / "policy.json"
OUTPUT_FILE = BASE_DIR / "output_report.json"

DOCUMENT_FILES = [
    BASE_DIR / "documents" / "employee_handbook.txt",
    BASE_DIR / "documents" / "leave_policy.txt",
]


        
# Add this near the top of adaptive_rag.py
def load_index_history():
    if not INDEX_HISTORY_FILE.exists():
        return []

    with open(INDEX_HISTORY_FILE, "r", encoding="utf-8") as file:
        return json.load(file)




def save_index_history(history):
    INDEX_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    temporary_file = INDEX_HISTORY_FILE.with_suffix(".tmp")

    with open(temporary_file, "w", encoding="utf-8") as file:
        json.dump(history[:5], file, indent=2)

    temporary_file.replace(INDEX_HISTORY_FILE)

def index_name(fingerprint):
    return (
        f"adaptive_rag_"
        f"{fingerprint[:16]}"
    )


def load_active_index():

    if not INDEX_STATE_FILE.exists():
        return None

    with open(
        INDEX_STATE_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def save_active_index(collection_name, fingerprint, chunk_count):
    INDEX_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

    current = load_active_index()
    history = load_index_history()

    if current and current.get("collection"):
        history.insert(0, current)

    history = history[:5]
    save_index_history(history)

    state = {
        "collection": collection_name,
        "fingerprint": fingerprint,
        "chunk_count": chunk_count,
        "updated_at": time.time()
    }

    temporary_file = INDEX_STATE_FILE.with_suffix(".tmp")

    with open(temporary_file, "w", encoding="utf-8") as file:
        json.dump(state, file, indent=2)

    temporary_file.replace(INDEX_STATE_FILE)

def validate_index_state(client, state):
    if not state:
        return False

    collection_name = state.get("collection")
    expected_count = state.get("chunk_count")

    if not collection_name:
        return False

    try:
        collection = client.get_collection(collection_name)
    except Exception:
        return False

    actual_count = collection.count()

    return actual_count == expected_count

def rollback_index():
    client = chromadb.PersistentClient(path=str(INDEX_DIR))

    current = load_active_index()
    history = load_index_history()

    if not history:
        raise RuntimeError("No previous index available for rollback")

    previous = history[0]

    if not validate_index_state(client, previous):
        raise RuntimeError(
            f"Previous index is invalid: {previous.get('collection')}"
        )

    temporary_file = INDEX_STATE_FILE.with_suffix(".tmp")

    with open(temporary_file, "w", encoding="utf-8") as file:
        json.dump(previous, file, indent=2)

    temporary_file.replace(INDEX_STATE_FILE)

    # Move the current index into history.
    if current:
        history = [current] + history[1:]

    save_index_history(history)

    return previous

def load_documents():
    documents = []

    for filename in DOCUMENT_FILES:
        filename = Path(filename)

        with open(filename, "r", encoding="utf-8") as file:
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
    with open(EVALUATION_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def load_policy():
    if not POLICY_FILE.exists():
        return {}

    with open(POLICY_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data.get("policy", {})

def load_index_manifest():

    if not INDEX_MANIFEST.exists():
        return None

    with open(
        INDEX_MANIFEST,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def save_index_manifest(fingerprint, chunk_count):

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    manifest = {
        "fingerprint": fingerprint,
        "chunk_count": chunk_count,
        "chunk_size": CHUNK_SIZE,
        "overlap": OVERLAP
    }

    with open(
        INDEX_MANIFEST,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2
        )

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
                    "source": str(document["metadata"]["source"]),
                    "chunk_index": index
                }
            })

    return chunks


def create_collection(chunks):

    import chromadb

    fingerprint = calculate_index_fingerprint(
        chunks
    )

    client = chromadb.PersistentClient(
        path=str(INDEX_DIR)
    )

    collection_name = index_name(
        fingerprint
    )

    active = load_active_index()

    # --------------------------------------------------
    # EXISTING CORRECT INDEX
    # --------------------------------------------------

    if (
        active is not None
        and active.get("fingerprint") == fingerprint
    ):

        try:

            collection = client.get_collection(
                name=active["collection"]
            )

            if collection.count() == len(chunks):

                print(
                    "Using active Chroma index"
                )

                print(
                    f"Collection: "
                    f"{active['collection']}"
                )

                return collection

        except Exception:

            print(
                "Active index unavailable."
            )

    # --------------------------------------------------
    # BUILD NEW INDEX BESIDE LIVE INDEX
    # --------------------------------------------------

    print(
        "Building new Chroma index..."
    )

    print(
        f"New fingerprint: "
        f"{fingerprint[:16]}"
    )

    new_collection = (
        client.get_or_create_collection(
            name=collection_name
        )
    )

    # Protect against a partially-created collection.

    if new_collection.count() != len(chunks):

        if new_collection.count() > 0:

            client.delete_collection(
                collection_name
            )

            new_collection = (
                client.create_collection(
                    name=collection_name
                )
            )

        new_collection.add(
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

    # --------------------------------------------------
    # VERIFY BEFORE ACTIVATION
    # --------------------------------------------------

    if new_collection.count() != len(chunks):

        raise RuntimeError(
            "New Chroma index failed verification"
        )

    # --------------------------------------------------
    # ATOMIC ACTIVE-INDEX SWITCH
    # --------------------------------------------------

    save_active_index(
        collection_name=collection_name,
        fingerprint=fingerprint,
        chunk_count=len(chunks)
    )

    print(
        "New Chroma index activated."
    )

    print(
        f"Collection: {collection_name}"
    )

    return new_collection


def vector_search(collection, question, top_k):
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


def retrieve(question, collection, hybrid_retriever):
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

    return {
        "vector_results": vector_results,
        "keyword_results": keyword_results,
        "hybrid_results": hybrid_results
    }


def rerank(question, hybrid_results, reranker):
    return reranker.rerank(
        question=question,
        documents=hybrid_results,
        top_n=RERANK_K
    )


def evaluate_evidence(required_evidence, documents, evidence_evaluator):
    return evidence_evaluator.evaluate(
        required_evidence=required_evidence,
        retrieved_documents=documents
    )


def choose_budget(
    question,
    evidence_recall,
    reranked_documents,
    policy,
    default_budget=3
):
    """
    Choose the context budget.

    Decision order:
    1. No retrieved documents -> abstain.
    2. Insufficient retrieval evidence -> abstain.
    3. Exact question-specific policy.
    4. Global empirical policy.
    5. Default budget.
    """

    # 1. No retrieved evidence
    if not reranked_documents:
        return {
            "budget": 0,
            "reason": "no_retrieved_documents"
        }

    # 2. Evidence gate
    if evidence_recall < 1.0:
        return {
            "budget": 0,
            "reason": "insufficient_evidence"
        }

    # 3. Question-specific policy
    if isinstance(policy, list):
        for item in policy:
            if not isinstance(item, dict):
                continue

            if item.get("question") == question:
                budget = item.get("recommended_budget")

                if budget is None:
                    budget = item.get("budget")

                if budget is not None:
                    return {
                        "budget": int(budget),
                        "reason": "question_specific_policy"
                    }

    elif isinstance(policy, dict):
        question_policy = policy.get(question)

        if isinstance(question_policy, dict):
            budget = question_policy.get("recommended_budget")

            if budget is None:
                budget = question_policy.get("budget")

            if budget is not None:
                return {
                    "budget": int(budget),
                    "reason": "question_specific_policy"
                }

    # 4. Global empirical policy
    if isinstance(policy, dict):
        budget = policy.get("recommended_budget")

        if budget is None:
            budget = policy.get("budget")

        if budget is not None:
            return {
                "budget": int(budget),
                "reason": "global_empirical_policy"
            }

    # 5. Default
    return {
        "budget": default_budget,
        "reason": "default_budget"
    }

def calculate_index_fingerprint(chunks):
    hasher = hashlib.sha256()

    for item in sorted(
        chunks,
        key=lambda x: x["id"]
    ):
        hasher.update(
            item["id"].encode("utf-8")
        )

        hasher.update(
            item["document"].encode("utf-8")
        )

        source = (
            item.get("metadata", {})
            .get("source", "")
        )

        hasher.update(
            str(source).encode("utf-8")
        )

    return hasher.hexdigest()

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

    for index, item in enumerate(documents, start=1):
        metadata = item.get("metadata", {}) or {}
        source = metadata.get("source", "unknown")

        parts.append(
            f"[SOURCE {index}: {source}]\n"
            f"{item['document']}"
        )

    return "\n\n".join(parts)


def generate(question, documents):
    if not documents:
        return {
            "answer": "I don't have enough information in the provided documents.",
            "latency_seconds": 0.0,
            "context_characters": 0,
            "cache_hit": False,
        }

    context = build_context(documents)

    start = time.perf_counter()
    result = cached_generate_answer(question=question, documents=documents)
    latency = time.perf_counter() - start

    return {
        "answer": result["answer"],
        "latency_seconds": latency,
        "context_characters": len(context),
        "cache_hit": result.get("cache_hit", False),
    }


def process_question(case, collection, hybrid_retriever, reranker,
                     evidence_evaluator, context_builder, policy):
    timings = {}
    total_started = time.perf_counter()

    question = case["question"]
    required_evidence = case.get("required_evidence", [])
    answerable = case.get("answerable", True)

    t = time.perf_counter()
    retrieval = retrieve(question, collection, hybrid_retriever)
    timings["retrieval_seconds"] = time.perf_counter() - t

    t = time.perf_counter()
    reranked_documents = rerank(question, retrieval["hybrid_results"], reranker)
    timings["reranking_seconds"] = time.perf_counter() - t

    t = time.perf_counter()
    evidence_result = evaluate_evidence(required_evidence, reranked_documents, evidence_evaluator)
    retrieval_evidence_recall = evidence_result["evidence_recall"]
    timings["evidence_seconds"] = time.perf_counter() - t

    decision = choose_budget(
        question=question,
        evidence_recall=retrieval_evidence_recall,
        reranked_documents=reranked_documents,
        policy=policy,
        default_budget=DEFAULT_BUDGET,
    )
    budget = decision["budget"]

    t = time.perf_counter()
    selected_documents = build_adaptive_context(
        question, required_evidence, reranked_documents, budget, context_builder
    )
    timings["context_seconds"] = time.perf_counter() - t

    context_evidence = evaluate_evidence(required_evidence, selected_documents, evidence_evaluator)
    context_evidence_recall = context_evidence["evidence_recall"]

    if not selected_documents or context_evidence_recall < 1.0:
        timings["total_seconds"] = time.perf_counter() - total_started
        return {
            "question": question, "answerable": answerable,
            "status": "abstained",
            "reason": decision["reason"] if budget == 0 else "insufficient_context_evidence",
            "budget": budget, "decision": decision,
            "retrieval_evidence_recall": retrieval_evidence_recall,
            "context_evidence_recall": context_evidence_recall,
            "answer": "I don't have enough information in the provided documents.",
            "judgement": None, "timings": timings,
            "retrieval": retrieval,
            "reranked_documents": reranked_documents,
            "selected_documents": selected_documents,
        }

    generation = generate(question, selected_documents)
    timings["generation_seconds"] = generation["latency_seconds"]

    judgement_result = cached_judge_answer(
        question=question,
        context=build_context(selected_documents),
        answer=generation["answer"],
    )
    judgement = {k: v for k, v in judgement_result.items() if k != "cache_hit"}

    timings["total_seconds"] = time.perf_counter() - total_started

    return {
        "question": question, "answerable": answerable, "status": "answered",
        "budget": budget, "budget_reason": decision["reason"],
        "retrieval_evidence_recall": retrieval_evidence_recall,
        "context_evidence_recall": context_evidence_recall,
        "answer": generation["answer"],
        "latency_seconds": generation["latency_seconds"],
        "context_characters": generation["context_characters"],
        "judgement": judgement,
        "generation_cache_hit": generation["cache_hit"],
        "judge_cache_hit": judgement_result["cache_hit"],
        "timings": timings,
        "retrieval": retrieval,
        "reranked_documents": reranked_documents,
        "selected_documents": selected_documents,
    }
def load_active_collection():
    state = load_active_index()

    if not state:
        raise RuntimeError("No active index configured")

    client = chromadb.PersistentClient(
        path=str(INDEX_DIR)
    )

    collection = client.get_collection(
        state["collection"]
    )

    if collection.count() != state["chunk_count"]:
        raise RuntimeError(
            "Active index failed validation"
        )

    return collection

    logger.info(
        "RAG request completed request_id=%s",
        request_id,
    )

    logger.exception(
        "RAG request failed request_id=%s",
        request_id,
    )

def recover_active_index():
    client = chromadb.PersistentClient(
        path=str(INDEX_DIR)
    )

    history = load_index_history()

    for candidate in history:
        if validate_index_state(client, candidate):
            temporary_file = INDEX_STATE_FILE.with_suffix(".tmp")

            with open(temporary_file, "w", encoding="utf-8") as file:
                json.dump(candidate, file, indent=2)

            temporary_file.replace(INDEX_STATE_FILE)

            return candidate

    raise RuntimeError("No valid recovery index available")
def main():
    print("=" * 70)
    print("ADAPTIVE RAG")
    print("=" * 70)

    # LOAD DATA
    documents = load_documents()
    evaluation_data = load_evaluation_data()
    policy = load_policy()

    print(f"Documents: {len(documents)}")
    print(f"Questions: {len(evaluation_data)}")

    if isinstance(policy, (list, dict)):
        print(f"Policy entries: {len(policy)}")
    else:
        print("Policy entries: 0")

    # BUILD CHUNKS
    chunks = build_chunks(documents)
    print(f"Chunks: {len(chunks)}")

    # CREATE RETRIEVAL COMPONENTS
    collection = create_collection(chunks)
    hybrid_retriever = HybridRetriever(documents=chunks)
    reranker = DocumentReranker()
    evidence_evaluator = SemanticEvidenceEvaluator()
    context_builder = ContextBuilder(evidence_threshold=EVIDENCE_THRESHOLD)

    # PROCESS QUESTIONS
    results = []

    for index, case in enumerate(evaluation_data, start=1):
        print()
        print(f"[{index}/{len(evaluation_data)}]")
        print(case["question"])

        try:
            result = process_question(
                case=case,
                collection=collection,
                hybrid_retriever=hybrid_retriever,
                reranker=reranker,
                evidence_evaluator=evidence_evaluator,
                context_builder=context_builder,
                policy=policy
            )
        except Exception as error:
            print(f"ERROR processing question: {error}")

            result = {
                "question": case.get("question", ""),
                "answerable": case.get("answerable", True),
                "status": "error",
                "reason": "question_processing_error",
                "error": str(error),
                "budget": 0,
                "retrieval_evidence_recall": 0.0,
                "context_evidence_recall": 0.0,
                "answer": "The question could not be processed.",
                "judgement": None
            }

        results.append(result)

        print("Status:", result["status"])
        print("Budget:", result["budget"])
        print("Retrieval evidence:", result["retrieval_evidence_recall"])
        print("Context evidence:", result["context_evidence_recall"])

        if result["status"] == "answered":
            print("Latency:", round(result["latency_seconds"], 3), "seconds")

    # SUMMARY
    answered = sum(result["status"] == "answered" for result in results)
    abstained = sum(result["status"] == "abstained" for result in results)
    errors = sum(result["status"] == "error" for result in results)

    report = {
        "config": {
            "chunk_size": CHUNK_SIZE,
            "overlap": OVERLAP,
            "vector_k": VECTOR_K,
            "hybrid_k": HYBRID_K,
            "rerank_k": RERANK_K,
            "default_budget": DEFAULT_BUDGET,
            "evidence_threshold": EVIDENCE_THRESHOLD
        },
        "summary": {
            "questions": len(results),
            "answered": answered,
            "abstained": abstained,
            "answer_rate": (
                answered / len(results)
                if results
                else 0.0
            ),
            "abstention_rate": (
                abstained / len(results)
                if results
                else 0.0
            ),
            "error_rate": (
                errors / len(results)
                if results
                else 0.0
            )
        },
        "results": results
    }

    # SAVE REPORT
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print("Questions:", len(results))
    print("Answered:", answered)
    print("Abstained:", abstained)
    print("Errors:", errors)
    print("Answer rate:", round(report["summary"]["answer_rate"], 3))
    print("Abstention rate:", round(report["summary"]["abstention_rate"], 3))
    print("Error rate:", round(report["summary"]["error_rate"], 3))

    print()
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()