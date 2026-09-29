import json
import time
from datetime import datetime
import threading
from collections import deque
import uuid
import logging
logger = logging.getLogger("uvicorn.error")

import adaptive_rag
from adaptive_rag import (
    load_documents,
    load_policy,
    create_collection,
    process_question,
    HybridRetriever,
    DocumentReranker,
    SemanticEvidenceEvaluator,
    ContextBuilder,
    EVIDENCE_THRESHOLD,
)

def percentile(values, pct):
    if not values:
        return 0.0
    values = sorted(values)
    return values[int((pct / 100) * (len(values) - 1))]

class ProductionRAGController:

    def __init__(self):
        self.request_count = 0
        self.index_lock = threading.Lock()
        self.metrics_lock = threading.Lock()   # add this line
        self.started_at = datetime.now().isoformat()
        self.index_lock = threading.Lock()
        self.index_available = True
        
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "degraded_requests": 0,
            "total_latency_seconds": 0.0,
            "recent_latencies": deque(maxlen=100),
        }
        print("Initializing production RAG...")

        # Load documents
        documents = load_documents()

        # Create chunks using the same logic/configuration
        chunks = adaptive_rag.build_chunks(documents)

        # Create retrieval components
        self.collection = create_collection(chunks)

        self.hybrid_retriever = HybridRetriever(documents=chunks)
        self.reranker = DocumentReranker()
        self.evidence_evaluator = SemanticEvidenceEvaluator()
        self.context_builder = ContextBuilder(evidence_threshold=EVIDENCE_THRESHOLD)
        self.policy = load_policy()
        try:
            self.collection = adaptive_rag.load_active_collection()
        except Exception:
            self.collection = create_collection(chunks)

        print(f"Loaded documents: {len(documents)}")
        print(f"Loaded chunks: {len(chunks)}")
        print("Production RAG initialized.")

    def _record(self, outcome, latency):
        with self.metrics_lock:
            self.metrics["total_requests"] += 1
            self.request_count += 1
            self.metrics[outcome] += 1
            self.metrics["total_latency_seconds"] += latency
            self.metrics["recent_latencies"].append(latency)
    SLO_TARGETS = {
        "availability": 0.99,
        "error_rate": 0.01,
        "p95_latency_seconds": 10.0,
        "degraded_rate": 0.01,
    }

    def slo_status(self):
        metrics = self.metrics_snapshot()

        total = metrics["total_requests"]

        error_rate = (
            metrics["failed_requests"] / total
            if total
            else 0.0
        )

        degraded_rate = (
            metrics["degraded_requests"] / total
            if total
            else 0.0
        )

        availability = (
            (
                metrics["successful_requests"]
                + metrics["degraded_requests"]
            ) / total
            if total
            else 1.0
        )

        checks = {
            "availability": availability >= self.SLO_TARGETS["availability"],
            "error_rate": error_rate <= self.SLO_TARGETS["error_rate"],
            "p95_latency": (
                metrics["latency_p95_seconds"]
                <= self.SLO_TARGETS["p95_latency_seconds"]
            ),
            "degraded_rate": (
                degraded_rate <= self.SLO_TARGETS["degraded_rate"]
            ),
        }

        return {
            "status": (
                "healthy"
                if all(checks.values())
                else "breached"
            ),
            "targets": self.SLO_TARGETS,
            "actual": {
                "availability": availability,
                "error_rate": error_rate,
                "degraded_rate": degraded_rate,
                "p95_latency_seconds": (
                    metrics["latency_p95_seconds"]
                ),
            },
            "checks": checks,
        }

    def _rebuild_retriever(self):
        """Build the keyword index from the ACTIVE collection, not from disk files."""
        data = self.collection.get(include=["documents", "metadatas"])
        chunks = [
            {"id": i, "document": d, "metadata": m or {}}
            for i, d, m in zip(data["ids"], data["documents"], data["metadatas"])
        ]
        self.hybrid_retriever = HybridRetriever(documents=chunks)
    def health(self):
        import chromadb
        import adaptive_rag
        
        client = chromadb.PersistentClient(path=str(adaptive_rag.INDEX_DIR))
        active = adaptive_rag.load_active_index()
        is_valid = adaptive_rag.validate_index_state(client, active)
        history = adaptive_rag.load_index_history()
        
        return {
            "healthy": is_valid,
            "status": "healthy" if is_valid else "degraded",
            "active_index": active,
            "is_valid": is_valid,
            "history_count": len(history),
            "history": history
        }

    def index_health(self):
        return self.health()

    def metrics_snapshot(self):
        total = self.metrics["total_requests"]
        latencies = list(self.metrics["recent_latencies"])

        average_latency = (
            self.metrics["total_latency_seconds"] / total
            if total
            else 0.0
        )

        return {
            "total_requests": total,
            "successful_requests": self.metrics["successful_requests"],
            "failed_requests": self.metrics["failed_requests"],
            "degraded_requests": self.metrics["degraded_requests"],
            "average_latency_seconds": average_latency,
            "latency_p50_seconds": percentile(latencies, 50),
            "latency_p95_seconds": percentile(latencies, 95),
            "latency_p99_seconds": percentile(latencies, 99),
            "recent_latency_count": len(latencies),
            "max_recent_latency_seconds": (
                max(latencies) if latencies else 0.0
            ),
            "index_available": self.index_available,
        }

    def query(self, question, request_id=None):
        request_started = time.perf_counter()
        request_id = request_id or f"rag-{uuid.uuid4().hex[:12]}"

        if not question or not question.strip():
            self._record("failed_requests", 0.0)
            return {"request_id": request_id, "status": "error",
                    "error": "Question cannot be empty."}

        question = question.strip()

        try:
            with self.index_lock:  # snapshot only; in-flight requests finish on old objects
                collection = self.collection
                retriever = self.hybrid_retriever
                evaluator = self.evidence_evaluator
                builder = self.context_builder
                policy = self.policy

            result = process_question(
                case={"question": question, "required_evidence": [], "answerable": True},
                collection=collection,
                hybrid_retriever=retriever,
                reranker=self.reranker,
                evidence_evaluator=evaluator,
                context_builder=builder,
                policy=policy,
            )

            latency = time.perf_counter() - request_started
            answered = result.get("status") == "answered"
            self._record("successful_requests" if answered else "degraded_requests", latency)

            return {
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
                "status": result.get("status", "unknown"),
                "question": question,
                "answer": result.get("answer"),
                "policy": {"budget": result.get("budget"),
                        "reason": result.get("budget_reason") or result.get("reason")},
                "evidence": {"retrieval_recall": result.get("retrieval_evidence_recall"),
                            "context_recall": result.get("context_evidence_recall")},
                "quality": result.get("judgement") or {},
                "performance": {"controller_latency_seconds": latency,
                                "generation_latency_seconds": result.get("latency_seconds")},
                "cache": {"generation_hit": result.get("generation_cache_hit", False),
                        "judge_hit": result.get("judge_cache_hit", False)},
                "sources": [d.get("metadata", {}).get("source")
                            for d in result.get("selected_documents", [])],
            }

        except Exception as exc:
            latency = time.perf_counter() - request_started
            self._record("failed_requests", latency)
            return {"request_id": request_id, "timestamp": datetime.now().isoformat(),
                    "status": "error", "question": question,
                    "performance": {"controller_latency_seconds": latency},
                    "error": {"type": type(exc).__name__, "message": str(exc)}}

    def recover_index(self):
        with self.index_lock:
            recovered = adaptive_rag.recover_active_index()
            self.collection = adaptive_rag.load_active_collection()

            self._rebuild_retriever()
            self.evidence_evaluator = SemanticEvidenceEvaluator()
            self.context_builder = ContextBuilder(evidence_threshold=EVIDENCE_THRESHOLD)
            self.policy = load_policy()

            self.hybrid_retriever = HybridRetriever(documents=chunks)
            self.evidence_evaluator = SemanticEvidenceEvaluator()
            self.context_builder = ContextBuilder(evidence_threshold=EVIDENCE_THRESHOLD)
            self.policy = load_policy()

            return {
                "recovered": True,
                "collection": self.collection.name,
                "fingerprint": recovered["fingerprint"],
                "chunk_count": self.collection.count(),
            }

    def reload_index(self):
        with self.index_lock:
            self._rebuild_retriever()
            self.evidence_evaluator = SemanticEvidenceEvaluator()
            self.context_builder = ContextBuilder(evidence_threshold=EVIDENCE_THRESHOLD)
            self.policy = load_policy()

            self.collection = create_collection(chunks)
            self.hybrid_retriever = HybridRetriever(documents=chunks)
            self.evidence_evaluator = SemanticEvidenceEvaluator()
            self.context_builder = ContextBuilder(evidence_threshold=EVIDENCE_THRESHOLD)
            self.policy = load_policy()

            return {
                "collection": self.collection.name,
                "chunk_count": len(chunks),
            }

    def rollback_and_reload(self):
        with self.index_lock:
            previous = adaptive_rag.rollback_index()
            self.collection = adaptive_rag.load_active_collection()

            documents = load_documents()
            chunks = adaptive_rag.build_chunks(documents)

            self.hybrid_retriever = HybridRetriever(documents=chunks)
            self.evidence_evaluator = SemanticEvidenceEvaluator()
            self.context_builder = ContextBuilder(evidence_threshold=EVIDENCE_THRESHOLD)
            self.policy = load_policy()

            return {
                "rolled_back_to": previous["collection"],
                "collection": self.collection.name,
                "chunk_count": self.collection.count(),
            }

    def percentile(values, percentile):
        if not values:
            return 0.0

        values = sorted(values)

        index = int(
            (percentile / 100) * (len(values) - 1)
        )

        return values[index]
    
def main():
    controller = ProductionRAGController()
    print()
    print("Production RAG Controller")
    print("Type 'exit' to stop.")

    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() == "exit":
            break

        result = controller.query(question)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()