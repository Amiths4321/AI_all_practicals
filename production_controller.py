import json
import time
from datetime import datetime
import threading
from collections import deque
import uuid
import logging
logger = logging.getLogger("uvicorn.error")
from pathlib import Path
from config import (
    load_config,
    calculate_config_fingerprint,
)

CONFIG = load_config()
OPERATIONAL_CONFIG = CONFIG.get("operational", {})

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

BASE_DIR = Path(__file__).resolve().parent
METRICS_FILE = BASE_DIR / "chroma_db" / "service_metrics.json"
MAX_ALERTS = 100
MAX_METRICS_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
CONFIG = load_config()
SERVICE_VERSION = "1.0.0"
DEPLOYMENT_MANIFEST = (
    BASE_DIR / "deployment_manifest.json"
)

OPERATIONAL_CONFIG = CONFIG.get(
    "operational",
    {}
)

SLO_TARGETS = OPERATIONAL_CONFIG.get(
    "slo",
    {
        "availability": 0.99,
        "error_rate": 0.01,
        "p95_latency_seconds": 10.0,
        "degraded_rate": 0.01,
    }
)



def rotate_metrics_file_if_needed():
    if not METRICS_FILE.exists():
        return

    if METRICS_FILE.stat().st_size < MAX_METRICS_FILE_SIZE:
        return

    rotated_file = METRICS_FILE.with_suffix(".old.json")

    if rotated_file.exists():
        rotated_file.unlink()

    METRICS_FILE.replace(rotated_file)

def load_persistent_metrics():
    if not METRICS_FILE.exists():
        return {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "degraded_requests": 0,
            "total_latency_seconds": 0.0,
            "alerts": [],
        }

    with open(METRICS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def save_persistent_metrics(metrics):
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)

    rotate_metrics_file_if_needed()

    metrics["alerts"] = metrics.get(
        "alerts",
        []
    )[-MAX_ALERTS:]

    temporary_file = METRICS_FILE.with_suffix(".tmp")

    with open(temporary_file, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    temporary_file.replace(METRICS_FILE)

def save_deployment_manifest(metadata):
    DEPLOYMENT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = DEPLOYMENT_MANIFEST.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    tmp.replace(DEPLOYMENT_MANIFEST)

def percentile(values, pct):
    if not values:
        return 0.0
    values = sorted(values)
    return values[int((pct / 100) * (len(values) - 1))]

class ProductionRAGController:
    def validate_deployment_manifest(self):
        if not DEPLOYMENT_MANIFEST.exists():
            return {
                "valid": False,
                "reason": "Deployment manifest does not exist",
            }

        with open(
            DEPLOYMENT_MANIFEST,
            "r",
            encoding="utf-8"
        ) as file:
            manifest = json.load(file)

        checks = {
            "service_version": (
                manifest.get("service_version")
                == self.deployment_metadata["service_version"]
            ),
            "config_fingerprint": (
                manifest.get("config_fingerprint")
                == self.deployment_metadata["config_fingerprint"]
            ),
            "index_collection": (
                manifest.get("index_collection")
                == self.deployment_metadata["index_collection"]
            ),
            "index_fingerprint": (
                manifest.get("index_fingerprint")
                == self.deployment_metadata["index_fingerprint"]
            ),
            "index_chunk_count": (
                manifest.get("index_chunk_count")
                == self.deployment_metadata["index_chunk_count"]
            ),
        }

        return {
            "valid": all(checks.values()),
            "checks": checks,
        }
    def __init__(self):
        self.request_count = 0
        self.index_lock = threading.Lock()
        self.metrics_lock = threading.Lock()
        self.started_at = datetime.now().isoformat()
        self.index_available = True

        print("Initializing production RAG...")

        documents = load_documents()
        chunks = adaptive_rag.build_chunks(documents)

        # Prefer the active index so a rollback survives restarts.
        try:
            self.collection = adaptive_rag.load_active_collection()
        except Exception:
            self.collection = create_collection(chunks)

        self._rebuild_retriever()   # keyword index from the same collection
        self.reranker = DocumentReranker()
        self.evidence_evaluator = SemanticEvidenceEvaluator()
        self.context_builder = ContextBuilder(evidence_threshold=EVIDENCE_THRESHOLD)
        self.policy = load_policy()

        persistent = load_persistent_metrics()
        self.metrics = {
            "total_requests": persistent.get("total_requests", 0),
            "successful_requests": persistent.get("successful_requests", 0),
            "failed_requests": persistent.get("failed_requests", 0),
            "degraded_requests": persistent.get("degraded_requests", 0),
            "total_latency_seconds": persistent.get("total_latency_seconds", 0.0),
            "recent_latencies": deque(maxlen=100),
        }
        self.alerts = persistent.get("alerts", [])

        self.deployment_metadata = {
            "service_version": SERVICE_VERSION,
            "started_at": self.started_at,
            "config_fingerprint": calculate_config_fingerprint(CONFIG),
            "index_collection": self.collection.name,
            "index_fingerprint": adaptive_rag.load_active_index()["fingerprint"],
            "index_chunk_count": self.collection.count(),
        }
        save_deployment_manifest(self.deployment_metadata)

        print(f"Loaded documents: {len(documents)}")
        print(f"Loaded chunks: {len(chunks)}")
        print("Production RAG initialized.")

    def _refresh_components(self):
        self._rebuild_retriever()
        self.evidence_evaluator = SemanticEvidenceEvaluator()
        self.context_builder = ContextBuilder(evidence_threshold=EVIDENCE_THRESHOLD)
        self.policy = load_policy()

    def recover_index(self):
        with self.index_lock:
            recovered = adaptive_rag.recover_active_index()
            self.collection = adaptive_rag.load_active_collection()
            self._refresh_components()
            return {"recovered": True, "collection": self.collection.name,
                    "fingerprint": recovered["fingerprint"],
                    "chunk_count": self.collection.count()}

    def reload_index(self):
        with self.index_lock:
            chunks = adaptive_rag.build_chunks(load_documents())
            self.collection = create_collection(chunks)
            self._refresh_components()
            save_deployment_manifest({**self.deployment_metadata,
                "index_collection": self.collection.name,
                "index_chunk_count": self.collection.count()})
            return {"collection": self.collection.name, "chunk_count": len(chunks)}

    def rollback_and_reload(self):
        with self.index_lock:
            previous = adaptive_rag.rollback_index()
            self.collection = adaptive_rag.load_active_collection()
            self._refresh_components()
            return {"rolled_back_to": previous["collection"],
                    "collection": self.collection.name,
                    "chunk_count": self.collection.count()}

    def persist_metrics(self):
        save_persistent_metrics({
            "total_requests": self.metrics["total_requests"],
            "successful_requests": self.metrics["successful_requests"],
            "failed_requests": self.metrics["failed_requests"],
            "degraded_requests": self.metrics["degraded_requests"],
            "total_latency_seconds": self.metrics[
                "total_latency_seconds"
            ],
            "alerts": self.alerts[-100:],
        })
        
    def _record(self, outcome, latency):
        with self.metrics_lock:
            self.metrics["total_requests"] += 1
            self.request_count += 1
            self.metrics[outcome] += 1
            self.metrics["total_latency_seconds"] += latency
            self.metrics["recent_latencies"].append(latency)
   
    def check_slo_breach(self):
        slo = self.slo_status()

        if slo["status"] == "healthy":
            return {"breached": False, "alerts": []}

        alerts = []

        for metric, passed in slo["checks"].items():
            if passed:
                continue

            # The checks use "p95_latency"; the targets/actual dicts use "p95_latency_seconds".
            key = "p95_latency_seconds" if metric == "p95_latency" else metric

            alerts.append({
                "type": "SLO_BREACH",
                "metric": metric,
                "actual": slo["actual"].get(key),
                "target": slo["targets"].get(key),
                "timestamp": time.time(),
            })

        self.alerts.extend(alerts)
        self.alerts = self.alerts[-100:]

        return {"breached": True, "alerts": alerts}
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
            "availability": availability >= SLO_TARGETS["availability"],
            "error_rate": error_rate <= SLO_TARGETS["error_rate"],
            "p95_latency": (
                metrics["latency_p95_seconds"]
                <= SLO_TARGETS["p95_latency_seconds"]
            ),
            "degraded_rate": (
                degraded_rate <= SLO_TARGETS["degraded_rate"]
            ),
        }

        return {
            "status": (
                "healthy"
                if all(checks.values())
                else "breached"
            ),
            "targets": SLO_TARGETS,
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
    
        def validate_deployment_manifest(self):
            problems = []
            active = adaptive_rag.load_active_index() or {}

            if active.get("collection") != self.collection.name:
                problems.append("active index differs from the loaded collection")

            if calculate_config_fingerprint(load_config()) != self.deployment_metadata["config_fingerprint"]:
                problems.append("config file changed since startup")

            if DEPLOYMENT_MANIFEST.exists():
                with open(DEPLOYMENT_MANIFEST, "r", encoding="utf-8") as f:
                    on_disk = json.load(f)
                for key in ("service_version", "config_fingerprint", "index_collection", "index_fingerprint"):
                    if on_disk.get(key) != self.deployment_metadata.get(key):
                        problems.append(f"manifest mismatch: {key}")
            else:
                problems.append("deployment manifest file is missing")

            return {"valid": not problems, "problems": problems}
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
    def shutdown(self):
        self.persist_metrics()

        return {
            "status": "shutdown_complete",
            "metrics_persisted": True,
        }

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