from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import json
import logging
import sys
from datetime import datetime, timezone

from production_controller import ProductionRAGController

app = FastAPI(title="Production RAG API", version="1.0.0")

from fastapi import HTTPException

@app.post("/admin/rollback")
def rollback_admin_index():
    try:
        return {"status": "success", **controller.rollback()}
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))

from fastapi.responses import RedirectResponse

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")
    
# Initialize the RAG controller once at startup
controller = ProductionRAGController()

def percentile(values, pct):
    if not values:
        return 0.0
    values = sorted(values)
    return values[int((pct / 100) * (len(values) - 1))]

class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


# Everything below is at module level (no indentation), AFTER the class
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())

logger = logging.getLogger("rag_api")
logger.setLevel(logging.INFO)
logger.handlers.clear()   # avoid duplicate handlers on --reload
logger.addHandler(handler)
logger.propagate = False

class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    request_id: str
    status: str
    timestamp: Optional[str] = None
    question: Optional[str] = None
    answerable: Optional[bool] = None
    answer: Optional[str] = None
    policy: dict = Field(default_factory=dict)
    evidence: dict = Field(default_factory=dict)
    quality: dict = Field(default_factory=dict)
    performance: dict = Field(default_factory=dict)
    cache: dict = Field(default_factory=dict)
    error: Optional[Any] = None


@app.get("/health")
def health_endpoint():
    return controller.health()


@app.get("/index/health")
def index_health_endpoint():
    return controller.index_health()


@app.get("/metrics")
def metrics_endpoint():
    return controller.metrics_snapshot()


@app.post("/query", response_model=QueryResponse)
def query_endpoint(payload: QueryRequest):
    if not payload.question or not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = controller.query(payload.question)

    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result)

    return result


@app.post("/index/recover")
def recover_endpoint():
    return controller.recover_index()


@app.post("/index/reload")
def reload_endpoint():
    return controller.reload_index()


@app.post("/index/rollback")
def rollback_endpoint():
    return controller.rollback_and_reload()

# TODO: adjust these imports to wherever these names actually live in your project.
from adaptive_rag import (
    EVIDENCE_THRESHOLD,
    ContextBuilder,
    DocumentReranker,
    HybridRetriever,
    SemanticEvidenceEvaluator,
    create_collection,
    load_documents,
    load_policy,
    process_question,
)


class ProductionRAGController:

    def __init__(self):
        self.started_at = datetime.now().isoformat()

        # Thread-safety primitives
        self._request_counter = itertools.count(1)
        self._metrics_lock = threading.Lock()
        self.index_lock = threading.Lock()    # guards reads/swaps of components (short)
        self._rebuild_lock = threading.Lock() # serialises slow rebuilds/reloads

        self.index_available = True
        self._chroma_client = None

        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "degraded_requests": 0,
            "total_latency_seconds": 0.0,
            "recent_latencies": deque(maxlen=100),
        }

        print("Initializing production RAG...")

        documents = load_documents()
        chunks = adaptive_rag.build_chunks(documents)

        self.reranker = DocumentReranker()
        self._components = self._build_components(chunks, create_collection(chunks))

        print(f"Loaded documents: {len(documents)}")
        print(f"Loaded chunks: {len(chunks)}")
        print("Production RAG initialized.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_components(self, chunks, collection):
        """Build a fresh set of retrieval components (does not touch self state)."""
        return {
            "collection": collection,
            "hybrid_retriever": HybridRetriever(documents=chunks),
            "evidence_evaluator": SemanticEvidenceEvaluator(),
            "context_builder": ContextBuilder(evidence_threshold=EVIDENCE_THRESHOLD),
            "policy": load_policy(),
        }

    def _swap_components(self, components):
        with self.index_lock:
            self._components = components

    def _snapshot_components(self):
        with self.index_lock:
            return self._components

    @property
    def collection(self):
        return self._snapshot_components()["collection"]

    def _get_chroma_client(self):
        if self._chroma_client is None:
            self._chroma_client = chromadb.PersistentClient(
                path=str(adaptive_rag.INDEX_DIR)
            )
        return self._chroma_client

    def _record(self, outcome, total_latency):
        with self._metrics_lock:
            self.metrics["total_latency_seconds"] += total_latency
            self.metrics["recent_latencies"].append(total_latency)
            if outcome == "error":
                self.metrics["failed_requests"] += 1
            elif outcome == "degraded":
                self.metrics["degraded_requests"] += 1
            else:
                self.metrics["successful_requests"] += 1

    # ------------------------------------------------------------------
    # Health / metrics
    # ------------------------------------------------------------------
    def health(self):
        client = self._get_chroma_client()
        active = adaptive_rag.load_active_index()
        is_valid = adaptive_rag.validate_index_state(client, active)
        history = adaptive_rag.load_index_history()

        self.index_available = bool(is_valid)

        return {
            "healthy": is_valid,
            "status": "healthy" if is_valid else "degraded",
            "active_index": active,
            "is_valid": is_valid,
            "history_count": len(history),
            "history": history,
        }

    def index_health(self):
        return self.health()

    def metrics_snapshot(self):
        with self._metrics_lock:
            total = self.metrics["total_requests"]
            latencies = list(self.metrics["recent_latencies"])
            average_latency = (
                self.metrics["total_latency_seconds"] / total if total else 0.0
            )
            return {
                "total_requests": total,
                "successful_requests": self.metrics["successful_requests"],
                "failed_requests": self.metrics["failed_requests"],
                "degraded_requests": self.metrics["degraded_requests"],
                "average_latency_seconds": average_latency,
                "recent_latency_count": len(latencies),
                "max_recent_latency_seconds": max(latencies) if latencies else 0.0,
                "index_available": self.index_available,
            }

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def query(self, question):
        request_started = time.perf_counter()

        request_id = f"rag-{next(self._request_counter):06d}"
        with self._metrics_lock:
            self.metrics["total_requests"] += 1

        if not question or not question.strip():
            self._record("error", time.perf_counter() - request_started)
            return {
                "request_id": request_id,
                "status": "error",
                "error": "Question cannot be empty.",
            }

        question = question.strip()

        try:
            # Snapshot so a concurrent reload can't swap parts mid-request
            comps = self._snapshot_components()

            case = {
                "question": question,
                "required_evidence": [],
                "answerable": True,
            }

            result = process_question(
                case=case,
                collection=comps["collection"],
                hybrid_retriever=comps["hybrid_retriever"],
                reranker=self.reranker,
                evidence_evaluator=comps["evidence_evaluator"],
                context_builder=comps["context_builder"],
                policy=comps["policy"],
            )

            latency = time.perf_counter() - request_started
            status = result.get("status", "unknown")

            if status == "error":
                outcome = "error"
            elif status == "degraded":
                outcome = "degraded"
            else:
                outcome = "success"
            self._record(outcome, latency)

            return {
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
                "status": status,
                "question": question,
                "answerable": result.get("answerable"),
                "answer": result.get("answer"),
                "policy": {
                    "budget": result.get("budget"),
                    "reason": result.get("budget_reason"),
                },
                "evidence": {
                    "retrieval_recall": result.get("retrieval_evidence_recall"),
                    "context_recall": result.get("context_evidence_recall"),
                },
                "quality": result.get("judgement", {}),
                "performance": {
                    "controller_latency_seconds": latency,
                    "generation_latency_seconds": result.get("latency_seconds"),
                },
                "cache": {
                    "generation_hit": result.get("generation_cache_hit", False),
                    "judge_hit": result.get("judge_cache_hit", False),
                },
            }

        except Exception as exc:
            latency = time.perf_counter() - request_started
            self._record("error", latency)

            return {
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "question": question,
                "performance": {"controller_latency_seconds": latency},
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------
    def recover_index(self):
        with self._rebuild_lock:
            recovered = adaptive_rag.recover_active_index()
            collection = adaptive_rag.load_active_collection()

            # NOTE: hybrid retriever is rebuilt from current documents; make sure
            # these match the chunks stored in the restored collection.
            chunks = adaptive_rag.build_chunks(load_documents())
            self._swap_components(self._build_components(chunks, collection))
            self.index_available = True

            return {
                "recovered": True,
                "collection": collection.name,
                "fingerprint": recovered["fingerprint"],
                "chunk_count": collection.count(),
            }

    def reload_index(self):
        with self._rebuild_lock:
            chunks = adaptive_rag.build_chunks(load_documents())
            collection = create_collection(chunks)
            self._swap_components(self._build_components(chunks, collection))
            self.index_available = True

            return {
                "collection": collection.name,
                "chunk_count": len(chunks),
            }

    def rollback_and_reload(self):
        with self._rebuild_lock:
            previous = adaptive_rag.rollback_index()
            collection = adaptive_rag.load_active_collection()

            # NOTE: same consistency caveat as recover_index().
            chunks = adaptive_rag.build_chunks(load_documents())
            self._swap_components(self._build_components(chunks, collection))
            self.index_available = True

            return {
                "rolled_back_to": previous["collection"],
                "collection": collection.name,
                "chunk_count": collection.count(),
            }
@app.get("/metrics/slo")
def metrics_slo():
    if controller is None:
        raise HTTPException(
            status_code=503,
            detail="RAG controller is not ready"
        )

    return controller.slo_status()

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