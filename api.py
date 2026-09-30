import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from config import load_config
from production_controller import ProductionRAGController

# ---------------------------------------------------------------- config
CONFIG = load_config()
OPERATIONAL_CONFIG = CONFIG.get("operational", {})

MAX_CONCURRENT_REQUESTS = OPERATIONAL_CONFIG.get("max_concurrent_requests", 4)
REQUEST_TIMEOUT_SECONDS = OPERATIONAL_CONFIG.get("request_timeout_seconds", 180)
INDEX_MONITOR_INTERVAL = OPERATIONAL_CONFIG.get("index_monitor_interval_seconds", 30)

request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

CONTAINER_IMAGE = os.getenv("RAG_IMAGE", "unknown")
DEPLOYMENT_ENVIRONMENT = os.getenv("DEPLOYMENT_ENVIRONMENT", "production")

# Single source of truth for the manifest location:
# explicit env var > /app (container) > folder of this script (local dev)
BASE_DIR = Path(__file__).resolve().parent
_default_dir = Path("/app") if Path("/app").exists() else BASE_DIR
DEPLOYMENT_MANIFEST_FILE = Path(
    os.getenv("DEPLOYMENT_MANIFEST_PATH", _default_dir / "deployment_manifest.json")
)


# ---------------------------------------------------------------- logging
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


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())

logger = logging.getLogger("rag_api")
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.addHandler(handler)
logger.propagate = False


# ---------------------------------------------------------------- manifest
def create_container_manifest(controller):
    meta = controller.deployment_metadata
    manifest = {
        "deployment_time": datetime.now(timezone.utc).isoformat(),
        "service_version": meta["service_version"],
        "container_image": CONTAINER_IMAGE,
        "deployment_environment": DEPLOYMENT_ENVIRONMENT,
        "config_fingerprint": meta["config_fingerprint"],
        "index_collection": meta["index_collection"],
        "index_fingerprint": meta["index_fingerprint"],
        "index_chunk_count": meta["index_chunk_count"],
    }

    DEPLOYMENT_MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DEPLOYMENT_MANIFEST_FILE, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    return manifest


def validate_container_manifest(controller):
    if not DEPLOYMENT_MANIFEST_FILE.exists():
        raise RuntimeError("Deployment manifest is missing")

    with open(DEPLOYMENT_MANIFEST_FILE, "r", encoding="utf-8") as file:
        manifest = json.load(file)

    meta = controller.deployment_metadata

    if manifest.get("container_image") != CONTAINER_IMAGE:
        raise RuntimeError("Deployment manifest image mismatch")

    if manifest.get("config_fingerprint") != meta["config_fingerprint"]:
        raise RuntimeError("Deployment manifest config fingerprint mismatch")

    if manifest.get("index_fingerprint") != meta["index_fingerprint"]:
        raise RuntimeError("Deployment manifest index fingerprint mismatch")

    return True


# ---------------------------------------------------------------- controller
controller = ProductionRAGController()
create_container_manifest(controller)
validate_container_manifest(controller)


# ---------------------------------------------------------------- background monitor
async def index_monitor(stop_event: asyncio.Event):
    while not stop_event.is_set():
        try:
            health = await run_in_threadpool(controller.health)
            controller.index_available = bool(health["healthy"])
            await run_in_threadpool(controller.check_slo_breach)
        except Exception:
            logger.exception("Index monitor iteration failed")

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=INDEX_MONITOR_INTERVAL)
        except asyncio.TimeoutError:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    stop_event = asyncio.Event()
    monitor_task = asyncio.create_task(index_monitor(stop_event))
    logger.info("RAG service started")

    yield

    stop_event.set()
    await monitor_task
    controller.shutdown()
    logger.info("RAG service shutdown complete")


app = FastAPI(title="Production RAG API", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------- models
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
    sources: list = Field(default_factory=list)
    error: Optional[Any] = None


# ---------------------------------------------------------------- routes
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service_version": controller.deployment_metadata["service_version"],
    }


@app.get("/ready")
def ready():
    health = controller.health()
    if not health["healthy"]:
        raise HTTPException(status_code=503, detail=health)
    return {"status": "ready", "collection": controller.collection.name}


@app.get("/metadata")
def metadata():
    return {
        **controller.deployment_metadata,
        "container_image": CONTAINER_IMAGE,
        "deployment_environment": DEPLOYMENT_ENVIRONMENT,
    }


@app.get("/metadata/validate")
def validate_metadata():
    result = controller.validate_deployment_manifest()
    if not result["valid"]:
        raise HTTPException(status_code=500, detail=result)
    return result


@app.get("/index/health")
def index_health_endpoint():
    return controller.index_health()


@app.get("/metrics")
def metrics_endpoint():
    return controller.metrics_snapshot()


@app.get("/metrics/slo")
def metrics_slo():
    return controller.slo_status()


@app.get("/metrics/alerts")
def metrics_alerts():
    return {"alerts": controller.alerts}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(payload: QueryRequest):
    if not payload.question or not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    async with request_semaphore:
        try:
            result = await asyncio.wait_for(
                run_in_threadpool(controller.query, payload.question),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=f"Request exceeded {REQUEST_TIMEOUT_SECONDS}s",
            )

    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result)

    return result


@app.post("/index/recover")
def recover_endpoint():
    return controller.recover_index()


@app.post("/index/reload")
def reload_endpoint():
    return controller.reload_index()


# One handler, two paths (previously defined twice)
@app.post("/index/rollback")
@app.post("/admin/rollback")
async def rollback_endpoint():
    try:
        result = await asyncio.to_thread(controller.rollback_and_reload)
        return {"status": "rolled_back", **result}
    except Exception as exc:
        logger.exception("Rollback failed")
        raise HTTPException(status_code=500, detail=str(exc))