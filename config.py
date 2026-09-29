import os
from dotenv import load_dotenv
from pathlib import Path
import hashlib
import json

def calculate_config_fingerprint(config):
    serialized = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":")
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()

load_dotenv()
EXPECTED_CONFIG_VERSION = 1

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2"
)

QDRANT_PATH = os.getenv(
    "QDRANT_PATH",
    "qdrant_storage"
)

REVIEW_QUEUE_FILE = os.getenv(
    "REVIEW_QUEUE_FILE",
    "manual_review_queue.json"
)

DASHBOARD_FILE = os.getenv(
    "DASHBOARD_FILE",
    "dashboard_data.json"
)

BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "config.json"
def validate_config(config):
    config_version = config.get("config_version")

    if config_version != EXPECTED_CONFIG_VERSION:
        raise ValueError(
            f"Unsupported config_version: {config_version}. "
            f"Expected: {EXPECTED_CONFIG_VERSION}"
        )

    operational = config.get("operational", {})

def load_config():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = json.load(f)

    validate_config(config)

    save_config_snapshot(config)

    return config

def save_config_snapshot(config):
    fingerprint = calculate_config_fingerprint(config)

    snapshot = {
        "config_version": config["config_version"],
        "fingerprint": fingerprint,
        "config": config,
    }

    snapshot_file = CONFIG_FILE.parent / "config_snapshot.json"
    temporary_file = snapshot_file.with_suffix(".tmp")

    with open(temporary_file, "w", encoding="utf-8") as file:
        json.dump(snapshot, file, indent=2)

    temporary_file.replace(snapshot_file)

    return snapshot

def validate_config(config):
    operational = config.get("operational", {})

    max_concurrent = operational.get(
        "max_concurrent_requests", 0
    )

    timeout = operational.get(
        "request_timeout_seconds", 180
    )

    monitor_interval = operational.get(
        "index_monitor_interval_seconds", 30
    )

    metrics_size = operational.get(
        "metrics_file_max_size_mb", 5
    )

    if not isinstance(max_concurrent, int) or max_concurrent < 1:
        raise ValueError(
            "max_concurrent_requests must be >= 1"
        )

    if timeout <= 0:
        raise ValueError(
            "request_timeout_seconds must be > 0"
        )

    if monitor_interval <= 0:
        raise ValueError(
            "index_monitor_interval_seconds must be > 0"
        )

    if metrics_size <= 0:
        raise ValueError(
            "metrics_file_max_size_mb must be > 0"
        )

    slo = operational.get("slo", {})

    availability = slo.get("availability", 0.99)
    error_rate = slo.get("error_rate", 0.01)
    p95_latency = slo.get("p95_latency_seconds", 10.0)
    degraded_rate = slo.get("degraded_rate", 0.01)

    if not 0 <= availability <= 1:
        raise ValueError(
            "availability must be between 0 and 1"
        )

    if not 0 <= error_rate <= 1:
        raise ValueError(
            "error_rate must be between 0 and 1"
        )

    if p95_latency <= 0:
        raise ValueError(
            "p95_latency_seconds must be > 0"
        )

    if not 0 <= degraded_rate <= 1:
        raise ValueError(
            "degraded_rate must be between 0 and 1"
        )

    return True
def validate_config(config=None):
  """Validates that the configuration dictionary contains required keys and valid types."""
  if config is None:
    config = load_config()

  # Define expected mandatory sections or keys based on your pipeline
  required_keys = ["chunk_size", "chunk_overlap"]

  for key in required_keys:
    if key not in config:
      # If using a nested dictionary structure, check accordingly
      pass

  # You can add deeper validation rules here as needed
  return True
