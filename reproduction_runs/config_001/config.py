import os
from dotenv import load_dotenv
from pathlib import Path
import json



load_dotenv()

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




BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "config.json"


def load_config():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
