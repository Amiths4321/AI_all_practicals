from pathlib import Path
import json
import os

BASE_DIR = Path(__file__).resolve().parent

CONFIG = {
    "chunk_size": 500,
    "chunk_overlap": 50,
}

def load_config():
    '''Safely loads and returns the configuration dictionary from config.json if available.'''
    config_file = BASE_DIR / "config.json"
    if config_file.exists() and config_file.stat().st_size > 0:
        with open(config_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return CONFIG
