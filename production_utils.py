import json
import logging
import time
from pathlib import Path


def setup_logging(log_file="rag_pipeline.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(
                log_file,
                encoding="utf-8"
            ),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger("rag_pipeline")


def load_json_file(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Required file does not exist: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def save_json_file(data, path):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with open(
        temporary_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False
        )

    temporary_path.replace(path)


def timed_call(function, *args, **kwargs):
    start = time.perf_counter()

    result = function(
        *args,
        **kwargs
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    return result, elapsed


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (
        TypeError,
        ValueError
    ):
        return default


def validate_required_keys(
    data,
    required_keys,
    name="object"
):
    if not isinstance(data, dict):
        raise ValueError(
            f"{name} must be a dictionary."
        )

    missing = [
        key
        for key in required_keys
        if key not in data
    ]

    if missing:
        raise ValueError(
            f"{name} is missing required keys: "
            f"{missing}"
        )

    return True