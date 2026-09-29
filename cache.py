import hashlib
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / ".cache"


def make_key(prefix, data):
    serialized = json.dumps(
        data,
        sort_keys=True,
        ensure_ascii=False
    )

    digest = hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()

    return f"{prefix}_{digest}"


def get_cache_path(key):
    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    return CACHE_DIR / f"{key}.json"


def cache_get(key):

    path = get_cache_path(key)

    if not path.exists():
        return None

    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:
            return json.load(file)

    except (
        json.JSONDecodeError,
        OSError
    ):
        return None


def cache_set(key, value):

    path = get_cache_path(key)

    temporary_path = path.with_suffix(
        ".tmp"
    )

    with open(
        temporary_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            value,
            file,
            indent=2,
            ensure_ascii=False
        )

    temporary_path.replace(path)


def cache_clear():

    if not CACHE_DIR.exists():
        return

    for path in CACHE_DIR.glob("*.json"):

        try:
            path.unlink()
        except OSError:
            pass