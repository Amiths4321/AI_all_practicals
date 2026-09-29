import hashlib
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def get_git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            check=True,
        )

        return result.stdout.strip()

    except Exception:
        return None


def get_git_branch():
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            check=True,
        )

        return result.stdout.strip()

    except Exception:
        return None


def fingerprint_files(files):
    result = {}

    for file_path in files:
        path = Path(file_path)

        if not path.exists():
            continue

        result[str(path)] = sha256_file(path)

    return result


def create_version_record(configuration):
    pipeline_files = [
        "adaptive_rag.py",
        "retriever.py",
        "hybrid_retriever.py",
        "reranker.py",
        "context_builder.py",
        "context_compression.py",
        "sentence_compression.py",
        "generation.py",
        "rag_judge.py",
        "semantic_evidence.py",
        "config.py",
    ]

    input_files = [
        "config.json",
        "evaluation_data.json",
        "evidence_policy_benchmark.json",
        "empirical_context_policy.json",
        "documents/leave_policy.txt",
        "documents/employee_handbook.txt",
    ]

    return {
        "version_record": 1,
        "created_at": datetime.now().isoformat(),

        "git": {
            "commit": get_git_commit(),
            "branch": get_git_branch(),
        },

        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },

        "configuration": configuration,

        "pipeline_fingerprint": fingerprint_files(
            pipeline_files
        ),

        "input_fingerprint": fingerprint_files(
            input_files
        ),
    }


def save_version_record(record, path="version_record.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)


if __name__ == "__main__":
    with open("config.json", "r", encoding="utf-8") as f:
        configuration = json.load(f)

    record = create_version_record(configuration)
    save_version_record(record)

    print("Version record created.")
    print("Output: version_record.json")