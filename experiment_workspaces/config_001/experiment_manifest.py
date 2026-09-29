from pathlib import Path
import hashlib
import json
import platform
import sys
import time


BASE_DIR = Path(__file__).resolve().parent


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def fingerprint_files(files):
    result = {}

    for filename in files:

        path = BASE_DIR / filename

        if path.exists():
            result[filename] = sha256_file(path)

    return result


def create_manifest(
    experiment_id,
    configuration,
    workspace,
    started_at,
    finished_at,
    retry_history=None,
):
    pipeline_files = [
        "adaptive_rag.py",
        "final_adaptive_evaluation.py",
        "final_rag_report.py",
        "failure_monitor.py",
        "config.py",
        "chunking.py",
        "hybrid_search.py",
        "reranking.py",
        "semantic_evidence.py",
        "context_builder.py",
        "context_compression.py",
        "sentence_compression.py",
        "cached_generation.py",
        "cached_judge.py",
        "cache.py",
        "generation.py",
        "rag_judge.py",
    ]

    input_files = [
        "evaluation_data.json",
        "evidence_policy_benchmark.json",
        "empirical_context_policy.json",
        "documents/leave_policy.txt",
        "documents/employee_handbook.txt",
    ]

    output_files = [
        path.name
        for path in workspace.iterdir()
        if path.is_file()
    ]

    manifest = {
        "experiment_id": experiment_id,

        "created_at": time.time(),

        "started_at": started_at,

        "finished_at": finished_at,

        "duration_seconds": (
            finished_at - started_at
        ),

        "configuration": configuration,

        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },

        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },

        "pipeline_fingerprint": fingerprint_files(
            pipeline_files
        ),

        "input_fingerprint": fingerprint_files(
            input_files
        ),

        "outputs": sorted(output_files),

        "retry_history": retry_history or [],
    }

    return manifest


def save_manifest(manifest, path):

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            manifest,
            f,
            indent=2
        )


if __name__ == "__main__":

    print(
        "Manifest utilities loaded successfully."
    )