import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    "adaptive_rag.py",
    "final_adaptive_evaluation.py",
    "final_rag_report.py",
]


def run_script(script, experiment_dir):
    print()
    print("=" * 70)
    print(f"RUNNING: {script}")
    print("=" * 70)

    result = subprocess.run(
        [sys.executable, script],
        cwd=BASE_DIR,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{script} failed with exit code "
            f"{result.returncode}"
        )

    # Copy generated JSON artifacts into the experiment directory.
    json_files = [
        "adaptive_rag_report.json",
        "final_adaptive_evaluation.json",
        "final_rag_report.json",
    ]

    for filename in json_files:
        source = BASE_DIR / filename

        if source.exists():
            destination = experiment_dir / filename
            shutil.copy2(source, destination)
            print(f"Captured: {filename}")


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    experiment_dir = BASE_DIR / "experiments" / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RAG EXPERIMENT RUNNER")
    print("=" * 70)

    print(f"Experiment directory:\n{experiment_dir}")

    for script in SCRIPTS:
        script_path = BASE_DIR / script

        if not script_path.exists():
            raise FileNotFoundError(
                f"Required script not found: {script_path}"
            )

    for script in SCRIPTS:
        run_script(script, experiment_dir)

    # Save a small manifest describing exactly what was executed.
    manifest = experiment_dir / "experiment_manifest.json"

    manifest_data = {
        "timestamp": timestamp,
        "scripts": SCRIPTS,
        "configuration": {
            "chunk_size": 500,
            "overlap": 100,
            "vector_k": 10,
            "hybrid_k": 10,
            "rerank_k": 10,
            "default_budget": 3,
            "evidence_threshold": 0.60
        }
    }

    with open(manifest, "w", encoding="utf-8") as file:
        json.dump(manifest_data, file, indent=2)

    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    print(f"Artifacts saved to:\n{experiment_dir}")


if __name__ == "__main__":
    main()