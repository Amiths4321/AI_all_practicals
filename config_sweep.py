import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
SWEEP_DIR = BASE_DIR / "sweeps"


CONFIGURATIONS = [
    {
        "name": "baseline",
        "chunk_size": 500,
        "overlap": 100,
        "rerank_k": 10,
        "default_budget": 3,
    },
    {
        "name": "small_chunks",
        "chunk_size": 300,
        "overlap": 50,
        "rerank_k": 10,
        "default_budget": 3,
    },
    {
        "name": "large_chunks",
        "chunk_size": 700,
        "overlap": 150,
        "rerank_k": 10,
        "default_budget": 3,
    },
    {
        "name": "small_budget",
        "chunk_size": 500,
        "overlap": 100,
        "rerank_k": 10,
        "default_budget": 1,
    },
    {
        "name": "medium_budget",
        "chunk_size": 500,
        "overlap": 100,
        "rerank_k": 10,
        "default_budget": 2,
    },
    {
        "name": "larger_rerank",
        "chunk_size": 500,
        "overlap": 100,
        "rerank_k": 15,
        "default_budget": 3,
    },
]


def load_config():
    with open(
        BASE_DIR / "config.json",
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def save_config(config):
    with open(
        BASE_DIR / "config.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            config,
            file,
            indent=2
        )


def run_pipeline():
    scripts = [
        "adaptive_rag.py",
        "final_adaptive_evaluation.py",
        "final_rag_report.py",
        "failure_monitor.py",
    ]

    for script in scripts:
        result = subprocess.run(
            [sys.executable, script],
            cwd=BASE_DIR
        )

        if result.returncode != 0:
            return False

    return True


def collect_metrics():
    path = (
        BASE_DIR /
        "final_adaptive_evaluation.json"
    )

    if not path.exists():
        return None

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        report = json.load(file)

    summary = report.get(
        "summary",
        {}
    )

    return {
        "evidence_recall": summary.get(
            "adaptive_average_evidence_recall",
            0.0
        ),
        "quality": summary.get(
            "adaptive_average_quality_index",
            0.0
        ),
        "latency": summary.get(
            "adaptive_average_latency_seconds",
            0.0
        ),
        "context_characters": summary.get(
            "adaptive_average_context_characters",
            0.0
        ),
    }


def main():
    print("=" * 70)
    print("RAG CONFIGURATION SWEEP")
    print("=" * 70)

    original_config = load_config()

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    sweep_dir = (
        SWEEP_DIR /
        timestamp
    )

    sweep_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    results = []

    try:
        for index, experiment in enumerate(
            CONFIGURATIONS,
            start=1
        ):
            print()
            print("=" * 70)
            print(
                f"CONFIGURATION {index}/"
                f"{len(CONFIGURATIONS)}"
            )
            print("=" * 70)

            config = json.loads(
                json.dumps(original_config)
            )

            config["chunking"]["chunk_size"] = (
                experiment["chunk_size"]
            )

            config["chunking"]["overlap"] = (
                experiment["overlap"]
            )

            config["retrieval"]["rerank_k"] = (
                experiment["rerank_k"]
            )

            config["context"]["default_budget"] = (
                experiment["default_budget"]
            )

            print(
                "Name:",
                experiment["name"]
            )

            print(
                "Chunk:",
                experiment["chunk_size"]
            )

            print(
                "Overlap:",
                experiment["overlap"]
            )

            print(
                "Rerank:",
                experiment["rerank_k"]
            )

            print(
                "Budget:",
                experiment["default_budget"]
            )

            save_config(config)

            success = run_pipeline()

            if not success:
                results.append({
                    "configuration": experiment,
                    "status": "error"
                })

                continue

            metrics = collect_metrics()

            results.append({
                "configuration": experiment,
                "status": "completed",
                "metrics": metrics
            })

    finally:
        save_config(original_config)

    output = {
        "timestamp": timestamp,
        "configurations": len(
            CONFIGURATIONS
        ),
        "results": results
    }

    output_file = (
        sweep_dir /
        "sweep_results.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            output,
            file,
            indent=2
        )

    print()
    print("=" * 70)
    print("SWEEP COMPLETE")
    print("=" * 70)

    print(
        "Results:",
        output_file
    )


if __name__ == "__main__":
    main()