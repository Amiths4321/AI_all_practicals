import concurrent.futures
import copy
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

CONFIG_FILE = BASE_DIR / "config.json"
SWEEP_CONFIG_FILE = BASE_DIR / "sweep_configurations.json"

OUTPUT_DIR = BASE_DIR / "parallel_sweeps"

MAX_WORKERS = 2
MAX_CONFIGURATIONS = 4


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def build_config(base_config, experiment):
    config = copy.deepcopy(base_config)

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

    return config


def run_one_experiment(experiment):
    config_id = experiment["id"]

    run_dir = (
        OUTPUT_DIR /
        config_id
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    config_path = (
        run_dir /
        "config.json"
    )

    base_config = load_json(
        CONFIG_FILE
    )

    experiment_config = build_config(
        base_config,
        experiment
    )

    save_json(
        config_path,
        experiment_config
    )

    started = time.perf_counter()

    try:
        result = subprocess.run(
            [
                sys.executable,
                "adaptive_rag.py"
            ],
            cwd=BASE_DIR,
            capture_output=True,
            text=True
        )

        elapsed = (
            time.perf_counter()
            - started
        )

        if result.returncode != 0:
            return {
                "configuration_id": config_id,
                "status": "error",
                "runtime_seconds": elapsed,
                "stderr": result.stderr[-3000:]
            }

        return {
            "configuration_id": config_id,
            "status": "completed",
            "runtime_seconds": elapsed
        }

    except Exception as error:
        return {
            "configuration_id": config_id,
            "status": "error",
            "error": str(error)
        }


def main():
    print("=" * 70)
    print("CONTROLLED PARALLEL SWEEP")
    print("=" * 70)

    configurations = load_json(
        SWEEP_CONFIG_FILE
    )["configurations"]

    configurations = configurations[
        :MAX_CONFIGURATIONS
    ]

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_dir = (
        OUTPUT_DIR /
        timestamp
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print(
        "Configurations:",
        len(configurations)
    )

    print(
        "Workers:",
        MAX_WORKERS
    )

    results = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {
            executor.submit(
                run_one_experiment,
                experiment
            ): experiment
            for experiment in configurations
        }

        for future in concurrent.futures.as_completed(
            futures
        ):
            experiment = futures[future]

            try:
                result = future.result()

            except Exception as error:
                result = {
                    "configuration_id":
                        experiment["id"],
                    "status":
                        "error",
                    "error":
                        str(error)
                }

            results.append(result)

            print()
            print(
                experiment["id"],
                "→",
                result["status"]
            )

    output = {
        "timestamp": timestamp,
        "workers": MAX_WORKERS,
        "configurations": configurations,
        "results": results
    }

    output_file = (
        output_dir /
        "parallel_sweep_results.json"
    )

    save_json(
        output_file,
        output
    )

    print()
    print("=" * 70)
    print("PARALLEL SWEEP COMPLETE")
    print("=" * 70)

    print(
        "Saved:",
        output_file
    )


if __name__ == "__main__":
    main()