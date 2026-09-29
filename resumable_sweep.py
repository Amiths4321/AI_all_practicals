import copy
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

CONFIG_FILE = BASE_DIR / "config.json"
SWEEP_CONFIG_FILE = BASE_DIR / "sweep_configurations.json"

SWEEPS_DIR = BASE_DIR / "sweeps"

MAX_CONFIGURATIONS = None
# Example:
# MAX_CONFIGURATIONS = 10


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def save_json(path, data):
    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            data,
            file,
            indent=2
        )


def load_base_config():
    return load_json(CONFIG_FILE)


def save_base_config(config):
    save_json(
        CONFIG_FILE,
        config
    )


def load_sweep_configurations():
    data = load_json(
        SWEEP_CONFIG_FILE
    )

    return data.get(
        "configurations",
        []
    )


def apply_configuration(
    base_config,
    experiment
):
    config = copy.deepcopy(
        base_config
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

    return config


def run_pipeline():
    scripts = [
        "adaptive_rag.py",
        "final_adaptive_evaluation.py",
        "final_rag_report.py",
        "failure_monitor.py",
    ]

    for script in scripts:

        print()
        print(
            f"Running {script}..."
        )

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

    report = load_json(path)

    summary = report.get(
        "summary",
        {}
    )

    return {
        "evidence_recall":
            summary.get(
                "adaptive_average_evidence_recall",
                0.0
            ),
        "quality":
            summary.get(
                "adaptive_average_quality_index",
                0.0
            ),
        "latency":
            summary.get(
                "adaptive_average_latency_seconds",
                0.0
            ),
        "context_characters":
            summary.get(
                "adaptive_average_context_characters",
                0.0
            )
    }


def create_sweep():
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    sweep_dir = (
        SWEEPS_DIR /
        timestamp
    )

    sweep_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    state = {
        "created_at": timestamp,
        "status": "running",
        "results": []
    }

    state_file = (
        sweep_dir /
        "sweep_state.json"
    )

    save_json(
        state_file,
        state
    )

    return sweep_dir, state_file, state


def load_or_create_sweep():
    if not SWEEPS_DIR.exists():
        return create_sweep()

    existing = sorted(
        [
            path
            for path in SWEEPS_DIR.iterdir()
            if path.is_dir()
            and (
                path /
                "sweep_state.json"
            ).exists()
        ],
        key=lambda path: path.name,
        reverse=True
    )

    if existing:
        sweep_dir = existing[0]

        state_file = (
            sweep_dir /
            "sweep_state.json"
        )

        state = load_json(
            state_file
        )

        if state.get("status") == "running":
            print()
            print(
                "Resuming sweep:",
                sweep_dir.name
            )

            return (
                sweep_dir,
                state_file,
                state
            )

    return create_sweep()


def completed_ids(state):
    return {
        result["configuration_id"]
        for result in state.get(
            "results",
            []
        )
        if result.get("status")
        == "completed"
    }


def main():
    print("=" * 70)
    print("RESUMABLE RAG CONFIGURATION SWEEP")
    print("=" * 70)

    configurations = (
        load_sweep_configurations()
    )

    if MAX_CONFIGURATIONS is not None:
        configurations = configurations[
            :MAX_CONFIGURATIONS
        ]

    if not configurations:
        print(
            "No configurations found."
        )
        return 1

    base_config = load_base_config()

    (
        sweep_dir,
        state_file,
        state
    ) = load_or_create_sweep()

    completed = completed_ids(
        state
    )

    print()
    print(
        "Total configurations:",
        len(configurations)
    )

    print(
        "Already completed:",
        len(completed)
    )

    try:

        for index, experiment in enumerate(
            configurations,
            start=1
        ):

            config_id = experiment["id"]

            if config_id in completed:
                print()
                print(
                    f"[{index}/{len(configurations)}]"
                    f" {config_id} - SKIP"
                )
                continue

            print()
            print("=" * 70)
            print(
                f"[{index}/{len(configurations)}]"
            )
            print(
                f"Configuration: {config_id}"
            )
            print("=" * 70)

            print(
                "chunk_size:",
                experiment["chunk_size"]
            )

            print(
                "overlap:",
                experiment["overlap"]
            )

            print(
                "rerank_k:",
                experiment["rerank_k"]
            )

            print(
                "default_budget:",
                experiment["default_budget"]
            )

            experiment_config = (
                apply_configuration(
                    base_config,
                    experiment
                )
            )

            save_base_config(
                experiment_config
            )

            started_at = (
                datetime.now().isoformat()
            )

            success = run_pipeline()

            finished_at = (
                datetime.now().isoformat()
            )

            if success:

                metrics = collect_metrics()

                result = {
                    "configuration_id":
                        config_id,
                    "configuration":
                        experiment,
                    "status":
                        "completed",
                    "started_at":
                        started_at,
                    "finished_at":
                        finished_at,
                    "metrics":
                        metrics
                }

                print(
                    "Status: COMPLETED"
                )

            else:

                result = {
                    "configuration_id":
                        config_id,
                    "configuration":
                        experiment,
                    "status":
                        "error",
                    "started_at":
                        started_at,
                    "finished_at":
                        finished_at
                }

                print(
                    "Status: ERROR"
                )

            state["results"].append(
                result
            )

            save_json(
                state_file,
                state
            )

            print(
                "Progress saved."
            )

    except KeyboardInterrupt:

        print()
        print(
            "Sweep interrupted."
        )

        print(
            "Progress has been saved."
        )

        return 1

    finally:
        save_base_config(
            base_config
        )

    state["status"] = "completed"

    save_json(
        state_file,
        state
    )

    results_file = (
        sweep_dir /
        "sweep_results.json"
    )

    save_json(
        results_file,
        {
            "created_at":
                state["created_at"],
            "completed_at":
                datetime.now().isoformat(),
            "configurations":
                len(configurations),
            "results":
                state["results"]
        }
    )

    print()
    print("=" * 70)
    print("SWEEP COMPLETE")
    print("=" * 70)

    print(
        "Results:",
        results_file
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())