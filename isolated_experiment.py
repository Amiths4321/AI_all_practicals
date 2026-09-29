import copy
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
WORKSPACES_DIR = BASE_DIR / "experiment_workspaces"

PIPELINE_SCRIPTS = [
    "adaptive_rag.py",
    "final_adaptive_evaluation.py",
    "final_rag_report.py",
    "failure_monitor.py",
]

OUTPUT_FILES = [
    "adaptive_rag_report.json",
    "final_adaptive_evaluation.json",
    "final_rag_report.json",
    "failure_monitoring.json",
]


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


def prepare_workspace(experiment):
    workspace = (
        WORKSPACES_DIR /
        experiment["id"]
    )

    workspace.mkdir(
        parents=True,
        exist_ok=True
    )

    # Copy source files required by the pipeline.
    shutil.copy2(
        BASE_DIR / "adaptive_rag.py",
        workspace / "adaptive_rag.py"
    )

    for filename in [
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
          "cached_generation.py",
          "cached_judge.py",
          "cache.py",
          "generation.py",
          "rag_judge.py",
          ]:
        source = BASE_DIR / filename

        if source.exists():
            shutil.copy2(
                source,
                workspace / filename
            )

    documents_dir = workspace / "documents"
    documents_dir.mkdir(
        exist_ok=True
    )

    for filename in [
        "leave_policy.txt",
        "employee_handbook.txt",
    ]:
        shutil.copy2(
            BASE_DIR / "documents" / filename,
            documents_dir / filename
        )

    # Copy evaluation inputs.
    for filename in [
        "evaluation_data.json",
        "empirical_context_policy.json",
    ]:
        shutil.copy2(
            BASE_DIR / filename,
            workspace / filename
        )

    # Build isolated configuration.
    base_config = load_json(
        BASE_DIR / "config.json"
    )

    experiment_config = build_config(
        base_config,
        experiment
    )

    save_json(
        workspace / "config.json",
        experiment_config
    )

    return workspace


def run_pipeline(workspace):
    for script in PIPELINE_SCRIPTS:
        result = subprocess.run(
            [sys.executable, script],
            cwd=workspace
        )

        if result.returncode != 0:
            return False

    return True


def archive_outputs(workspace):
    archive_dir = (
        workspace /
        "outputs"
    )

    archive_dir.mkdir(
        exist_ok=True
    )

    for filename in OUTPUT_FILES:
        source = workspace / filename

        if source.exists():
            shutil.copy2(
                source,
                archive_dir / filename
            )


def run_experiment(experiment):
    print()
    print("=" * 70)
    print(
        "EXPERIMENT:",
        experiment["id"]
    )
    print("=" * 70)

    workspace = prepare_workspace(
        experiment
    )

    print(
        "Workspace:",
        workspace
    )

    success = run_pipeline(
        workspace
    )

    if success:
        archive_outputs(
            workspace
        )

        print(
            "Status: COMPLETED"
        )

        return True

    print(
        "Status: FAILED"
    )

    return False


def main():
    print("=" * 70)
    print("ISOLATED EXPERIMENT")
    print("=" * 70)

    sweep_file = (
        BASE_DIR /
        "sweep_configurations.json"
    )

    data = load_json(
        sweep_file
    )

    configurations = data[
        "configurations"
    ]

    # Test only one configuration first.
    experiment = configurations[0]

    success = run_experiment(
        experiment
    )

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())