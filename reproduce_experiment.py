from pathlib import Path
import json
import shutil
import subprocess
import sys
import time


BASE_DIR = Path(__file__).resolve().parent

OUTPUT_ROOT = BASE_DIR / "experiment_outputs"
REPRO_ROOT = BASE_DIR / "reproduction_runs"

PIPELINE = [
    "adaptive_rag.py",
    "final_adaptive_evaluation.py",
    "final_rag_report.py",
    "failure_monitor.py",
]

PIPELINE_FILES = [
    "adaptive_rag.py",
    "final_adaptive_evaluation.py",
    "final_rag_report.py",
    "failure_monitor.py",
    "experiment_manifest.py",
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

INPUT_FILES = [
    "evaluation_data.json",
    "evidence_policy_benchmark.json",
    "empirical_context_policy.json",
]

DOCUMENT_FILES = [
    "documents/leave_policy.txt",
    "documents/employee_handbook.txt",
]


def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def prepare_workspace(
    experiment_id,
    configuration
):

    workspace = (
        REPRO_ROOT / experiment_id
    )

    if workspace.exists():
        shutil.rmtree(workspace)

    workspace.mkdir(
        parents=True
    )

    for filename in PIPELINE_FILES:

        source = BASE_DIR / filename

        if not source.exists():
            raise FileNotFoundError(
                f"Missing: {source}"
            )

        shutil.copy2(
            source,
            workspace / filename
        )

    for filename in INPUT_FILES:

        source = BASE_DIR / filename

        if not source.exists():
            raise FileNotFoundError(
                f"Missing: {source}"
            )

        shutil.copy2(
            source,
            workspace / filename
        )

    for filename in DOCUMENT_FILES:

        source = BASE_DIR / filename

        destination = (
            workspace / filename
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        shutil.copy2(
            source,
            destination
        )

    config = {
        "chunking": {
            "chunk_size": configuration[
                "chunk_size"
            ],
            "overlap": configuration[
                "overlap"
            ],
        },
        "retrieval": {
            "vector_k": 10,
            "hybrid_k": 10,
            "rerank_k": configuration[
                "rerank_k"
            ],
        },
        "context": {
            "default_budget": configuration[
                "default_budget"
            ],
        },
        "evidence": {
            "threshold": 0.60,
        },
    }

    with open(
        workspace / "config.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            config,
            f,
            indent=2
        )

    return workspace


def run_pipeline(workspace):

    results = []

    for script in PIPELINE:

        started = time.time()

        result = subprocess.run(
            [
                sys.executable,
                script,
            ],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=900,
        )

        results.append({
            "script": script,
            "returncode": result.returncode,
            "elapsed_seconds": (
                time.time() - started
            ),
            "stdout": result.stdout[-3000:],
            "stderr": result.stderr[-3000:],
        })

        if result.returncode != 0:

            raise RuntimeError(
                f"{script} failed:\n"
                f"{result.stderr}"
            )

    return results


def extract_evaluation(workspace):

    path = (
        workspace
        / "final_adaptive_evaluation.json"
    )

    if not path.exists():
        return None

    return load_json(path)


def normalize_metrics(evaluation):

    if evaluation is None:
        return {}

    if isinstance(evaluation, dict):

        result = {}

        for key in [
            "evidence_recall",
            "quality",
            "average_latency_seconds",
            "average_context_characters",
            "success_rate",
        ]:

            if key in evaluation:
                result[key] = evaluation[key]

        return result

    if isinstance(evaluation, list):

        return {
            "record_count": len(evaluation)
        }

    return {}


def compare_metrics(
    original,
    reproduced,
    tolerance=0.05
):

    comparison = {}

    keys = sorted(
        set(original) |
        set(reproduced)
    )

    for key in keys:

        old = original.get(key)
        new = reproduced.get(key)

        if (
            isinstance(old, (int, float))
            and isinstance(new, (int, float))
        ):

            difference = new - old

            comparison[key] = {
                "original": old,
                "reproduced": new,
                "difference": difference,
                "within_tolerance": (
                    abs(difference)
                    <= tolerance
                ),
            }

        else:

            comparison[key] = {
                "original": old,
                "reproduced": new,
                "equal": old == new,
            }

    return comparison


def main():

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            "python reproduce_experiment.py "
            "config_001"
        )

        sys.exit(1)

    experiment_id = sys.argv[1]

    original_dir = (
        OUTPUT_ROOT / experiment_id
    )

    manifest_path = (
        original_dir
        / "experiment_manifest.json"
    )

    if not manifest_path.exists():

        raise FileNotFoundError(
            f"Manifest not found: "
            f"{manifest_path}"
        )

    manifest = load_json(
        manifest_path
    )

    configuration = manifest[
        "configuration"
    ]

    print("=" * 70)
    print("REPRODUCIBILITY RERUN")
    print("=" * 70)

    print(
        f"Experiment: {experiment_id}"
    )

    print(
        f"Configuration: {configuration}"
    )

    workspace = prepare_workspace(
        experiment_id,
        configuration
    )

    print(
        f"Workspace: {workspace}"
    )

    started = time.time()

    try:

        pipeline_results = run_pipeline(
            workspace
        )

        status = "success"

    except Exception as exc:

        pipeline_results = []

        status = "failed"

        error = str(exc)

    elapsed = time.time() - started

    original_evaluation = (
        load_json(
            original_dir
            / "final_adaptive_evaluation.json"
        )
        if (
            original_dir
            / "final_adaptive_evaluation.json"
        ).exists()
        else None
    )

    reproduced_evaluation = (
        extract_evaluation(
            workspace
        )
    )

    original_metrics = (
        normalize_metrics(
            original_evaluation
        )
    )

    reproduced_metrics = (
        normalize_metrics(
            reproduced_evaluation
        )
    )

    metric_comparison = compare_metrics(
        original_metrics,
        reproduced_metrics
    )

    metrics_reproducible = all(
        item.get(
            "within_tolerance",
            item.get("equal", True)
        )
        for item in metric_comparison.values()
    )

    report = {
        "experiment_id": experiment_id,

        "status": status,

        "original_configuration": (
            configuration
        ),

        "reproduction_elapsed_seconds": (
            elapsed
        ),

        "pipeline_results": (
            pipeline_results
        ),

        "original_metrics": (
            original_metrics
        ),

        "reproduced_metrics": (
            reproduced_metrics
        ),

        "metric_comparison": (
            metric_comparison
        ),

        "metrics_reproducible": (
            metrics_reproducible
        ),
    }

    if status == "failed":
        report["error"] = error

    report_path = (
        REPRO_ROOT
        / f"{experiment_id}_reproducibility.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    if status == "failed":

        print("Pipeline rerun: FAILED")

    elif metrics_reproducible:

        print(
            "Pipeline rerun: SUCCESS"
        )

        print(
            "Metric reproducibility: PASS"
        )

    else:

        print(
            "Pipeline rerun: SUCCESS"
        )

        print(
            "Metric reproducibility: DRIFT"
        )

    print()
    print(
        f"Report: {report_path}"
    )


if __name__ == "__main__":
    main()