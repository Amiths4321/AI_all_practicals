from pathlib import Path
import hashlib
import json
import time


BASE_DIR = Path(__file__).resolve().parent

OUTPUT_ROOT = BASE_DIR / "experiment_outputs"

REGISTRY_FILE = BASE_DIR / "experiment_registry.json"


def sha256_file(path):

    digest = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def load_json(path):

    try:

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception:

        return None


def extract_metrics(experiment_dir):

    metrics = {}

    evaluation_file = (
        experiment_dir
        / "final_adaptive_evaluation.json"
    )

    if evaluation_file.exists():

        evaluation = load_json(
            evaluation_file
        )

        if isinstance(evaluation, dict):

            metrics["evaluation"] = evaluation

        elif isinstance(evaluation, list):

            metrics["evaluation"] = {
                "records": evaluation
            }

    failure_file = (
        experiment_dir
        / "failure_monitoring.json"
    )

    if failure_file.exists():

        failure = load_json(
            failure_file
        )

        if failure is not None:
            metrics["failure_monitoring"] = failure

    return metrics


def calculate_summary(metrics):

    evaluation = metrics.get(
        "evaluation",
        {}
    )

    failure = metrics.get(
        "failure_monitoring",
        {}
    )

    summary = {}

    # Preserve the complete evaluation data,
    # while extracting common scalar metrics
    # when available.

    if isinstance(evaluation, dict):

        for key in [
            "evidence_recall",
            "quality",
            "average_latency_seconds",
            "average_context_characters",
            "success_rate",
        ]:

            if key in evaluation:

                summary[key] = evaluation[key]

    if isinstance(failure, dict):

        failure_summary = failure.get(
            "summary",
            {}
        )

        for key in [
            "success_rate",
            "failure_rate",
            "retrieval_failures",
            "ungrounded_answers",
            "abstentions",
        ]:

            if key in failure_summary:

                summary[
                    f"failure_{key}"
                ] = failure_summary[key]

    return summary


def build_record(experiment_dir):

    manifest_file = (
        experiment_dir
        / "experiment_manifest.json"
    )

    if not manifest_file.exists():
        return None

    manifest = load_json(
        manifest_file
    )

    if not isinstance(manifest, dict):
        return None

    experiment_id = manifest.get(
        "experiment_id",
        experiment_dir.name
    )

    metrics = extract_metrics(
        experiment_dir
    )

    record = {
        "experiment_id": experiment_id,

        "directory": str(
            experiment_dir.relative_to(BASE_DIR)
        ),

        "configuration": manifest.get(
            "configuration",
            {}
        ),

        "started_at": manifest.get(
            "started_at"
        ),

        "finished_at": manifest.get(
            "finished_at"
        ),

        "duration_seconds": manifest.get(
            "duration_seconds"
        ),

        "pipeline_fingerprint": manifest.get(
            "pipeline_fingerprint",
            {}
        ),

        "input_fingerprint": manifest.get(
            "input_fingerprint",
            {}
        ),

        "retry_history": manifest.get(
            "retry_history",
            []
        ),

        "summary": calculate_summary(
            metrics
        ),

        "metrics": metrics,
    }

    return record


def build_registry():

    experiments = []

    if OUTPUT_ROOT.exists():

        directories = sorted(
            path
            for path in OUTPUT_ROOT.iterdir()
            if path.is_dir()
        )

        for directory in directories:

            record = build_record(
                directory
            )

            if record is not None:

                experiments.append(record)

    registry = {
        "registry_version": 1,

        "generated_at": time.time(),

        "experiment_count": len(
            experiments
        ),

        "experiments": experiments,
    }

    return registry


def save_registry(registry):

    temporary = REGISTRY_FILE.with_suffix(
        ".tmp"
    )

    with open(
        temporary,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            registry,
            f,
            indent=2
        )

    temporary.replace(
        REGISTRY_FILE
    )


def main():

    registry = build_registry()

    save_registry(
        registry
    )

    print("=" * 70)
    print("EXPERIMENT REGISTRY")
    print("=" * 70)

    print(
        f"Experiments: "
        f"{registry['experiment_count']}"
    )

    print(
        f"Registry: "
        f"{REGISTRY_FILE}"
    )


if __name__ == "__main__":
    main()