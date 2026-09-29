import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from config import load_config
import hashlib


config = load_config()

BASE_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    "adaptive_rag.py",
    "final_adaptive_evaluation.py",
    "final_rag_report.py",
]


def configuration_fingerprint(config):
    """Return a short, stable hash of the current configuration."""
    serialized = json.dumps(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()[:12]


def update_experiment_registry():
    print()
    print("=" * 70)
    print("UPDATING EXPERIMENT REGISTRY")
    print("=" * 70)

    registry_result = subprocess.run(
        [sys.executable, "experiment_registry.py"],
        cwd=BASE_DIR
    )

    if registry_result.returncode != 0:
        print("WARNING: experiment registry update failed")


def run_validation():
    """Run the validation gate script. Returns True if it passes."""
    print()
    print("=" * 70)
    print("RUNNING VALIDATION GATE")
    print("=" * 70)

    validation_result = subprocess.run(
        [sys.executable, "validation_gate.py"],
        cwd=BASE_DIR
    )

    if validation_result.returncode != 0:
        print()
        print("=" * 70)
        print("VALIDATION GATE FAILED")
        print("=" * 70)
        return False

    return True


def run_quality_gate():
    """Run the quality gate script. Returns True if it passes."""
    print()
    print("=" * 70)
    print("RUNNING QUALITY GATE")
    print("=" * 70)

    quality_result = subprocess.run(
        [sys.executable, "quality_gate.py"],
        cwd=BASE_DIR
    )

    if quality_result.returncode != 0:
        print()
        print("=" * 70)
        print("QUALITY GATE FAILED")
        print("EXPERIMENT WILL NOT BE MARKED VALID")
        print("=" * 70)
        return False

    return True


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

    # Run the validation gate
    validation_passed = run_validation()

    if not validation_passed:
        print()
        print("=" * 70)
        print("EXPERIMENT HALTED: VALIDATION FAILED")
        print("=" * 70)
        return 1

    # Run the quality gate
    quality_passed = run_quality_gate()

    if not quality_passed:
        return 1

    # Keep the registry up to date
    update_experiment_registry()

    # Save a small manifest describing exactly what was executed.
    manifest = experiment_dir / "experiment_manifest.json"

    manifest_data = {
        "experiment": {
            "timestamp": timestamp,
            "configuration_fingerprint": configuration_fingerprint(config),
            "directory": str(experiment_dir)
        },

        "scripts": SCRIPTS,

        "input_files": [
            "config.json",
            "evaluation_data.json",
            "empirical_context_policy.json",
            "documents/leave_policy.txt",
            "documents/employee_handbook.txt"
        ],

        "configuration": config
    }

    with open(manifest, "w", encoding="utf-8") as file:
        json.dump(manifest_data, file, indent=2, ensure_ascii=False)

    print(f"Manifest saved: {manifest}")

    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    print(f"Artifacts saved to:\n{experiment_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())