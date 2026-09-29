import json
import shutil
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

PACKAGE_ROOT = BASE_DIR / "deployment_package"


FILES = [
    "production_controller.py",
    "rag_service.py",
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
    "config.json",

    "experiment_logger.py",
    "experiment_manifest.py",

    "production_evaluation.py",
    "production_monitor.py",
    "drift_detector.py",

    "service_health.py",

    "regression_tests.py",
    "validate_pipeline.py",
    "quality_gate.py",
    "final_validation.py",

    "version_manager.py",
]


DIRECTORIES = [
    "documents",
]


OPTIONAL_ARTIFACTS = [
    "production_health.json",
    "production_evaluation.json",
    "drift_report.json",
    "final_validation_report.json",
    "version_record.json",
]


def clean_package():
    if PACKAGE_ROOT.exists():
        shutil.rmtree(PACKAGE_ROOT)

    PACKAGE_ROOT.mkdir(parents=True)


def copy_files():
    copied = []

    for filename in FILES:

        source = BASE_DIR / filename

        if not source.exists():
            print(
                f"[SKIP] {filename}"
            )
            continue

        destination = (
            PACKAGE_ROOT / filename
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            source,
            destination,
        )

        copied.append(filename)

    return copied


def copy_directories():
    copied = []

    for directory in DIRECTORIES:

        source = BASE_DIR / directory

        if not source.exists():
            continue

        destination = (
            PACKAGE_ROOT / directory
        )

        shutil.copytree(
            source,
            destination,
        )

        copied.append(directory)

    return copied


def copy_optional_artifacts():

    copied = []

    for filename in OPTIONAL_ARTIFACTS:

        source = BASE_DIR / filename

        if not source.exists():
            continue

        destination = (
            PACKAGE_ROOT / filename
        )

        shutil.copy2(
            source,
            destination,
        )

        copied.append(filename)

    return copied


def create_package_manifest(
    files,
    directories,
    artifacts,
):

    manifest = {
        "package_version": 1,

        "created_at":
            datetime.now().isoformat(),

        "package_name":
            "production_rag",

        "files": files,

        "directories": directories,

        "artifacts": artifacts,

        "entry_point":
            "production_controller.py",

        "health_check":
            "service_health.py",

        "validation":
            "final_validation.py",
    }

    with open(
        PACKAGE_ROOT /
        "deployment_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
        )


def main():

    print(
        "Building deployment package..."
    )

    clean_package()

    files = copy_files()

    directories = copy_directories()

    artifacts = copy_optional_artifacts()

    create_package_manifest(
        files,
        directories,
        artifacts,
    )

    print()
    print(
        "Deployment package created."
    )

    print(
        f"Location: {PACKAGE_ROOT}"
    )

    print(
        f"Files: {len(files)}"
    )

    print(
        f"Directories: {len(directories)}"
    )

    print(
        f"Artifacts: {len(artifacts)}"
    )


if __name__ == "__main__":
    main()