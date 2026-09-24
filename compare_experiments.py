import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = BASE_DIR / "experiments"
OUTPUT_FILE = BASE_DIR / "experiment_comparison.json"


def load_report(experiment_dir):
    path = experiment_dir / "final_adaptive_evaluation.json"

    if not path.exists():
        return None

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def extract_metrics(report):
    if not report:
        return None

    summary = report.get(
        "summary",
        {}
    )

    return {
        "questions": summary.get(
            "questions",
            0
        ),

        "context_reduction": summary.get(
            "context_reduction",
            0.0
        ),

        "fixed_context": summary.get(
            "fixed_average_context_characters",
            0.0
        ),

        "adaptive_context": summary.get(
            "adaptive_average_context_characters",
            0.0
        ),

        "fixed_evidence_recall": summary.get(
            "fixed_average_evidence_recall",
            0.0
        ),

        "adaptive_evidence_recall": summary.get(
            "adaptive_average_evidence_recall",
            0.0
        ),

        "fixed_quality": summary.get(
            "fixed_average_quality_index",
            0.0
        ),

        "adaptive_quality": summary.get(
            "adaptive_average_quality_index",
            0.0
        ),

        "quality_delta": summary.get(
            "average_quality_delta",
            0.0
        ),

        "evidence_delta": summary.get(
            "average_evidence_delta",
            0.0
        ),

        "latency_delta": summary.get(
            "average_latency_delta_seconds",
            0.0
        )
    }


def main():

    if not EXPERIMENTS_DIR.exists():
        print(
            "No experiments directory found."
        )
        return

    experiment_results = []

    directories = sorted(
        [
            path
            for path in EXPERIMENTS_DIR.iterdir()
            if path.is_dir()
        ]
    )

    for directory in directories:

        report = load_report(
            directory
        )

        if report is None:
            print(
                f"Skipping {directory.name}: "
                f"final_adaptive_evaluation.json not found"
            )
            continue

        metrics = extract_metrics(
            report
        )

        if metrics is None:
            continue

        experiment_results.append({
            "experiment": directory.name,
            **metrics
        })

    if not experiment_results:
        print(
            "No completed experiments found."
        )
        return

    comparison = {
        "experiments": experiment_results
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            comparison,
            file,
            indent=2
        )

    print()
    print("=" * 80)
    print("EXPERIMENT COMPARISON")
    print("=" * 80)

    for item in experiment_results:

        print()
        print(
            f"Experiment: {item['experiment']}"
        )

        print(
            f"  Questions: "
            f"{item['questions']}"
        )

        print(
            f"  Context reduction: "
            f"{item['context_reduction'] * 100:.1f}%"
        )

        print(
            f"  Adaptive evidence recall: "
            f"{item['adaptive_evidence_recall']:.3f}"
        )

        print(
            f"  Adaptive quality: "
            f"{item['adaptive_quality']:.3f}"
        )

        print(
            f"  Quality delta: "
            f"{item['quality_delta']:+.3f}"
        )

        print(
            f"  Evidence delta: "
            f"{item['evidence_delta']:+.3f}"
        )

        print(
            f"  Latency delta: "
            f"{item['latency_delta']:+.3f}s"
        )

    print()
    print(
        f"Saved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()