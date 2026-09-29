import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
SWEEPS_DIR = BASE_DIR / "sweeps"


def find_latest_sweep():
    if not SWEEPS_DIR.exists():
        return None

    sweeps = [
        path
        for path in SWEEPS_DIR.iterdir()
        if path.is_dir()
    ]

    if not sweeps:
        return None

    return sorted(
        sweeps,
        key=lambda path: path.name,
        reverse=True
    )[0]


def load_results():
    sweep_dir = find_latest_sweep()

    if sweep_dir is None:
        raise FileNotFoundError(
            "No sweep directory found."
        )

    result_file = (
        sweep_dir /
        "sweep_results.json"
    )

    if not result_file.exists():
        raise FileNotFoundError(
            f"Missing {result_file}"
        )

    with open(
        result_file,
        "r",
        encoding="utf-8"
    ) as file:
        return sweep_dir, json.load(file)


def dominates(a, b):
    """
    True if configuration a dominates b.

    Higher is better:
        evidence_recall
        quality

    Lower is better:
        latency
        context_characters
    """

    metrics_a = a["metrics"]
    metrics_b = b["metrics"]

    no_worse = (
        metrics_a["evidence_recall"]
        >= metrics_b["evidence_recall"]
        and
        metrics_a["quality"]
        >= metrics_b["quality"]
        and
        metrics_a["latency"]
        <= metrics_b["latency"]
        and
        metrics_a["context_characters"]
        <= metrics_b["context_characters"]
    )

    strictly_better = (
        metrics_a["evidence_recall"]
        > metrics_b["evidence_recall"]
        or
        metrics_a["quality"]
        > metrics_b["quality"]
        or
        metrics_a["latency"]
        < metrics_b["latency"]
        or
        metrics_a["context_characters"]
        < metrics_b["context_characters"]
    )

    return no_worse and strictly_better


def find_pareto_frontier(results):
    frontier = []

    for candidate in results:
        dominated = False

        for other in results:
            if candidate is other:
                continue

            if dominates(other, candidate):
                dominated = True
                break

        if not dominated:
            frontier.append(candidate)

    return frontier


def print_results(results):
    print()
    print("=" * 100)
    print("SWEEP RESULTS")
    print("=" * 100)

    header = (
        f"{'Configuration':<20}"
        f"{'Evidence':>12}"
        f"{'Quality':>12}"
        f"{'Latency':>12}"
        f"{'Context':>12}"
    )

    print(header)
    print("-" * 100)

    for result in results:
        config = result["configuration"]
        metrics = result.get("metrics")

        if not metrics:
            continue

        print(
            f"{config['name']:<20}"
            f"{metrics['evidence_recall']:>12.3f}"
            f"{metrics['quality']:>12.3f}"
            f"{metrics['latency']:>12.2f}"
            f"{metrics['context_characters']:>12.0f}"
        )


def main():
    sweep_dir, data = load_results()

    results = [
        result
        for result in data.get(
            "results",
            []
        )
        if result.get("status") == "completed"
        and result.get("metrics")
    ]

    if not results:
        print("No completed sweep results.")
        return 1

    print_results(results)

    frontier = find_pareto_frontier(
        results
    )

    print()
    print("=" * 100)
    print("PARETO-EFFICIENT CONFIGURATIONS")
    print("=" * 100)

    for result in frontier:
        config = result["configuration"]
        metrics = result["metrics"]

        print()
        print(
            f"Configuration: {config['name']}"
        )

        print(
            f"  chunk_size:     "
            f"{config['chunk_size']}"
        )

        print(
            f"  overlap:        "
            f"{config['overlap']}"
        )

        print(
            f"  rerank_k:       "
            f"{config['rerank_k']}"
        )

        print(
            f"  default_budget: "
            f"{config['default_budget']}"
        )

        print(
            f"  evidence:       "
            f"{metrics['evidence_recall']:.3f}"
        )

        print(
            f"  quality:        "
            f"{metrics['quality']:.3f}"
        )

        print(
            f"  latency:        "
            f"{metrics['latency']:.2f}s"
        )

        print(
            f"  context:        "
            f"{metrics['context_characters']:.0f}"
        )

    output = {
        "source_sweep": sweep_dir.name,
        "completed_configurations": len(
            results
        ),
        "pareto_frontier": [
            {
                "configuration":
                    result["configuration"],
                "metrics":
                    result["metrics"]
            }
            for result in frontier
        ]
    }

    output_file = (
        sweep_dir /
        "sweep_analysis.json"
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
    print(
        "Saved:",
        output_file
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())