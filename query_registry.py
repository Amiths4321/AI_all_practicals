import json
from pathlib import Path


REGISTRY_FILE = (
    Path(__file__).resolve().parent
    / "experiment_registry.json"
)


def load_registry():

    with open(
        REGISTRY_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def main():

    registry = load_registry()

    experiments = registry["experiments"]

    print()
    print(
        f"{'ID':<15}"
        f"{'Chunk':<10}"
        f"{'Overlap':<10}"
        f"{'Rerank':<10}"
        f"{'Budget':<10}"
    )

    print("-" * 55)

    for experiment in experiments:

        config = experiment[
            "configuration"
        ]

        print(
            f"{experiment['experiment_id']:<15}"
            f"{config.get('chunk_size', '-'):<10}"
            f"{config.get('overlap', '-'):<10}"
            f"{config.get('rerank_k', '-'):<10}"
            f"{config.get('default_budget', '-'):<10}"
        )


if __name__ == "__main__":
    main()