import itertools
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "sweep_configurations.json"


PARAMETERS = {
    "chunk_size": [300, 500, 700],
    "overlap": [50, 100, 150],
    "rerank_k": [5, 10, 15],
    "default_budget": [1, 2, 3, 4],
}


def generate_configurations():
    names = list(PARAMETERS.keys())
    values = list(PARAMETERS.values())

    configurations = []

    for index, combination in enumerate(
        itertools.product(*values),
        start=1
    ):
        config = dict(
            zip(names, combination)
        )

        config["id"] = f"config_{index:03d}"

        configurations.append(config)

    return configurations


def validate_configuration(config):
    chunk_size = config["chunk_size"]
    overlap = config["overlap"]

    if overlap >= chunk_size:
        return False

    if config["rerank_k"] <= 0:
        return False

    if config["default_budget"] <= 0:
        return False

    return True


def main():
    configurations = (
        generate_configurations()
    )

    valid = [
        config
        for config in configurations
        if validate_configuration(config)
    ]

    output = {
        "parameter_space": PARAMETERS,
        "total_configurations": len(valid),
        "configurations": valid
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            output,
            file,
            indent=2
        )

    print("=" * 70)
    print("SWEEP CONFIGURATION GENERATOR")
    print("=" * 70)

    print()
    print(
        "Parameter combinations:",
        len(configurations)
    )

    print(
        "Valid configurations:",
        len(valid)
    )

    print()
    print("Saved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()