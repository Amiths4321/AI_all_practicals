from pathlib import Path
import hashlib
import json
import sys


BASE_DIR = Path(__file__).resolve().parent


def sha256_file(path):

    digest = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def fingerprint_files(files):

    result = {}

    for filename in files:

        path = BASE_DIR / filename

        if path.exists():

            result[filename] = (
                sha256_file(path)
            )

    return result


def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def compare_dicts(
    expected,
    actual,
    label
):

    differences = []

    expected_keys = set(
        expected.keys()
    )

    actual_keys = set(
        actual.keys()
    )

    for key in sorted(
        expected_keys | actual_keys
    ):

        if key not in expected:

            differences.append(
                f"{label}: unexpected key {key}"
            )

        elif key not in actual:

            differences.append(
                f"{label}: missing key {key}"
            )

        elif expected[key] != actual[key]:

            differences.append(
                f"{label}: {key} changed"
            )

    return differences


def verify_manifest(
    manifest,
    current_configuration
):

    differences = []

    original_configuration = (
        manifest.get(
            "configuration",
            {}
        )
    )

    differences.extend(
        compare_dicts(
            original_configuration,
            current_configuration,
            "configuration"
        )
    )

    pipeline_files = list(
        manifest.get(
            "pipeline_fingerprint",
            {}
        ).keys()
    )

    input_files = list(
        manifest.get(
            "input_fingerprint",
            {}
        ).keys()
    )

    current_pipeline = (
        fingerprint_files(
            pipeline_files
        )
    )

    current_inputs = (
        fingerprint_files(
            input_files
        )
    )

    differences.extend(
        compare_dicts(
            manifest.get(
                "pipeline_fingerprint",
                {}
            ),
            current_pipeline,
            "pipeline"
        )
    )

    differences.extend(
        compare_dicts(
            manifest.get(
                "input_fingerprint",
                {}
            ),
            current_inputs,
            "input"
        )
    )

    return differences


def main():

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            "python verify_reproducibility.py "
            "config_001"
        )

        sys.exit(1)

    experiment_id = sys.argv[1]

    manifest_path = (
        BASE_DIR
        / "experiment_outputs"
        / experiment_id
        / "experiment_manifest.json"
    )

    if not manifest_path.exists():

        print(
            f"Manifest not found: "
            f"{manifest_path}"
        )

        sys.exit(1)

    manifest = load_json(
        manifest_path
    )

    configuration = manifest.get(
        "configuration",
        {}
    )

    differences = verify_manifest(
        manifest,
        configuration
    )

    print("=" * 70)
    print("REPRODUCIBILITY VERIFICATION")
    print("=" * 70)

    print(
        f"Experiment: {experiment_id}"
    )

    print()

    if differences:

        print("STATUS: DRIFT DETECTED")
        print()

        for difference in differences:
            print(
                f" - {difference}"
            )

        sys.exit(2)

    print(
        "STATUS: REPRODUCIBLE"
    )

    print(
        "Configuration, pipeline fingerprints, "
        "and input fingerprints match."
    )


if __name__ == "__main__":
    main()