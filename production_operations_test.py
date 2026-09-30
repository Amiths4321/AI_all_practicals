import sys
import subprocess
import requests

BASE_URL = "http://localhost:8000"


def check(path):
    response = requests.get(
        f"{BASE_URL}{path}",
        timeout=100,
    )

    response.raise_for_status()

    return response.json()


def main():
    print("=" * 60)
    print("PRODUCTION OPERATIONS TEST")
    print("=" * 60)

    health = check("/health")
    print("✓ health")

    ready = check("/ready")
    print("✓ ready")

    metrics = check("/metrics")
    print("✓ metrics")

    slo = check("/metrics/slo")
    print("✓ SLO")

    metadata = check("/metadata")
    print("✓ metadata")

    assert health["status"] == "healthy"
    assert ready["status"] == "ready"

    assert metadata["container_image"]

    print("\nRunning Docker status...")

    result = subprocess.run(
        [
            "docker",
            "compose",
            "ps",
        ],
        capture_output=True,
        text=True,
    )

    print(result.stdout)

    if result.returncode != 0:
        raise RuntimeError(
            "Docker Compose status failed"
        )

    print("=" * 60)
    print("PRODUCTION OPERATIONS TEST: PASSED")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print("\nPRODUCTION OPERATIONS TEST: FAILED")
        print(type(exc).__name__, str(exc))
        sys.exit(1)