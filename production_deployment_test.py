import sys
import time
import requests

BASE_URL = "http://localhost:8000"

TIMEOUT = 180
POLL_INTERVAL = 5


def wait_for_ready():
    deadline = time.time() + TIMEOUT

    while time.time() < deadline:
        try:
            response = requests.get(
                f"{BASE_URL}/ready",
                timeout=5,
            )

            if response.status_code == 200:
                data = response.json()

                if data.get("status") == "ready":
                    print("✓ ready")
                    return

        except requests.RequestException:
            pass

        print("Waiting for service...")

        time.sleep(POLL_INTERVAL)

    raise RuntimeError(
        "Service did not become ready"
    )


def check_health():
    response = requests.get(
        f"{BASE_URL}/health",
        timeout=10,
    )

    response.raise_for_status()

    data = response.json()

    assert data["status"] == "healthy"

    print("✓ health")


def check_metadata():
    response = requests.get(
        f"{BASE_URL}/metadata",
        timeout=10,
    )

    response.raise_for_status()

    data = response.json()

    required = [
        "service_version",
        "config_fingerprint",
        "index_collection",
        "index_fingerprint",
        "container_image",
        "deployment_environment",
    ]

    for field in required:
        assert data.get(field), (
            f"Missing metadata field: {field}"
        )

    print("✓ metadata")

    return data


def check_query():
    response = requests.post(
        f"{BASE_URL}/query",
        json={
            "question":
                "What is the annual leave policy?"
        },
        timeout=180,
    )

    response.raise_for_status()

    data = response.json()

    assert data.get("request_id")
    assert data.get("question")
    assert "answer" in data

    print("✓ query")


def main():
    print("=" * 60)
    print("PRODUCTION RAG DEPLOYMENT TEST")
    print("=" * 60)

    wait_for_ready()
    check_health()

    metadata = check_metadata()

    check_query()

    print("\nDeployment test PASSED")
    print(
        f"Image: {metadata['container_image']}"
    )
    print(
        f"Environment: "
        f"{metadata['deployment_environment']}"
    )
    print(
        f"Index: {metadata['index_collection']}"
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print("\nDeployment test FAILED")
        print(type(exc).__name__, str(exc))
        sys.exit(1)