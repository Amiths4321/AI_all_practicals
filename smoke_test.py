import sys
import requests

BASE_URL = "http://localhost:8000"


def check_health():
    response = requests.get(
        f"{BASE_URL}/health",
        timeout=10
    )
    response.raise_for_status()

    data = response.json()

    assert data["status"] == "healthy"

    return data


def check_ready():
    response = requests.get(
        f"{BASE_URL}/ready",
        timeout=10
    )
    response.raise_for_status()

    data = response.json()

    assert data["status"] == "ready"

    return data


def check_metadata():
    response = requests.get(
        f"{BASE_URL}/metadata",
        timeout=10
    )
    response.raise_for_status()

    data = response.json()

    assert data["service_version"]
    assert data["config_fingerprint"]
    assert data["index_collection"]
    assert data["index_fingerprint"]

    return data


def check_query():
    response = requests.post(
        f"{BASE_URL}/query",
        json={
            "question": "What is the annual leave policy?"
        },
        timeout=180
    )

    response.raise_for_status()

    data = response.json()

    assert data["request_id"]
    assert data["question"]
    assert "answer" in data

    return data


def main():
    print("Running production smoke test...")

    health = check_health()
    print("✓ health")

    ready = check_ready()
    print("✓ ready")

    metadata = check_metadata()
    print("✓ metadata")

    query = check_query()
    print("✓ query")

    print()
    print("Smoke test PASSED")
    print(f"Service: {metadata['service_version']}")
    print(f"Index: {metadata['index_collection']}")
    print(f"Request: {query['request_id']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("Smoke test FAILED")
        print(type(exc).__name__, str(exc))
        sys.exit(1)