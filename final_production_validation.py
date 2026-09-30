import json
import subprocess
import sys
import time

import requests


BASE_URL = "http://localhost:8000"
CONTAINER = "production-rag"


def run(command):
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
    )


def get(path, timeout=20):
    response = requests.get(
        f"{BASE_URL}{path}",
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def post(path, payload, timeout=180):
    response = requests.post(
        f"{BASE_URL}{path}",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def check_container():
    result = run([
        "docker",
        "inspect",
        CONTAINER,
        "--format",
        "{{.State.Status}}",
    ])

    if result.returncode != 0:
        raise RuntimeError(
            "Production container does not exist"
        )

    status = result.stdout.strip()

    assert status == "running", (
        f"Container status: {status}"
    )

    print("✓ container running")


def check_health():
    data = get("/health")

    assert data["status"] == "healthy"

    print("✓ health")


def check_ready():
    data = get("/ready")

    assert data["status"] == "ready"

    print("✓ readiness")


def check_metadata():
    data = get("/metadata")

    required = [
        "service_version",
        "config_fingerprint",
        "index_collection",
        "index_fingerprint",
        "index_chunk_count",
        "container_image",
        "deployment_environment",
    ]

    for field in required:
        assert data.get(field), (
            f"Missing metadata: {field}"
        )

    print("✓ deployment metadata")

    return data


def check_metrics():
    data = get("/metrics")

    assert isinstance(data, dict)

    print("✓ operational metrics")


def check_slo():
    data = get("/metrics/slo")

    assert isinstance(data, dict)

    print("✓ SLO metrics")

    return data


def check_query():
    data = post(
        "/query",
        {
            "question":
                "What is the annual leave policy?"
        },
    )

    assert data.get("request_id")
    assert data.get("question")
    assert "answer" in data

    print("✓ RAG query")

    return data


def check_manifest():
    result = run([
        "docker",
        "exec",
        CONTAINER,
        "cat",
        "/app/deployment_manifest.json",
    ])

    if result.returncode != 0:
        raise RuntimeError(
            "Deployment manifest unavailable"
        )

    manifest = json.loads(result.stdout)

    required = [
        "deployment_time",
        "service_version",
        "container_image",
        "deployment_environment",
        "config_fingerprint",
        "index_collection",
        "index_fingerprint",
        "index_chunk_count",
    ]

    for field in required:
        assert manifest.get(field) is not None, (
            f"Manifest missing: {field}"
        )

    print("✓ deployment manifest")

    return manifest


def check_docker_health():
    result = run([
        "docker",
        "inspect",
        CONTAINER,
        "--format",
        "{{.State.Health.Status}}",
    ])

    if result.returncode != 0:
        raise RuntimeError(
            "Unable to inspect Docker health"
        )

    health = result.stdout.strip()

    assert health == "healthy", (
        f"Docker health: {health}"
    )

    print("✓ Docker healthcheck")


def main():
    print("=" * 70)
    print("FINAL PRODUCTION RAG VALIDATION")
    print("=" * 70)

    check_container()
    check_docker_health()

    check_health()
    check_ready()

    metadata = check_metadata()
    check_manifest()

    check_metrics()
    slo = check_slo()

    query = check_query()

    print("\n" + "=" * 70)
    print("FINAL VALIDATION PASSED")
    print("=" * 70)

    print(
        f"\nService version: "
        f"{metadata['service_version']}"
    )

    print(
        f"Container image: "
        f"{metadata['container_image']}"
    )

    print(
        f"Index: "
        f"{metadata['index_collection']}"
    )

    print(
        f"Request ID: "
        f"{query['request_id']}"
    )

    print(
        f"SLO status: "
        f"{slo.get('healthy', 'available')}"
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print("\n" + "=" * 70)
        print("FINAL VALIDATION FAILED")
        print("=" * 70)

        print(
            type(exc).__name__,
            str(exc),
        )

        sys.exit(1)