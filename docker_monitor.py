import time
import requests

BASE_URL = "http://localhost:8000"


def get_json(path):
    response = requests.get(
        f"{BASE_URL}{path}",
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def print_status():
    health = get_json("/health")
    ready = get_json("/ready")
    metrics = get_json("/metrics")
    slo = get_json("/metrics/slo")

    print("=" * 60)
    print("RAG PRODUCTION STATUS")
    print("=" * 60)

    print("\nHealth:")
    print(health)

    print("\nReadiness:")
    print(ready)

    print("\nMetrics:")
    print(metrics)

    print("\nSLO:")
    print(slo)


def main():
    print_status()


if __name__ == "__main__":
    main()