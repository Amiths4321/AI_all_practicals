import time
import requests

BASE_URL = "http://localhost:8000"
INTERVAL = 30


def check_endpoint(path):
    try:
        response = requests.get(
            f"{BASE_URL}{path}",
            timeout=10,
        )

        return {
            "ok": response.status_code == 200,
            "status_code": response.status_code,
            "data": response.json(),
        }

    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
        }


def main():
    print("Starting production health monitor...")

    while True:
        health = check_endpoint("/health")
        ready = check_endpoint("/ready")

        timestamp = time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        print(
            f"[{timestamp}] "
            f"health={health['ok']} "
            f"ready={ready['ok']}"
        )

        if not health["ok"] or not ready["ok"]:
            print("WARNING: service unhealthy")

        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()