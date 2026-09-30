import subprocess
import sys
import time
import requests


BASE_URL = "http://localhost:8000"


def wait_for_service(timeout=120):
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            response = requests.get(
                f"{BASE_URL}/health",
                timeout=5,
            )

            if response.status_code == 200:
                data = response.json()

                if data.get("status") == "healthy":
                    return True

        except requests.RequestException:
            pass

        time.sleep(5)

    return False


def main():
    print("=" * 60)
    print("CONTAINER RECOVERY TEST")
    print("=" * 60)

    print("\nKilling container...")
    result = subprocess.run(
        ["docker", "kill", "production-rag"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(result.stderr)
        sys.exit(1)

    print("✓ Container stopped")

    print("\nWaiting for Docker recovery...")

    if not wait_for_service():
        print("RECOVERY TEST: FAILED")
        sys.exit(1)

    print("✓ Container recovered")
    print("✓ API healthy")

    print("\nRECOVERY TEST: PASSED")


if __name__ == "__main__":
    main()