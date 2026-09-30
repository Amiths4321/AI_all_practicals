import subprocess
import sys
import time
import requests

BASE_URL = "http://localhost:8000"
MAX_WAIT_SECONDS = 120


def run(command):
    print(f"\n> {' '.join(command)}")

    result = subprocess.run(
        command,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(command)}"
        )


def wait_for_health():
    print("\nWaiting for RAG service...")

    deadline = time.time() + MAX_WAIT_SECONDS

    while time.time() < deadline:
        try:
            response = requests.get(
                f"{BASE_URL}/health",
                timeout=5,
            )

            if response.status_code == 200:
                data = response.json()

                if data.get("status") == "healthy":
                    print("✓ Service is healthy")
                    return

        except requests.RequestException:
            pass

        time.sleep(3)

    raise RuntimeError(
        f"Service did not become healthy within "
        f"{MAX_WAIT_SECONDS} seconds"
    )


def run_smoke_test():
    print("\nRunning deployment smoke test...")

    result = subprocess.run(
        [sys.executable, "docker_smoke_test.py"],
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError("Smoke test failed")

    print("✓ Smoke test passed")


def show_logs():
    print("\nRecent container logs:")

    subprocess.run(
        [
            "docker",
            "compose",
            "logs",
            "--tail",
            "50",
            "rag",
        ]
    )


def main():
    print("=" * 60)
    print("RAG DOCKER DEPLOYMENT")
    print("=" * 60)

    try:
        run([
            "docker",
            "compose",
            "up",
            "--build",
            "-d",
        ])

        wait_for_health()

        run_smoke_test()

    except Exception as exc:
        print("\nDEPLOYMENT GATE: FAILED")
        print(type(exc).__name__, str(exc))

        show_logs()

        sys.exit(1)

    print("\n" + "=" * 60)
    print("DEPLOYMENT GATE: PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()