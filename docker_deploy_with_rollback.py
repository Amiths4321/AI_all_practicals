import subprocess
import sys
import time

import requests


BASE_URL = "http://localhost:8000"
HEALTH_URL = f"{BASE_URL}/health"

STARTUP_TIMEOUT = 180
POLL_INTERVAL = 5


def run(command):
    print(f"> {' '.join(command)}")

    return subprocess.run(
        command,
        capture_output=True,
        text=True,
    )


def start_deployment():
    result = run([
        "docker",
        "compose",
        "up",
        "--build",
        "-d",
    ])

    print(result.stdout)

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Docker deployment failed")


def wait_for_health():
    deadline = time.time() + STARTUP_TIMEOUT

    while time.time() < deadline:
        try:
            response = requests.get(
                HEALTH_URL,
                timeout=5,
            )

            if response.status_code == 200:
                data = response.json()

                if data.get("status") == "healthy":
                    return True

        except requests.RequestException:
            pass

        time.sleep(POLL_INTERVAL)

    return False


def run_smoke_test():
    result = run([
        sys.executable,
        "docker_smoke_test.py",
    ])

    print(result.stdout)

    if result.stderr:
        print(result.stderr)

    return result.returncode == 0


def rollback():
    print("\nRolling back Docker deployment...")

    result = run([
        "docker",
        "compose",
        "down",
    ])

    print(result.stdout)

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Docker rollback failed")

    print("✓ Docker deployment stopped")


def main():
    print("=" * 60)
    print("DOCKER DEPLOYMENT WITH ROLLBACK")
    print("=" * 60)

    try:
        print("\nStarting deployment...")
        start_deployment()

        print("\nWaiting for health...")
        if not wait_for_health():
            raise RuntimeError(
                "Service failed to become healthy"
            )

        print("✓ Service healthy")

        print("\nRunning smoke test...")
        if not run_smoke_test():
            raise RuntimeError(
                "Docker smoke test failed"
            )

        print("\n" + "=" * 60)
        print("DEPLOYMENT PASSED")
        print("=" * 60)

    except Exception as exc:
        print("\n" + "=" * 60)
        print("DEPLOYMENT FAILED")
        print("=" * 60)

        print(type(exc).__name__, str(exc))

        try:
            rollback()
        except Exception as rollback_error:
            print("\nROLLBACK FAILED")
            print(
                type(rollback_error).__name__,
                str(rollback_error),
            )
            sys.exit(2)

        print("\nRollback completed.")
        sys.exit(1)


if __name__ == "__main__":
    main()