import os
import subprocess
import sys
import requests


BASE_URL = "http://localhost:8000"


def run(command):
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
    )


def wait_for_health():
    import time

    deadline = time.time() + 120

    while time.time() < deadline:
        try:
            response = requests.get(
                f"{BASE_URL}/health",
                timeout=5,
            )

            if response.status_code == 200:
                if response.json().get("status") == "healthy":
                    return True

        except requests.RequestException:
            pass

        time.sleep(5)

    return False


def deploy(image):
    print(f"Deploying {image}")

    os.environ["RAG_IMAGE"] = image

    result = run([
        "docker",
        "compose",
        "up",
        "-d",
    ])

    print(result.stdout)

    if result.returncode != 0:
        print(result.stderr)
        return False

    return wait_for_health()


def main():
    if len(sys.argv) != 2:
        print(
            "Usage: python docker_rollback.py "
            "production-rag:2.2"
        )
        sys.exit(1)

    image = sys.argv[1]

    print("=" * 60)
    print("DOCKER RELEASE ROLLBACK")
    print("=" * 60)

    if deploy(image):
        print(f"\n✓ Rollback successful: {image}")
        return

    print("\n✗ Rollback failed")
    sys.exit(1)


if __name__ == "__main__":
    main()