import subprocess
import sys
import requests


BASE_URL = "http://localhost:8000"


def run_smoke_test():
    result = subprocess.run(
        [sys.executable, "smoke_test.py"],
        capture_output=True,
        text=True,
    )

    print(result.stdout)

    if result.stderr:
        print(result.stderr)

    return result.returncode == 0


def rollback():
    print("Deployment failed.")
    print("Attempting automatic index rollback...")

    response = requests.post(
        f"{BASE_URL}/admin/rollback",
        timeout=30,
    )

    response.raise_for_status()

    print("Rollback successful:")
    print(response.json())


def main():
    print("Running deployment validation...")

    if run_smoke_test():
        print("DEPLOYMENT PASSED")
        return

    print("DEPLOYMENT FAILED")

    try:
        rollback()
    except Exception as exc:
        print("ROLLBACK FAILED")
        print(type(exc).__name__, str(exc))
        sys.exit(2)

    print("Deployment rolled back successfully.")
    sys.exit(1)


if __name__ == "__main__":
    main()