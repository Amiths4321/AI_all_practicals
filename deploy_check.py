import subprocess
import sys


def main():
    print("Running deployment gate...")

    result = subprocess.run(
        [sys.executable, "smoke_test.py"],
        capture_output=True,
        text=True,
    )

    print(result.stdout)

    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        print("DEPLOYMENT GATE: FAILED")
        sys.exit(1)

    print("DEPLOYMENT GATE: PASSED")


if __name__ == "__main__":
    main()