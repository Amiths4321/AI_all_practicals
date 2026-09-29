import json
from pathlib import Path


HEALTH_FILE = Path(
    "production_health.json"
)


def get_health():

    if not HEALTH_FILE.exists():
        return {
            "status": "unknown",
            "reason": (
                "Health report not available."
            ),
        }

    with open(
        HEALTH_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        report = json.load(f)

    return {
        "status": report.get(
            "status",
            "unknown",
        ),
        "timestamp": report.get(
            "timestamp"
        ),
        "quality": report.get(
            "quality",
            {},
        ),
    }


if __name__ == "__main__":
    print(
        json.dumps(
            get_health(),
            indent=2,
        )
    )