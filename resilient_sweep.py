from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from pathlib import Path
import json
import time

from parallel_isolated_sweep import run_experiment, load_configurations


BASE_DIR = Path(__file__).resolve().parent

STATE_FILE = BASE_DIR / "resilient_sweep_state.json"

MAX_WORKERS = 2
MAX_CONFIGURATIONS = 4

MAX_RETRIES = 2

EXPERIMENT_TIMEOUT_SECONDS = 900

BACKOFF_SECONDS = 10


def load_state():
    if not STATE_FILE.exists():
        return {
            "experiments": {},
            "created_at": time.time(),
            "updated_at": time.time(),
        }

    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state):
    state["updated_at"] = time.time()

    temporary = STATE_FILE.with_suffix(".tmp")

    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    temporary.replace(STATE_FILE)


def classify_failure(result):
    """
    Classify failures conservatively.

    Returns:
        transient
        permanent
    """

    error = str(result.get("error", "")).lower()

    transient_patterns = [
        "timeout",
        "timed out",
        "connection",
        "connectionerror",
        "temporarily",
        "503",
        "502",
        "504",
        "ollama",
    ]

    for pattern in transient_patterns:
        if pattern in error:
            return "transient"

    return "permanent"


def execute_experiment(config):

    config_id = config["id"]

    start = time.time()

    try:

        result = run_experiment(config)

        result["worker_elapsed_seconds"] = (
            time.time() - start
        )

        return result

    except Exception as exc:

        return {
            "id": config_id,
            "status": "failed",
            "configuration": config,
            "error": str(exc),
            "worker_elapsed_seconds": (
                time.time() - start
            ),
        }


def get_experiment_state(state, config_id):

    if config_id not in state["experiments"]:

        state["experiments"][config_id] = {
            "status": "pending",
            "attempts": 0,
            "history": [],
        }

    return state["experiments"][config_id]


def is_finished(experiment_state):

    return experiment_state["status"] in {
        "success",
        "failed_permanent",
    }


def main():

    configurations = load_configurations()

    # Safely extract the list whether the JSON is a dict or a list
    if isinstance(configurations, dict):
      config_list = configurations.get(
          "configurations", list(configurations.values())
      )
    elif isinstance(configurations, list):
      config_list = configurations
    else:
      config_list = []

    configurations = config_list[:MAX_CONFIGURATIONS]

    if MAX_CONFIGURATIONS is not None:
        configurations = configurations[
            :MAX_CONFIGURATIONS
        ]

    state = load_state()

    print("=" * 70)
    print("RESILIENT EXPERIMENT SWEEP")
    print("=" * 70)

    print(
        f"Configurations: {len(configurations)}"
    )

    print(
        f"Workers:        {MAX_WORKERS}"
    )

    print(
        f"Max retries:    {MAX_RETRIES}"
    )

    print(
        f"Timeout:        {EXPERIMENT_TIMEOUT_SECONDS}s"
    )

    print()

    while True:

        pending = []

        for config in configurations:

            experiment = get_experiment_state(
                state,
                config["id"]
            )

            if is_finished(experiment):
                continue

            if experiment["attempts"] >= MAX_RETRIES + 1:

                experiment["status"] = "failed_permanent"

                continue

            pending.append(config)

        save_state(state)

        if not pending:
            break

        print(
            f"Pending experiments: {len(pending)}"
        )

        futures = {}

        with ThreadPoolExecutor(
            max_workers=MAX_WORKERS
        ) as executor:

            for config in pending:

                config_id = config["id"]

                experiment = get_experiment_state(
                    state,
                    config_id
                )

                experiment["attempts"] += 1

                attempt = experiment["attempts"]

                experiment["status"] = "running"

                experiment["started_at"] = time.time()

                experiment["history"].append({
                    "attempt": attempt,
                    "status": "running",
                    "started_at": time.time(),
                })

                save_state(state)

                print(
                    f"[START] {config_id} "
                    f"attempt {attempt}/"
                    f"{MAX_RETRIES + 1}"
                )

                future = executor.submit(
                    execute_experiment,
                    config
                )

                futures[future] = config

            for future in as_completed(futures):

                config = futures[future]

                config_id = config["id"]

                experiment = get_experiment_state(
                    state,
                    config_id
                )

                try:

                    result = future.result(
                        timeout=EXPERIMENT_TIMEOUT_SECONDS
                    )

                except TimeoutError:

                    result = {
                        "id": config_id,
                        "status": "failed",
                        "configuration": config,
                        "error": (
                            "Experiment exceeded "
                            "timeout"
                        ),
                    }

                except Exception as exc:

                    result = {
                        "id": config_id,
                        "status": "failed",
                        "configuration": config,
                        "error": str(exc),
                    }

                if result["status"] == "success":

                    experiment["status"] = "success"

                    experiment["result"] = result

                    experiment["history"][-1].update({
                        "status": "success",
                        "finished_at": time.time(),
                    })

                    print(
                        f"[SUCCESS] {config_id}"
                    )

                else:

                    failure_type = classify_failure(
                        result
                    )

                    experiment["history"][-1].update({
                        "status": "failed",
                        "failure_type": failure_type,
                        "error": result.get("error"),
                        "finished_at": time.time(),
                    })

                    if (
                        failure_type == "transient"
                        and experiment["attempts"]
                        < MAX_RETRIES + 1
                    ):

                        experiment["status"] = (
                            "retry_wait"
                        )

                        print(
                            f"[RETRY] {config_id} "
                            f"after transient failure"
                        )

                    else:

                        experiment["status"] = (
                            "failed_permanent"
                        )

                        experiment["result"] = result

                        print(
                            f"[FAILED] {config_id} "
                            f"({failure_type})"
                        )

                save_state(state)

        retrying = [
            item
            for item in state["experiments"].values()
            if item["status"] == "retry_wait"
        ]

        if retrying:

            delay = BACKOFF_SECONDS

            print(
                f"Waiting {delay}s before retries..."
            )

            time.sleep(delay)

    save_state(state)

    success = sum(
        1
        for item in state["experiments"].values()
        if item["status"] == "success"
    )

    failed = sum(
        1
        for item in state["experiments"].values()
        if item["status"] == "failed_permanent"
    )

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(f"Success: {success}")
    print(f"Failed:  {failed}")

    print(
        f"State:   {STATE_FILE}"
    )


if __name__ == "__main__":
    main()