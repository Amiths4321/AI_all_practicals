from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import time

from parallel_isolated_sweep import run_experiment, load_configurations


BASE_DIR = Path(__file__).resolve().parent
STATE_FILE = BASE_DIR / "robust_sweep_state.json"

MAX_WORKERS = 2
MAX_CONFIGURATIONS = 4
MAX_RETRIES = 2


def load_state():
    if not STATE_FILE.exists():
        return {
            "completed": {},
            "failed": {},
            "attempts": {},
            "started_at": time.time(),
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


def run_worker(config):
    """
    Worker does NOT modify shared state.
    It only executes the experiment and returns a result.
    """
    config_id = config["id"]

    start = time.time()

    try:
        result = run_experiment(config)

        return {
            "config_id": config_id,
            "result": result,
            "worker_elapsed_seconds": time.time() - start,
        }

    except Exception as exc:

        return {
            "config_id": config_id,
            "result": {
                "id": config_id,
                "status": "failed",
                "configuration": config,
                "error": str(exc),
            },
            "worker_elapsed_seconds": time.time() - start,
        }


def record_result(state, config, worker_result):

    config_id = config["id"]
    result = worker_result["result"]

    if result["status"] == "success":

        state["completed"][config_id] = result
        state["failed"].pop(config_id, None)

    else:

        state["failed"][config_id] = result


def should_run(config, state):

    config_id = config["id"]

    if config_id in state["completed"]:
        return False

    attempts = state["attempts"].get(config_id, 0)

    return attempts < MAX_RETRIES + 1


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
        configurations = configurations[:MAX_CONFIGURATIONS]

    state = load_state()

    pending = [
        config
        for config in configurations
        if should_run(config, state)
    ]

    print("=" * 70)
    print("ROBUST PARALLEL SWEEP")
    print("=" * 70)

    print(f"Total:     {len(configurations)}")
    print(f"Completed: {len(state['completed'])}")
    print(f"Failed:    {len(state['failed'])}")
    print(f"Pending:   {len(pending)}")
    print(f"Workers:   {MAX_WORKERS}")
    print()

    if not pending:
        print("Nothing to run.")
        return

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {}

        for config in pending:

            config_id = config["id"]

            attempts = state["attempts"].get(
                config_id,
                0
            )

            if attempts >= MAX_RETRIES + 1:
                continue

            state["attempts"][config_id] = attempts + 1

            # State is written by the main thread only.
            save_state(state)

            print(
                f"[SUBMIT] {config_id} "
                f"attempt {attempts + 1}/{MAX_RETRIES + 1}"
            )

            future = executor.submit(
                run_worker,
                config
            )

            futures[future] = config

        # SINGLE WRITER:
        # only this main thread modifies state.
        for future in as_completed(futures):

            config = futures[future]
            config_id = config["id"]

            worker_result = future.result()

            record_result(
                state,
                config,
                worker_result
            )

            save_state(state)

            result = worker_result["result"]

            print(
                f"[RESULT] {config_id}: "
                f"{result['status']}"
            )

    print()
    print("=" * 70)
    print("RUN COMPLETE")
    print("=" * 70)

    print(
        f"Completed: {len(state['completed'])}"
    )

    print(
        f"Failed:    {len(state['failed'])}"
    )

    print(
        f"Attempts:  {sum(state['attempts'].values())}"
    )

    print(
        f"State:     {STATE_FILE}"
    )


if __name__ == "__main__":
    main()