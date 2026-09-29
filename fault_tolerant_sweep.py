from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import time

from parallel_isolated_sweep import run_experiment, load_configurations


BASE_DIR = Path(__file__).resolve().parent

STATE_FILE = BASE_DIR / "sweep_state.json"

MAX_WORKERS = 2
MAX_CONFIGURATIONS = 4

MAX_RETRIES = 2


def load_state():
    if not STATE_FILE.exists():
        return {
            "completed": {},
            "failed": {},
            "attempts": {}
        }

    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state):
    temporary = STATE_FILE.with_suffix(".tmp")

    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(
            state,
            f,
            indent=2
        )

    temporary.replace(STATE_FILE)


def should_run(config_id, state):
    if config_id in state["completed"]:
        return False

    attempts = state["attempts"].get(config_id, 0)

    if attempts >= MAX_RETRIES + 1:
        return False

    return True


def execute_with_retry(config, state):
    config_id = config["id"]

    attempt = state["attempts"].get(config_id, 0) + 1
    state["attempts"][config_id] = attempt
    save_state(state)

    print(
        f"[{config_id}] "
        f"attempt {attempt}/{MAX_RETRIES + 1}"
    )

    result = run_experiment(config)

    if result["status"] == "success":
        state["completed"][config_id] = result

        state["failed"].pop(config_id, None)

    else:
        state["failed"][config_id] = result

    save_state(state)

    return result


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

    state = load_state()

    pending = [
        config
        for config in configurations
        if should_run(config["id"], state)
    ]

    print("=" * 70)
    print("FAULT-TOLERANT EXPERIMENT SWEEP")
    print("=" * 70)

    print(f"Total configurations: {len(configurations)}")
    print(f"Already completed:    {len(state['completed'])}")
    print(f"Pending:              {len(pending)}")
    print(f"Workers:              {MAX_WORKERS}")
    print(f"Max retries:          {MAX_RETRIES}")
    print()

    if not pending:
        print("Nothing to run.")
        return

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {
            executor.submit(
                execute_with_retry,
                config,
                state
            ): config["id"]
            for config in pending
        }

        for future in as_completed(futures):

            config_id = futures[future]

            try:
                result = future.result()

                print(
                    f"[FINISHED] {config_id}: "
                    f"{result['status']}"
                )

            except Exception as exc:

                print(
                    f"[ORCHESTRATOR ERROR] "
                    f"{config_id}: {exc}"
                )

    save_state(state)

    print()
    print("=" * 70)
    print("SWEEP STATE")
    print("=" * 70)

    print(
        f"Completed: {len(state['completed'])}"
    )

    print(
        f"Failed:    {len(state['failed'])}"
    )

    print(
        f"State:     {STATE_FILE}"
    )


if __name__ == "__main__":
    main()