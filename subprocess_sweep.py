from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import subprocess
import sys
import time

from parallel_isolated_sweep import (
    prepare_workspace,
    load_configurations,
)


BASE_DIR = Path(__file__).resolve().parent

STATE_FILE = BASE_DIR / "subprocess_sweep_state.json"

MAX_WORKERS = 2
MAX_CONFIGURATIONS = 4

MAX_RETRIES = 2

EXPERIMENT_TIMEOUT_SECONDS = 900
BACKOFF_SECONDS = 10


PIPELINE = [
    "adaptive_rag.py",
    "final_adaptive_evaluation.py",
    "final_rag_report.py",
    "failure_monitor.py",
]


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


def run_script(workspace, script):
    return subprocess.run(
        [sys.executable, script],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=EXPERIMENT_TIMEOUT_SECONDS,
    )


def run_experiment_process(config):
    """
    One complete experiment runs inside its own workspace/process.
    """

    config_id = config["id"]

    start = time.time()

    try:
        workspace = prepare_workspace(config)

        logs = []

        for script in PIPELINE:

            print(
                f"[{config_id}] Running {script}"
            )

            result = run_script(
                workspace,
                script
            )

            logs.append({
                "script": script,
                "returncode": result.returncode,
                "stdout": result.stdout[-5000:],
                "stderr": result.stderr[-5000:],
            })

            if result.returncode != 0:

                return {
                    "id": config_id,
                    "status": "failed",
                    "configuration": config,
                    "failure_type": "pipeline_error",
                    "error": (
                        f"{script} exited with "
                        f"code {result.returncode}"
                    ),
                    "logs": logs,
                    "elapsed_seconds": (
                        time.time() - start
                    ),
                }

        return {
            "id": config_id,
            "status": "success",
            "configuration": config,
            "logs": logs,
            "elapsed_seconds": (
                time.time() - start
            ),
        }

    except subprocess.TimeoutExpired as exc:

        return {
            "id": config_id,
            "status": "failed",
            "configuration": config,
            "failure_type": "timeout",
            "error": (
                f"{exc.cmd} exceeded "
                f"{EXPERIMENT_TIMEOUT_SECONDS}s"
            ),
            "elapsed_seconds": (
                time.time() - start
            ),
        }

    except Exception as exc:

        return {
            "id": config_id,
            "status": "failed",
            "configuration": config,
            "failure_type": "exception",
            "error": str(exc),
            "elapsed_seconds": (
                time.time() - start
            ),
        }


def classify_failure(result):
    failure_type = result.get(
        "failure_type",
        "exception"
    )

    if failure_type in {
        "timeout",
        "connection_error",
        "server_error",
    }:
        return "transient"

    return "permanent"


def get_state(state, config_id):

    if config_id not in state["experiments"]:

        state["experiments"][config_id] = {
            "status": "pending",
            "attempts": 0,
            "history": [],
        }

    return state["experiments"][config_id]


def execute(config, state):

    config_id = config["id"]

    experiment = get_state(
        state,
        config_id
    )

    experiment["attempts"] += 1

    attempt = experiment["attempts"]

    experiment["status"] = "running"

    experiment["history"].append({
        "attempt": attempt,
        "status": "running",
        "started_at": time.time(),
    })

    # State mutation happens before worker launch.
    save_state(state)

    print(
        f"[START] {config_id} "
        f"attempt {attempt}/{MAX_RETRIES + 1}"
    )

    result = run_experiment_process(config)

    return {
        "config": config,
        "result": result,
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

    while True:

        pending = []

        for config in configurations:

            experiment = get_state(
                state,
                config["id"]
            )

            if experiment["status"] == "success":
                continue

            if experiment["status"] == "failed_permanent":
                continue

            if (
                experiment["attempts"]
                >= MAX_RETRIES + 1
            ):
                experiment["status"] = (
                    "failed_permanent"
                )
                continue

            pending.append(config)

        save_state(state)

        if not pending:
            break

        print()
        print(
            f"Launching {len(pending)} experiments"
        )

        with ThreadPoolExecutor(
            max_workers=MAX_WORKERS
        ) as executor:

            futures = {
                executor.submit(
                    execute,
                    config,
                    state
                ): config
                for config in pending
            }

            for future in as_completed(futures):

                config = futures[future]
                config_id = config["id"]

                try:
                    result_bundle = future.result()

                except Exception as exc:

                    result_bundle = {
                        "config": config,
                        "result": {
                            "id": config_id,
                            "status": "failed",
                            "failure_type": "exception",
                            "error": str(exc),
                        },
                    }

                result = result_bundle["result"]

                experiment = get_state(
                    state,
                    config_id
                )

                history = experiment["history"][-1]

                if result["status"] == "success":

                    experiment["status"] = "success"

                    experiment["result"] = result

                    history.update({
                        "status": "success",
                        "finished_at": time.time(),
                    })

                    print(
                        f"[SUCCESS] {config_id}"
                    )

                else:

                    failure_class = classify_failure(
                        result
                    )

                    history.update({
                        "status": "failed",
                        "failure_type": failure_class,
                        "error": result.get("error"),
                        "finished_at": time.time(),
                    })

                    if (
                        failure_class == "transient"
                        and experiment["attempts"]
                        < MAX_RETRIES + 1
                    ):

                        experiment["status"] = (
                            "retry_wait"
                        )

                        print(
                            f"[RETRY] {config_id}"
                        )

                    else:

                        experiment["status"] = (
                            "failed_permanent"
                        )

                        experiment["result"] = result

                        print(
                            f"[FAILED] {config_id}"
                        )

                save_state(state)

        retrying = any(
            item["status"] == "retry_wait"
            for item in state["experiments"].values()
        )

        if retrying:

            print(
                f"Waiting {BACKOFF_SECONDS}s..."
            )

            time.sleep(BACKOFF_SECONDS)

    save_state(state)

    print()
    print("=" * 70)
    print("SUBPROCESS SWEEP COMPLETE")
    print("=" * 70)

    for config_id, experiment in sorted(
        state["experiments"].items()
    ):
        print(
            f"{config_id}: "
            f"{experiment['status']} "
            f"attempts={experiment['attempts']}"
        )


if __name__ == "__main__":
    main()