from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import time

from parallel_isolated_sweep import (
    prepare_workspace,
    load_configurations,
)


BASE_DIR = Path(__file__).resolve().parent

STATE_FILE = BASE_DIR / "crash_safe_sweep_state.json"
WORKSPACE_ROOT = BASE_DIR / "experiment_workspaces"

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
        json.dump(
            state,
            f,
            indent=2
        )

    temporary.replace(STATE_FILE)


def get_state(state, config_id):

    if config_id not in state["experiments"]:

        state["experiments"][config_id] = {
            "status": "pending",
            "attempts": 0,
            "history": [],
        }

    return state["experiments"][config_id]


def cleanup_workspace(config_id):

    workspace = (
        WORKSPACE_ROOT / config_id
    )

    if workspace.exists():

        try:
            shutil.rmtree(workspace)

            print(
                f"[CLEANUP] Removed {workspace}"
            )

        except Exception as exc:

            print(
                f"[CLEANUP WARNING] "
                f"{config_id}: {exc}"
            )


def reconcile_running_jobs(state):

    """
    Any job marked RUNNING when the orchestrator
    starts again is assumed to have been interrupted.
    """

    changed = False

    for config_id, experiment in (
        state["experiments"].items()
    ):

        if experiment["status"] != "running":
            continue

        print(
            f"[RECOVER] {config_id} "
            f"was RUNNING during previous execution"
        )

        experiment["status"] = "retry_wait"

        if experiment["history"]:

            experiment["history"][-1].update({
                "status": "interrupted",
                "finished_at": time.time(),
            })

        cleanup_workspace(config_id)

        changed = True

    if changed:
        save_state(state)


def run_process(config):

    config_id = config["id"]

    workspace = prepare_workspace(config)

    start = time.time()

    logs = []

    process = None

    try:

        for script in PIPELINE:

            print(
                f"[{config_id}] "
                f"running {script}"
            )

            process = subprocess.Popen(
                [sys.executable, script],
                cwd=workspace,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            try:

                stdout, stderr = (
                    process.communicate(
                        timeout=EXPERIMENT_TIMEOUT_SECONDS
                    )
                )

            except subprocess.TimeoutExpired:

                print(
                    f"[TIMEOUT] {config_id} "
                    f"{script}"
                )

                process.kill()

                stdout, stderr = (
                    process.communicate()
                )

                return {
                    "id": config_id,
                    "status": "failed",
                    "failure_type": "timeout",
                    "error": (
                        f"{script} exceeded "
                        f"{EXPERIMENT_TIMEOUT_SECONDS}s"
                    ),
                    "logs": logs,
                    "elapsed_seconds": (
                        time.time() - start
                    ),
                }

            logs.append({
                "script": script,
                "returncode": process.returncode,
                "stdout": stdout[-5000:],
                "stderr": stderr[-5000:],
            })

            if process.returncode != 0:

                return {
                    "id": config_id,
                    "status": "failed",
                    "failure_type": "pipeline_error",
                    "error": (
                        f"{script} exited with "
                        f"code {process.returncode}"
                    ),
                    "logs": logs,
                    "elapsed_seconds": (
                        time.time() - start
                    ),
                }

        return {
            "id": config_id,
            "status": "success",
            "logs": logs,
            "elapsed_seconds": (
                time.time() - start
            ),
        }

    finally:

        process = None


def classify_failure(result):

    if result.get("failure_type") in {
        "timeout",
        "connection_error",
        "server_error",
    }:
        return "transient"

    return "permanent"


def worker(config):

    """
    Worker performs the expensive operation.
    It does not modify shared state.
    """

    result = run_process(config)

    return {
        "config_id": config["id"],
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

    # Recover jobs from a previous crash.
    reconcile_running_jobs(state)

    while True:

        pending = []

        for config in configurations:

            experiment = get_state(
                state,
                config["id"]
            )

            if experiment["status"] in {
                "success",
                "failed_permanent",
            }:
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
            f"[ORCHESTRATOR] "
            f"{len(pending)} pending jobs"
        )

        futures = {}

        with ThreadPoolExecutor(
            max_workers=MAX_WORKERS
        ) as executor:

            for config in pending:

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

                # Main thread is the ONLY state writer.
                save_state(state)

                future = executor.submit(
                    worker,
                    config
                )

                futures[future] = config

            for future in as_completed(futures):

                config = futures[future]

                config_id = config["id"]

                experiment = get_state(
                    state,
                    config_id
                )

                try:

                    bundle = future.result()

                    result = bundle["result"]

                except Exception as exc:

                    result = {
                        "id": config_id,
                        "status": "failed",
                        "failure_type": "exception",
                        "error": str(exc),
                    }

                history = experiment[
                    "history"
                ][-1]

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

                    failure_type = (
                        classify_failure(result)
                    )

                    history.update({
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

                cleanup_workspace(config_id)

                # Single writer.
                save_state(state)

        retrying = any(
            item["status"] == "retry_wait"
            for item in state[
                "experiments"
            ].values()
        )

        if retrying:

            print(
                f"Waiting {BACKOFF_SECONDS}s "
                f"before retry..."
            )

            time.sleep(BACKOFF_SECONDS)

    save_state(state)

    print()
    print("=" * 70)
    print("CRASH-SAFE SWEEP COMPLETE")
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