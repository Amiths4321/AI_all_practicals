from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import os
import signal
import shutil
import subprocess
import sys
import threading
import time
from experiment_logger import ExperimentLogger

from parallel_isolated_sweep import (
    prepare_workspace,
    load_configurations,
)


BASE_DIR = Path(__file__).resolve().parent

STATE_FILE = BASE_DIR / "graceful_sweep_state.json"
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


shutdown_event = threading.Event()

active_processes = {}
active_processes_lock = threading.Lock()


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

        logger = ExperimentLogger(
            experiment_id=config_id,
            output_file=str(workspace / "experiment_log.json"),
        )

        experiment["attempts"] += 1
        attempt = experiment["attempts"]

        
    return state["experiments"][config_id]


def cleanup_workspace(config_id):

    workspace = (
        WORKSPACE_ROOT / config_id
    )

    if workspace.exists():

        try:
            shutil.rmtree(workspace)

            print(
                f"[CLEANUP] {config_id}"
            )

        except Exception as exc:

            print(
                f"[CLEANUP WARNING] "
                f"{config_id}: {exc}"
            )


def terminate_process_tree(process):

    if process is None:
        return

    if process.poll() is not None:
        return

    print(
        f"[TERMINATE] PID={process.pid}"
    )

    try:

        if os.name == "nt":

            # Windows: terminate the entire process tree.
            subprocess.run(
                [
                    "taskkill",
                    "/PID",
                    str(process.pid),
                    "/T",
                    "/F",
                ],
                capture_output=True,
                text=True,
            )

        else:

            os.killpg(
                os.getpgid(process.pid),
                signal.SIGTERM,
            )

    except Exception as exc:

        print(
            f"[TERMINATION WARNING] {exc}"
        )


def shutdown_handler(signum, frame):

    if shutdown_event.is_set():
        return

    print()
    print("=" * 70)
    print("SHUTDOWN REQUESTED")
    print("=" * 70)

    shutdown_event.set()

    with active_processes_lock:

        processes = list(
            active_processes.items()
        )

    for config_id, process in processes:

        print(
            f"[SHUTDOWN] "
            f"Terminating {config_id}"
        )

        terminate_process_tree(process)


def run_process(config):

    config_id = config["id"]

    workspace = prepare_workspace(config)

    start = time.time()

    logs = []

    process = None

    try:

        for script in PIPELINE:

            if shutdown_event.is_set():

                return {
                    "id": config_id,
                    "status": "interrupted",
                    "failure_type": "shutdown",
                    "error": (
                        "Orchestrator shutdown "
                        "requested"
                    ),
                }

            print(
                f"[{config_id}] "
                f"running {script}"
            )

            process = subprocess.Popen(
                [
                    sys.executable,
                    script,
                ],
                cwd=workspace,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            with active_processes_lock:

                active_processes[
                    config_id
                ] = process

            try:

                stdout, stderr = (
                    process.communicate(
                        timeout=EXPERIMENT_TIMEOUT_SECONDS
                    )
                )

            except subprocess.TimeoutExpired:

                terminate_process_tree(
                    process
                )

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

            finally:

                with active_processes_lock:

                    active_processes.pop(
                        config_id,
                        None
                    )

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

    except Exception as exc:

        return {
            "id": config_id,
            "status": "failed",
            "failure_type": "exception",
            "error": str(exc),
            "logs": logs,
            "elapsed_seconds": (
                time.time() - start
            ),
        }

    finally:

        with active_processes_lock:

            active_processes.pop(
                config_id,
                None
            )


def worker(config):

    return {
        "config_id": config["id"],
        "result": run_process(config),
    }


def reconcile_state(state):

    changed = False

    for config_id, experiment in (
        state["experiments"].items()
    ):

        if experiment["status"] != "running":
            continue

        print(
            f"[RECOVER] {config_id} "
            f"was running during shutdown"
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


def classify_failure(result):

    if result.get("failure_type") in {
        "timeout",
        "connection_error",
        "server_error",
    }:
        return "transient"

    return "permanent"


def main():

    signal.signal(
        signal.SIGINT,
        shutdown_handler
    )

    if hasattr(signal, "SIGTERM"):
        signal.signal(
            signal.SIGTERM,
            shutdown_handler
        )

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

    reconcile_state(state)

    print("=" * 70)
    print("GRACEFUL EXPERIMENT SWEEP")
    print("=" * 70)

    while not shutdown_event.is_set():

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

        print(
            f"[ORCHESTRATOR] "
            f"Pending={len(pending)}"
        )

        futures = {}

        with ThreadPoolExecutor(
            max_workers=MAX_WORKERS
        ) as executor:

            for config in pending:

                if shutdown_event.is_set():
                    break

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

                if shutdown_event.is_set():

                    experiment["status"] = (
                        "retry_wait"
                    )

                    if experiment["history"]:

                        experiment["history"][-1].update({
                            "status": "interrupted",
                            "finished_at": time.time(),
                        })

                    cleanup_workspace(
                        config_id
                    )

                    save_state(state)

                    continue

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

                    else:

                        experiment["status"] = (
                            "failed_permanent"
                        )

                        experiment["result"] = result

                cleanup_workspace(
                    config_id
                )

                save_state(state)

        if shutdown_event.is_set():
            break

        retrying = any(
            item["status"] == "retry_wait"
            for item in state[
                "experiments"
            ].values()
        )

        if retrying:

    for item in experiments:
        if item["status"] == "retry_wait":

            logger = loggers.get(item["config_id"])

            if logger:
                logger.retry(
                    stage=f"experiment:{item['config_id']}",
                    attempt=item["attempts"],
                )

    print(
        f"Waiting {BACKOFF_SECONDS}s "
        f"before retry..."
    )

    time.sleep(BACKOFF_SECONDS)

    save_state(state)

    if shutdown_event.is_set():

        print()
        print(
            "Sweep stopped safely. "
            "Run again to resume."
        )

        return

    print()
    print("=" * 70)
    print("SWEEP COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()