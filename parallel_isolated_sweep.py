from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_compression import SentenceCompressor
from context_builder import ContextBuilder
from pathlib import Path
import json
import shutil
import subprocess
import sys
import time
from experiment_manifest import create_manifest, save_manifest

# --- Module-Level Directory Setup ---
BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = BASE_DIR / "experiment_workspaces"
OUTPUT_ROOT = BASE_DIR / "experiment_outputs"

CONFIG_FILE = BASE_DIR / "sweep_configurations.json"
MAX_WORKERS = 2
MAX_CONFIGURATIONS = 1

# config.py is handled dynamically via config_code, so remove it from static copy list
PIPELINE_FILES = [
    "adaptive_rag.py",
    "final_adaptive_evaluation.py",
    "final_rag_report.py",
    "failure_monitor.py",
    "chunking.py",
    "hybrid_search.py",
    "reranking.py",
    "experiment_manifest.py",
    "semantic_evidence.py",
    "context_builder.py",
    "evidence_compression.py",
    "sentence_compression.py",
    "context_compression.py",
    "cached_generation.py",
    "cached_judge.py",
    "cache.py",
    "generation.py",
    "rag_judge.py",
]

INPUT_FILES = [
    "evaluation_data.json",
    "evidence_policy_benchmark.json",
    "empirical_context_policy.json",
]

DOCUMENT_FILES = [
    "documents/leave_policy.txt",
    "documents/employee_handbook.txt",
]

# Template for dynamic config.py generation in each workspace
config_code = """from pathlib import Path
import json
import os

BASE_DIR = Path(__file__).resolve().parent

CONFIG = {
    "chunk_size": 500,
    "chunk_overlap": 50,
}

def load_config():
    '''Safely loads and returns the configuration dictionary from config.json if available.'''
    config_file = BASE_DIR / "config.json"
    if config_file.exists() and config_file.stat().st_size > 0:
        with open(config_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return CONFIG
"""


def load_configurations():
  with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    return json.load(f)


def prepare_workspace(config):
  config_id = config["id"]
  workspace = WORKSPACE_ROOT / config_id

  if workspace.exists():
    shutil.rmtree(workspace)

  workspace.mkdir(parents=True)

  # 1. Copy standard pipeline files
  for filename in PIPELINE_FILES:
    source = BASE_DIR / filename
    if not source.exists():
      raise FileNotFoundError(f"Missing pipeline file: {source}")
    shutil.copy2(source, workspace / filename)

  # 2. Write dynamic config.py containing load_config()
  config_py_path = workspace / "config.py"
  with open(config_py_path, "w", encoding="utf-8") as f:
    f.write(config_code)

  # 3. Copy input files
  for filename in INPUT_FILES:
    source = BASE_DIR / filename
    if not source.exists():
      raise FileNotFoundError(f"Missing input file: {source}")
    shutil.copy2(source, workspace / filename)

  # 4. Copy documents
  for filename in DOCUMENT_FILES:
    source = BASE_DIR / filename
    if not source.exists():
      raise FileNotFoundError(f"Missing document: {source}")
    destination = workspace / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)

  # 5. Write specific config.json for this workspace
  config_path = workspace / "config.json"
  with open(config_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "chunking": {
                "chunk_size": config["chunk_size"],
                "overlap": config["overlap"],
            },
            "retrieval": {
                "vector_k": 10,
                "hybrid_k": 10,
                "rerank_k": config["rerank_k"],
            },
            "context": {
                "default_budget": config["default_budget"],
            },
            "evidence": {
                "threshold": 0.60,
            },
        },
        f,
        indent=2,
    )

  return workspace


def run_script(workspace, script_name):
  result = subprocess.run(
      [sys.executable, script_name],
      cwd=workspace,
      capture_output=True,
      text=True,
  )

  if result.returncode != 0:
    raise RuntimeError(
        f"{script_name} failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

  return result.stdout


def extract_metrics(workspace):
  metrics = {}

  evaluation_file = workspace / "final_adaptive_evaluation.json"
  if evaluation_file.exists():
    with open(evaluation_file, "r", encoding="utf-8") as f:
      metrics["evaluation"] = json.load(f)

  failure_file = workspace / "failure_monitoring.json"
  if failure_file.exists():
    with open(failure_file, "r", encoding="utf-8") as f:
      metrics["failure_monitoring"] = json.load(f)

  return metrics


def run_experiment(config):
  config_id = config["id"]
  start = time.time()
  print(f"[START] {config_id}")

  try:
    workspace = prepare_workspace(config)

    scripts = [
        "adaptive_rag.py",
        "final_adaptive_evaluation.py",
        "final_rag_report.py",
        "failure_monitor.py",
    ]

    for script in scripts:
      print(f"[{config_id}] Running {script}")
      run_script(workspace, script)

    metrics = extract_metrics(workspace)
    finished_at = time.time()

    manifest = create_manifest(
        experiment_id=config_id,
        configuration=config,
        workspace=workspace,
        started_at=start,
        finished_at=finished_at,
    )

    save_manifest(
        manifest,
        workspace / "experiment_manifest.json"
    )
    elapsed = time.time() - start
    elapsed = finished_at - start
    output_dir = OUTPUT_ROOT / config_id
    if output_dir.exists():
      shutil.rmtree(output_dir)
    shutil.copytree(workspace, output_dir)

    print(f"[DONE] {config_id} ({elapsed:.1f}s)")
    return {
        "id": config_id,
        "status": "success",
        "configuration": config,
        "elapsed_seconds": elapsed,
        "metrics": metrics,
    }
  
  except Exception as exc:
    elapsed = time.time() - start
    print(f"[FAILED] {config_id}: {exc}")
    return {
        "id": config_id,
        "status": "failed",
        "configuration": config,
        "elapsed_seconds": elapsed,
        "error": str(exc),
    }


def main():
  WORKSPACE_ROOT.mkdir(exist_ok=True, parents=True)
  OUTPUT_ROOT.mkdir(exist_ok=True, parents=True)

  print(f"Using workspace root: {WORKSPACE_ROOT.absolute()}")

  configurations = load_configurations()

    # Safely extract list whether JSON is a dict containing "configurations", a raw dict of items, or already a list
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

  print("=" * 70)
  print("PARALLEL ISOLATED SWEEP")
  print("=" * 70)
  print(f"Configurations: {len(configurations)}")
  print(f"Workers:       {MAX_WORKERS}")
  print()

  results = []

  with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {
        executor.submit(run_experiment, config): config["id"]
        for config in configurations
    }

    for future in as_completed(futures):
      result = future.result()
      results.append(result)

  results.sort(key=lambda item: item["id"])

  output_file = BASE_DIR / "sweep_results.json"
  with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

  successful = sum(
      1 for result in results if result["status"] == "success"
  )
  failed = len(results) - successful

  print()
  print("=" * 70)
  print("SWEEP COMPLETE")
  print("=" * 70)
  print(f"Successful: {successful}")
  print(f"Failed:    {failed}")
  print(f"Results:   {output_file}")


if __name__ == "__main__":
  main()