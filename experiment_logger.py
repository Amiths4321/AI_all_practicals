import json
import time
from datetime import datetime
from pathlib import Path


class ExperimentLogger:

    def __init__(self, experiment_id, output_file="experiment_log.json"):
        self.experiment_id = experiment_id
        self.output_file = Path(output_file)

        self.started_at = datetime.now().isoformat()
        self.events = []

    def log(self, event, stage=None, **data):
        record = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "stage": stage,
            **data,
        }

        self.events.append(record)
        self._save()

    def stage_start(self, stage):
        self.log(
            event="stage_start",
            stage=stage,
        )

    def stage_end(self, stage, started_at):
        elapsed = time.perf_counter() - started_at

        self.log(
            event="stage_end",
            stage=stage,
            latency_seconds=elapsed,
        )

    def error(self, stage, error):
        self.log(
            event="error",
            stage=stage,
            error_type=type(error).__name__,
            error_message=str(error),
        )

    def retry(self, stage, attempt):
        self.log(
            event="retry",
            stage=stage,
            attempt=attempt,
        )

    def finish(self, status):
        self.log(
            event="experiment_finished",
            status=status,
            started_at=self.started_at,
            finished_at=datetime.now().isoformat(),
        )

    def _save(self):
        payload = {
            "experiment_id": self.experiment_id,
            "started_at": self.started_at,
            "events": self.events,
        }

        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def timed_stage(logger, stage):
    class StageContext:

        def __enter__(self):
            self.started = time.perf_counter()
            logger.stage_start(stage)
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            if exc_value is not None:
                logger.error(stage, exc_value)

            logger.stage_end(
                stage,
                self.started,
            )

            return False

    return StageContext()