from experiment_logger import ExperimentLogger, timed_stage


logger = ExperimentLogger(
    experiment_id="test_observability"
)

with timed_stage(logger, "retrieval"):
    pass

with timed_stage(logger, "reranking"):
    pass

with timed_stage(logger, "generation"):
    pass

logger.finish("success")


print("Observability test complete.")