"""Compatibility wrapper for the migrated EXP-001 baseline."""

from src.models.baseline_logistic import (
    DATA_DIR,
    EXPERIMENT_ID,
    ID_COLUMN,
    LOG_PATH,
    METRICS_DIR,
    METRICS_PATH,
    N_SPLITS,
    OOF_PATH,
    PROJECT_ROOT,
    RANDOM_STATE,
    SUBMISSIONS_DIR,
    SUBMISSION_PATH,
    TARGET,
    build_pipeline,
    main,
    update_experiment_log,
    validate_submission,
)

__all__ = [
    "DATA_DIR",
    "EXPERIMENT_ID",
    "ID_COLUMN",
    "LOG_PATH",
    "METRICS_DIR",
    "METRICS_PATH",
    "N_SPLITS",
    "OOF_PATH",
    "PROJECT_ROOT",
    "RANDOM_STATE",
    "SUBMISSIONS_DIR",
    "SUBMISSION_PATH",
    "TARGET",
    "build_pipeline",
    "main",
    "update_experiment_log",
    "validate_submission",
]


if __name__ == "__main__":
    main()
