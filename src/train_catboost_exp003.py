"""Compatibility wrapper for the migrated EXP-003 CatBoost model."""

from src.models.catboost_model import (
    DATA_DIR,
    EXP002_METRICS_PATH,
    EXPERIMENT_ID,
    ID_COLUMN,
    LOG_PATH,
    METRICS_DIR,
    METRICS_PATH,
    MODEL_PARAMS,
    N_SPLITS,
    OOF_PATH,
    PROJECT_ROOT,
    RANDOM_SEED,
    SUBMISSIONS_DIR,
    SUBMISSION_PATH,
    TARGET,
    main,
    parse_metrics,
    prepare_categorical_missing,
    update_experiment_log,
    validate_submission,
)

__all__ = [
    "DATA_DIR",
    "EXP002_METRICS_PATH",
    "EXPERIMENT_ID",
    "ID_COLUMN",
    "LOG_PATH",
    "METRICS_DIR",
    "METRICS_PATH",
    "MODEL_PARAMS",
    "N_SPLITS",
    "OOF_PATH",
    "PROJECT_ROOT",
    "RANDOM_SEED",
    "SUBMISSIONS_DIR",
    "SUBMISSION_PATH",
    "TARGET",
    "main",
    "parse_metrics",
    "prepare_categorical_missing",
    "update_experiment_log",
    "validate_submission",
]


if __name__ == "__main__":
    main()
