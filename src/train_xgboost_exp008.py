"""Compatibility wrapper for the migrated EXP-008 XGBoost baseline."""

from src.models.xgboost_baseline import (
    DATA_DIR,
    EXPERIMENT_ID,
    ID_COLUMN,
    LOG_PATH,
    METRICS_DIR,
    METRICS_PATH,
    MODEL_PARAMS,
    OOF_PATH,
    OUTPUTS_DIR,
    PREDICTIONS_DIR,
    PROJECT_ROOT,
    REFERENCE_OOF_PATHS,
    SUBMISSIONS_DIR,
    SUBMISSION_PATH,
    TARGET,
    TEST_PREDICTIONS_PATH,
    diversity_correlations,
    main,
    ordinal_encode_categories,
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
    "MODEL_PARAMS",
    "OOF_PATH",
    "OUTPUTS_DIR",
    "PREDICTIONS_DIR",
    "PROJECT_ROOT",
    "REFERENCE_OOF_PATHS",
    "SUBMISSIONS_DIR",
    "SUBMISSION_PATH",
    "TARGET",
    "TEST_PREDICTIONS_PATH",
    "diversity_correlations",
    "main",
    "ordinal_encode_categories",
    "update_experiment_log",
    "validate_submission",
]


if __name__ == "__main__":
    main()
