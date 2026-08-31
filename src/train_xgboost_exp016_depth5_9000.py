"""Compatibility wrapper for the migrated EXP-016 XGBoost threshold model."""

from pathlib import Path
import sys

_PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_FOR_IMPORTS))

from src.models.xgboost_thresholds import (
    CURVE_PATH,
    DATA,
    EARLY_STOPPING,
    EXP015_FOLDS,
    EXPERIMENT,
    ID,
    LOG_PATH,
    METRICS,
    METRICS_PATH,
    MODEL,
    MODEL_PARAMS,
    NEW_FEATURES,
    OOF_PATH,
    OUTPUTS,
    PREDICTIONS,
    REFERENCE_PATHS,
    REPORTS,
    ROOT,
    SUBMISSIONS,
    SUBMISSION_PATH,
    TARGET,
    TEST_PATH,
    add_threshold_features,
    calculate_correlations,
    main,
    ordinal_encode_categories,
    update_log,
    validate_submission,
)

__all__ = [
    "CURVE_PATH",
    "DATA",
    "EARLY_STOPPING",
    "EXP015_FOLDS",
    "EXPERIMENT",
    "ID",
    "LOG_PATH",
    "METRICS",
    "METRICS_PATH",
    "MODEL",
    "MODEL_PARAMS",
    "NEW_FEATURES",
    "OOF_PATH",
    "OUTPUTS",
    "PREDICTIONS",
    "REFERENCE_PATHS",
    "REPORTS",
    "ROOT",
    "SUBMISSIONS",
    "SUBMISSION_PATH",
    "TARGET",
    "TEST_PATH",
    "add_threshold_features",
    "calculate_correlations",
    "main",
    "ordinal_encode_categories",
    "update_log",
    "validate_submission",
]


if __name__ == "__main__":
    main()
