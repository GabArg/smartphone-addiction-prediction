"""Compatibility wrapper for the migrated final EXP-039 ensemble."""

from pathlib import Path
import sys

_PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_FOR_IMPORTS))

from src.ensembles.final_ensemble import (
    BLEND_METHOD,
    BLEND_REPORT_PATH,
    EXPECTED_TEST_ROWS,
    EXP022_OOF_PATH,
    EXP027_SEED_PATHS,
    EXP027_TEST_PATH,
    EXP037_OOF_PATH,
    EXP037_TEST_PATH,
    EXP039_OOF_PATH,
    FINAL_WEIGHTS,
    HISTORICAL_FOLD_AUCS,
    HISTORICAL_OOF_AUC,
    HISTORICAL_SUBMISSION_PATH,
    ID_COLUMN,
    LEGACY_WEIGHTS,
    METRIC_PATH,
    PREDICTION_COLUMN,
    TARGET,
    blend_predictions,
    build_final_submission,
    load_predictions,
    main,
    normalized_rank,
    reconstruct_historical_submission,
    recover_component_from_blend,
    refine_legacy_submission,
    validate_prediction_frame,
    write_submission,
)

__all__ = [name for name in globals() if not name.startswith("_") and name not in {"Path", "sys"}]


if __name__ == "__main__":
    main()
