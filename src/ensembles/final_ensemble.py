"""Final EXP-039 rank ensemble built exclusively from persisted predictions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.project_paths import (
    DATA_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
    REPORTS_DIR,
    SUBMISSIONS_DIR,
)

ID_COLUMN = "id"
TARGET = "addicted_label"
PREDICTION_COLUMN = "prediction"
EXPECTED_TEST_ROWS = 296_302

FINAL_WEIGHTS = {"EXP-027": 0.375, "EXP-037": 0.225, "EXP-039": 0.400}
LEGACY_WEIGHTS = {"EXP-027": 0.4375, "EXP-037": 0.2625, "EXP-039": 0.3000}
BLEND_METHOD = "rank"
HISTORICAL_OOF_AUC = 0.9674680607837304
HISTORICAL_FOLD_AUCS = (
    0.9668479888956474,
    0.967439309426724,
    0.9676534694408027,
    0.9681823431103556,
    0.9672203307563628,
)

EXP027_TEST_PATH = SUBMISSIONS_DIR / "submission_exp027_seed_ensemble.csv"
EXP037_TEST_PATH = PREDICTIONS_DIR / "test_exp037_relational_logistic.csv"
HISTORICAL_SUBMISSION_PATH = SUBMISSIONS_DIR / "submission_exp039_highbin_lgbm_ensemble.csv"
METRIC_PATH = METRICS_DIR / "exp039_highbin_metrics.txt"
BLEND_REPORT_PATH = REPORTS_DIR / "exp039_blends.csv"

EXP027_SEED_PATHS = (
    PREDICTIONS_DIR / "oof_exp027_seed42.csv",
    PREDICTIONS_DIR / "oof_exp027_seed2026.csv",
    PREDICTIONS_DIR / "oof_exp027_seed777.csv",
)
EXP022_OOF_PATH = PREDICTIONS_DIR / "oof_exp022_catboost_thresholds_9000.csv"
EXP037_OOF_PATH = PREDICTIONS_DIR / "oof_exp037_relational_logistic.csv"
EXP039_OOF_PATH = PREDICTIONS_DIR / "oof_exp039_lgbm_highbin.csv"


def normalized_rank(values: Sequence[float] | pd.Series | np.ndarray) -> np.ndarray:
    """Return the exact historical average-tie rank scaled to [0, 1]."""
    series = pd.Series(values)
    if len(series) < 2:
        raise ValueError("Rank normalization requires at least two predictions")
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("Predictions must be finite")
    ranks = pd.Series(numeric).rank(method="average").to_numpy(np.float64)
    return (ranks - 1.0) / (len(ranks) - 1.0)


def validate_prediction_frame(
    frame: pd.DataFrame,
    prediction_column: str,
    *,
    expected_ids: pd.Series | Sequence[object] | None = None,
    name: str = "predictions",
) -> np.ndarray:
    """Validate IDs and scores, preserving the historical row-order contract."""
    required = {ID_COLUMN, prediction_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    if frame[ID_COLUMN].isna().any() or frame[ID_COLUMN].duplicated().any():
        raise ValueError(f"{name} contains missing or duplicate IDs")
    if expected_ids is not None:
        expected = pd.Series(expected_ids).reset_index(drop=True)
        actual = frame[ID_COLUMN].reset_index(drop=True)
        if len(actual) != len(expected):
            raise ValueError(f"{name} prediction length mismatch")
        if not actual.equals(expected):
            raise ValueError(f"{name} IDs are not aligned in the expected order")
    values = pd.to_numeric(frame[prediction_column], errors="coerce").to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} predictions must be finite")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError(f"{name} predictions must be within [0, 1]")
    return values


def load_predictions(
    path: Path,
    prediction_column: str,
    *,
    expected_ids: pd.Series | Sequence[object] | None = None,
    name: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load one persisted prediction artifact and validate its contract."""
    frame = pd.read_csv(path)
    values = validate_prediction_frame(
        frame, prediction_column, expected_ids=expected_ids, name=name or path.name
    )
    return frame, values


def blend_predictions(
    components: Mapping[str, Sequence[float] | np.ndarray],
    weights: Mapping[str, float] = FINAL_WEIGHTS,
) -> np.ndarray:
    """Combine already rank-normalized components with fixed historical weights."""
    if set(components) != set(weights):
        raise ValueError("Component names must match weight names exactly")
    if not np.isclose(sum(weights.values()), 1.0, rtol=0.0, atol=1e-15):
        raise ValueError("Blend weights must sum to one")
    arrays = {name: np.asarray(values, dtype=np.float64) for name, values in components.items()}
    lengths = {len(values) for values in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("Prediction length mismatch between components")
    if not all(np.isfinite(values).all() for values in arrays.values()):
        raise ValueError("Blend components must be finite")
    blended = sum(float(weights[name]) * arrays[name] for name in weights)
    if ((blended < 0.0) | (blended > 1.0)).any():
        raise ValueError("Blended predictions must be within [0, 1]")
    return np.asarray(blended, dtype=np.float64)


def recover_component_from_blend(
    blended: Sequence[float] | np.ndarray,
    known_components: Mapping[str, Sequence[float] | np.ndarray],
    weights: Mapping[str, float],
    component_name: str,
) -> np.ndarray:
    """Recover the unstored fold-rank mean exactly as the historical finalizer did."""
    if component_name not in weights or weights[component_name] == 0:
        raise ValueError("Recovered component must have a non-zero weight")
    result = np.asarray(blended, dtype=np.float64).copy()
    for name, values in known_components.items():
        result -= float(weights[name]) * np.asarray(values, dtype=np.float64)
    return result / float(weights[component_name])


def build_final_submission(ids: Sequence[object], components: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Build the final two-column submission without writing it."""
    id_series = pd.Series(ids).reset_index(drop=True)
    if id_series.isna().any() or id_series.duplicated().any():
        raise ValueError("Submission IDs must be present and unique")
    predictions = blend_predictions(components)
    if len(id_series) != len(predictions):
        raise ValueError("Submission ID and prediction length mismatch")
    return pd.DataFrame({ID_COLUMN: id_series, TARGET: predictions})


def refine_legacy_submission(
    legacy_submission: pd.DataFrame,
    exp027_test: pd.DataFrame,
    exp037_test: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the exact historical old-to-refined EXP-039 transformation."""
    old = validate_prediction_frame(legacy_submission, TARGET, name="legacy EXP-039 submission")
    ids = legacy_submission[ID_COLUMN]
    rank027 = normalized_rank(validate_prediction_frame(
        exp027_test, TARGET, expected_ids=ids, name="EXP-027 test"
    ))
    rank037 = normalized_rank(validate_prediction_frame(
        exp037_test, PREDICTION_COLUMN, expected_ids=ids, name="EXP-037 test"
    ))
    mean_rank039 = recover_component_from_blend(
        old,
        {"EXP-027": rank027, "EXP-037": rank037},
        LEGACY_WEIGHTS,
        "EXP-039",
    )
    return build_final_submission(
        ids,
        {"EXP-027": rank027, "EXP-037": rank037, "EXP-039": mean_rank039},
    )


def reconstruct_historical_submission() -> pd.DataFrame:
    """Rebuild the current final submission read-only for exact parity checks."""
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    historical, final_values = load_predictions(
        HISTORICAL_SUBMISSION_PATH, TARGET, expected_ids=sample[ID_COLUMN], name="final submission"
    )
    exp027, values027 = load_predictions(
        EXP027_TEST_PATH, TARGET, expected_ids=sample[ID_COLUMN], name="EXP-027 test"
    )
    exp037, values037 = load_predictions(
        EXP037_TEST_PATH, PREDICTION_COLUMN, expected_ids=sample[ID_COLUMN], name="EXP-037 test"
    )
    rank027 = normalized_rank(values027)
    rank037 = normalized_rank(values037)
    mean_rank039 = recover_component_from_blend(
        final_values,
        {"EXP-027": rank027, "EXP-037": rank037},
        FINAL_WEIGHTS,
        "EXP-039",
    )
    return build_final_submission(
        historical[ID_COLUMN],
        {"EXP-027": rank027, "EXP-037": rank037, "EXP-039": mean_rank039},
    )


def write_submission(submission: pd.DataFrame, output_path: Path) -> Path:
    """Write to an explicit path; callers control whether it is temporary or productive."""
    validate_prediction_frame(submission, TARGET, name="output submission")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return output_path


def main(output_path: Path | str | None = None) -> Path:
    """Preserve the legacy no-op final state or write a requested alternate copy."""
    if output_path is None:
        if (
            "refined_blend=rank" in METRIC_PATH.read_text(encoding="utf8")
            and "REFINED_TRIPLE" in pd.read_csv(BLEND_REPORT_PATH).candidate.astype(str).values
        ):
            print(HISTORICAL_SUBMISSION_PATH)
            return HISTORICAL_SUBMISSION_PATH
        raise RuntimeError("Historical pre-refinement state is unavailable; refusing to overwrite outputs")
    destination = Path(output_path)
    result = write_submission(reconstruct_historical_submission(), destination)
    print(result)
    return result


__all__ = [
    "BLEND_METHOD", "BLEND_REPORT_PATH", "EXPECTED_TEST_ROWS", "EXP022_OOF_PATH",
    "EXP027_SEED_PATHS", "EXP027_TEST_PATH", "EXP037_OOF_PATH", "EXP037_TEST_PATH",
    "EXP039_OOF_PATH", "FINAL_WEIGHTS", "HISTORICAL_FOLD_AUCS",
    "HISTORICAL_OOF_AUC", "HISTORICAL_SUBMISSION_PATH", "ID_COLUMN", "LEGACY_WEIGHTS",
    "METRIC_PATH", "PREDICTION_COLUMN", "TARGET", "blend_predictions",
    "build_final_submission", "load_predictions", "main", "normalized_rank",
    "reconstruct_historical_submission", "recover_component_from_blend",
    "refine_legacy_submission", "validate_prediction_frame", "write_submission",
]


if __name__ == "__main__":
    main()
