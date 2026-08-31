"""Centralization and smoke checks for the EXP-012 threshold features."""

from __future__ import annotations

import hashlib
import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.features.thresholds import THRESHOLD_FEATURES, add_threshold_features
from src import train_xgboost_exp012_threshold_features as exp012


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_THRESHOLD_FEATURES = [
    "screen_gt_8", "screen_le_6", "social_gt_4", "social_le_4",
    "clear_positive_zone", "clear_negative_zone", "ambiguous_zone",
    "screen_dist_to_6", "screen_dist_to_8", "social_dist_to_4",
    "screen_abs_dist_to_6", "screen_abs_dist_to_8", "social_abs_dist_to_4",
    "screen_x_social", "screen_plus_social", "screen_minus_social",
    "screen_over_social", "social_over_screen", "min_abs_threshold_distance",
    "screen_mid_band", "social_near_threshold", "screen_near_threshold",
    "region_code",
]
HISTORICAL_OUTPUTS = (
    ROOT / "outputs" / "metrics" / "experiment_log.csv",
    ROOT / "outputs" / "metrics" / "experiment_log_v2.csv",
    ROOT / "outputs" / "metrics" / "exp012_xgboost_thresholds_metrics.txt",
    ROOT / "outputs" / "predictions" / "oof_exp012_xgboost_thresholds.csv",
    ROOT / "outputs" / "predictions" / "test_exp012_xgboost_thresholds.csv",
    ROOT / "outputs" / "submissions" / "submission_exp012_xgboost_thresholds.csv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_samples(rows: int = 2000) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    train_path = exp012.DATA_DIR / "train.csv"
    test_path = exp012.DATA_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        pytest.skip("Competition train/test data is not available locally")
    train = pd.read_csv(train_path, nrows=rows)
    test = pd.read_csv(test_path, nrows=rows // 2)
    originals = [column for column in train if column not in {exp012.ID_COLUMN, exp012.TARGET}]
    return train[originals], train[exp012.TARGET], test[originals]


def test_exp012_uses_the_official_threshold_feature_api() -> None:
    top_level = importlib.import_module("train_xgboost_exp012_threshold_features")
    assert THRESHOLD_FEATURES == EXPECTED_THRESHOLD_FEATURES
    assert exp012.NEW_FEATURES is THRESHOLD_FEATURES
    assert exp012.add_threshold_features is add_threshold_features
    assert top_level.add_threshold_features is add_threshold_features
    assert top_level.NEW_FEATURES is THRESHOLD_FEATURES


def test_threshold_api_preserves_copy_index_order_dtype_and_nan_contract() -> None:
    frame = pd.DataFrame(
        {
            "daily_screen_time_hours": [5.0, 7.0, 9.0, np.nan, 0.0],
            "social_media_hours": [3.0, 4.0, 5.0, 4.0, 0.0],
            "unrelated": [1, 2, 3, 4, 5],
        },
        index=[11, 7, 5, 3, 1],
    )
    original = frame.copy(deep=True)
    result = add_threshold_features(frame)

    assert frame.equals(original)
    assert result is not frame
    assert result.index.equals(frame.index)
    assert result.columns.tolist() == [*frame.columns, *THRESHOLD_FEATURES]
    assert result[frame.columns].equals(frame)
    assert all(result[name].dtype == np.dtype("float64") for name in THRESHOLD_FEATURES)
    assert np.isnan(result.loc[3, "screen_gt_8"])
    assert np.isnan(result.loc[3, "screen_over_social"])
    assert np.isnan(result.loc[1, "screen_over_social"])
    assert result.loc[5, "clear_positive_zone"] == 1.0
    assert result.loc[11, "clear_negative_zone"] == 1.0
    assert result.loc[7, "ambiguous_zone"] == 1.0


def test_threshold_post_migration_report_is_exact() -> None:
    report = pd.read_csv(ROOT / "outputs" / "reports" / "threshold_feature_parity_post_migration.csv")
    assert len(report) == 46
    assert set(report["feature"]) == set(THRESHOLD_FEATURES)
    assert set(report["source_script"]) == {
        "src/train_xgboost_exp012_threshold_features.py",
        "src/train_xgboost_exp016_depth5_9000.py",
    }
    assert report["nan_match"].all()
    assert (report["max_abs_diff"] == 0.0).all()
    assert (report["status"] == "PASS").all()


def test_exp012_import_has_no_output_side_effects(tmp_path: Path) -> None:
    existing = [path for path in HISTORICAL_OUTPUTS if path.exists()]
    before = {path: _sha256(path) for path in existing}
    script = ROOT / "src" / "train_xgboost_exp012_threshold_features.py"
    code = (
        "import importlib.util; "
        f"s=importlib.util.spec_from_file_location('exp012_legacy', {str(script)!r}); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert {path: _sha256(path) for path in existing} == before


def test_exp012_small_fit_predict_smoke() -> None:
    X_raw, y, X_test_raw = _load_samples()
    X_features = add_threshold_features(X_raw)
    X_test_features = add_threshold_features(X_test_raw)
    categoricals = [
        column for column in X_raw if not pd.api.types.is_numeric_dtype(X_features[column])
    ]
    X, _, _ = exp012.ordinal_encode_categories(X_features, X_test_features, categoricals)
    train_indices, valid_indices = train_test_split(
        np.arange(len(X)), test_size=0.2, random_state=42, stratify=y
    )
    params = {**exp012.MODEL_PARAMS, "n_estimators": 35, "n_jobs": 1}
    model = XGBClassifier(**params, early_stopping_rounds=5)
    model.fit(
        X.iloc[train_indices], y.iloc[train_indices],
        eval_set=[(X.iloc[valid_indices], y.iloc[valid_indices])], verbose=False,
    )
    assert model.best_iteration >= 0
    probabilities = model.predict_proba(
        X.iloc[valid_indices], iteration_range=(0, model.best_iteration + 1)
    )[:, 1]
    assert probabilities.shape == (len(valid_indices),)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
