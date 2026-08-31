"""Compatibility, configuration and smoke checks for migrated EXP-016."""

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
from src.models import xgboost_thresholds as migrated
from src import train_xgboost_exp016_depth5_9000 as legacy


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODEL = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_estimators": 9000,
    "learning_rate": 0.03,
    "max_depth": 5,
    "min_child_weight": 1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "gamma": 0.0,
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}
HISTORICAL_OUTPUTS = (
    ROOT / "outputs" / "metrics" / "experiment_log.csv",
    ROOT / "outputs" / "metrics" / "experiment_log_v2.csv",
    ROOT / "outputs" / "metrics" / "exp016_xgboost_depth5_9000_metrics.txt",
    ROOT / "outputs" / "predictions" / "oof_exp016_xgboost_depth5_9000.csv",
    ROOT / "outputs" / "predictions" / "test_exp016_xgboost_depth5_9000.csv",
    ROOT / "outputs" / "submissions" / "submission_exp016_xgboost_depth5_9000.csv",
    ROOT / "outputs" / "reports" / "exp016_learning_curve.csv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_samples(rows: int = 2000) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    if not (migrated.DATA / "train.csv").exists() or not (migrated.DATA / "test.csv").exists():
        pytest.skip("Competition train/test data is not available locally")
    train = pd.read_csv(migrated.DATA / "train.csv", nrows=rows)
    test = pd.read_csv(migrated.DATA / "test.csv", nrows=rows // 2)
    originals = [column for column in train if column not in {migrated.ID, migrated.TARGET}]
    return train[originals], train[migrated.TARGET], test[originals]


def test_exp016_wrapper_reexports_public_api() -> None:
    top_level = importlib.import_module("train_xgboost_exp016_depth5_9000")
    for name in (
        "update_log", "calculate_correlations", "main",
        "add_threshold_features", "ordinal_encode_categories", "validate_submission",
    ):
        assert getattr(legacy, name) is getattr(migrated, name)
        assert getattr(top_level, name) is getattr(migrated, name)
    assert migrated.add_threshold_features is add_threshold_features
    assert migrated.NEW_FEATURES is THRESHOLD_FEATURES
    assert legacy.MODEL is migrated.MODEL


def test_exp016_productive_configuration_and_output_contract_are_preserved() -> None:
    assert migrated.MODEL == EXPECTED_MODEL
    assert migrated.EARLY_STOPPING == 300
    assert migrated.EXPERIMENT == "EXP-016"
    assert migrated.ID == "id"
    assert migrated.TARGET == "addicted_label"
    source = Path(migrated.__file__).read_text(encoding="utf-8")
    assert "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)" in source
    assert "XGBClassifier(**MODEL, early_stopping_rounds=EARLY_STOPPING)" in source
    assert "eval_set=[(X.iloc[train_idx], y.iloc[train_idx])]" not in source
    assert "eval_set=[(X.iloc[valid_idx], y.iloc[valid_idx])]" in source
    assert "iteration_range = (0, best + 1)" in source
    assert "verbose=False" in source
    assert migrated.OOF_PATH.name == "oof_exp016_xgboost_depth5_9000.csv"
    assert migrated.TEST_PATH.name == "test_exp016_xgboost_depth5_9000.csv"
    assert migrated.SUBMISSION_PATH.name == "submission_exp016_xgboost_depth5_9000.csv"
    assert migrated.METRICS_PATH.name == "exp016_xgboost_depth5_9000_metrics.txt"
    assert migrated.CURVE_PATH.name == "exp016_learning_curve.csv"
    assert migrated.LOG_PATH.name == "experiment_log.csv"


def test_exp016_import_has_no_output_side_effects(tmp_path: Path) -> None:
    existing = [path for path in HISTORICAL_OUTPUTS if path.exists()]
    before = {path: _sha256(path) for path in existing}
    script = ROOT / "src" / "train_xgboost_exp016_depth5_9000.py"
    code = (
        "import importlib.util; "
        f"s=importlib.util.spec_from_file_location('exp016_legacy', {str(script)!r}); "
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


def test_exp016_small_fit_predict_smoke() -> None:
    X_raw, y, X_test_raw = _load_samples()
    raw = add_threshold_features(X_raw)
    raw_test = add_threshold_features(X_test_raw)
    assert raw.columns.tolist() == raw_test.columns.tolist()
    categoricals = [column for column in X_raw if not pd.api.types.is_numeric_dtype(raw[column])]
    X, _, _ = migrated.ordinal_encode_categories(raw, raw_test, categoricals)
    train_indices, valid_indices = train_test_split(
        np.arange(len(X)), test_size=0.2, random_state=42, stratify=y
    )
    params = {**migrated.MODEL, "n_estimators": 40, "n_jobs": 1}
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
