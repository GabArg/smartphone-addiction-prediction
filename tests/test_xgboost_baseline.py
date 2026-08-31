"""Parity and smoke checks for the migrated EXP-008 XGBoost baseline."""

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

from src import train_xgboost_exp008 as legacy
from src.models import xgboost_baseline as migrated


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FEATURES = [
    "age",
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
    "gender",
    "stress_level",
    "academic_work_impact",
]
EXPECTED_CATEGORICAL = ["gender", "stress_level", "academic_work_impact"]
EXPECTED_MODEL_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_estimators": 6000,
    "learning_rate": 0.03,
    "max_depth": 7,
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
    ROOT / "outputs" / "metrics" / "exp008_xgboost_metrics.txt",
    ROOT / "outputs" / "predictions" / "oof_exp008_xgboost.csv",
    ROOT / "outputs" / "predictions" / "test_exp008_xgboost.csv",
    ROOT / "outputs" / "submissions" / "submission_exp008_xgboost.csv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_samples(
    train_rows: int = 2000, test_rows: int = 1000
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str]]:
    train_path = migrated.DATA_DIR / "train.csv"
    test_path = migrated.DATA_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        pytest.skip("Competition train/test data is not available locally")
    train = pd.read_csv(train_path, nrows=train_rows)
    test = pd.read_csv(test_path, nrows=test_rows)
    features = [
        column for column in train.columns if column not in {migrated.ID_COLUMN, migrated.TARGET}
    ]
    return train[features], train[migrated.TARGET], test[features], features


def test_xgboost_wrapper_reexports_public_api() -> None:
    top_level_legacy = importlib.import_module("train_xgboost_exp008")
    for name in (
        "ordinal_encode_categories",
        "validate_submission",
        "diversity_correlations",
        "update_experiment_log",
        "main",
    ):
        assert getattr(legacy, name) is getattr(migrated, name)
        assert getattr(top_level_legacy, name) is getattr(migrated, name)

    assert legacy.MODEL_PARAMS is migrated.MODEL_PARAMS
    assert legacy.REFERENCE_OOF_PATHS is migrated.REFERENCE_OOF_PATHS
    assert legacy.EXPERIMENT_ID == "EXP-008"
    assert legacy.TARGET == "addicted_label"
    assert legacy.ID_COLUMN == "id"


def test_exp008_productive_configuration_is_preserved() -> None:
    assert migrated.MODEL_PARAMS == EXPECTED_MODEL_PARAMS
    source = Path(migrated.__file__).read_text(encoding="utf-8")
    assert "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)" in source
    assert "XGBClassifier(**MODEL_PARAMS, early_stopping_rounds=200)" in source
    assert "eval_set=[(X.iloc[valid_indices], y.iloc[valid_indices])]" in source
    assert "verbose=False" in source
    assert "iteration_range = (0, best_iteration + 1)" in source
    assert "iteration_range=iteration_range" in source

    assert migrated.OOF_PATH.name == "oof_exp008_xgboost.csv"
    assert migrated.TEST_PREDICTIONS_PATH.name == "test_exp008_xgboost.csv"
    assert migrated.SUBMISSION_PATH.name == "submission_exp008_xgboost.csv"
    assert migrated.METRICS_PATH.name == "exp008_xgboost_metrics.txt"
    assert migrated.LOG_PATH.name == "experiment_log.csv"


def test_exp008_feature_order_and_ordinal_mapping_are_preserved() -> None:
    X_raw, _, X_test_raw, features = _load_samples()
    assert features == EXPECTED_FEATURES
    numeric = X_raw.select_dtypes(include=["number"]).columns.tolist()
    categorical = [column for column in features if column not in numeric]
    assert categorical == EXPECTED_CATEGORICAL

    X, X_test, mappings = migrated.ordinal_encode_categories(
        X_raw, X_test_raw, categorical
    )
    assert X.columns.tolist() == features
    assert X_test.columns.tolist() == features
    assert X[numeric].equals(X_raw[numeric])
    assert X_test[numeric].equals(X_test_raw[numeric])
    for column in categorical:
        train_values = X_raw[column].fillna("__MISSING__").astype(str)
        test_values = X_test_raw[column].fillna("__MISSING__").astype(str)
        categories = sorted(set(train_values.unique()) | set(test_values.unique()))
        expected_mapping = {category: code for code, category in enumerate(categories)}
        assert mappings[column] == expected_mapping
        assert X[column].dtype == np.dtype("int32")
        assert X_test[column].dtype == np.dtype("int32")
        assert X[column].equals(train_values.map(expected_mapping).astype(np.int32))
        assert X_test[column].equals(test_values.map(expected_mapping).astype(np.int32))


def test_exp008_mapping_includes_test_and_missing_without_fallback_encoder() -> None:
    train = pd.DataFrame({"category": ["b", None, "a"]})
    test = pd.DataFrame({"category": ["c", None]})
    encoded_train, encoded_test, mappings = migrated.ordinal_encode_categories(
        train, test, ["category"]
    )
    assert mappings == {"category": {"__MISSING__": 0, "a": 1, "b": 2, "c": 3}}
    assert encoded_train["category"].tolist() == [2, 0, 1]
    assert encoded_test["category"].tolist() == [3, 0]
    assert encoded_train["category"].dtype == np.dtype("int32")


def test_xgboost_imports_have_no_output_side_effects() -> None:
    existing = [path for path in HISTORICAL_OUTPUTS if path.exists()]
    before = {path: _sha256(path) for path in existing}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import src.models.xgboost_baseline; import src.train_xgboost_exp008",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert {path: _sha256(path) for path in existing} == before


def test_xgboost_small_fit_predict_smoke() -> None:
    X_raw, y, X_test_raw, features = _load_samples()
    assert y.nunique() == 2
    numeric = X_raw.select_dtypes(include=["number"]).columns.tolist()
    categorical = [column for column in features if column not in numeric]
    X, _, _ = migrated.ordinal_encode_categories(X_raw, X_test_raw, categorical)
    train_indices, valid_indices = train_test_split(
        np.arange(len(X)), test_size=0.2, random_state=42, stratify=y
    )

    smoke_params = {
        **migrated.MODEL_PARAMS,
        "n_estimators": 40,
        "n_jobs": 1,
    }
    model = XGBClassifier(**smoke_params, early_stopping_rounds=5)
    model.fit(
        X.iloc[train_indices],
        y.iloc[train_indices],
        eval_set=[(X.iloc[valid_indices], y.iloc[valid_indices])],
        verbose=False,
    )
    assert isinstance(model.best_iteration, int)
    assert model.best_iteration >= 0
    iteration_range = (0, model.best_iteration + 1)
    probabilities = model.predict_proba(
        X.iloc[valid_indices], iteration_range=iteration_range
    )[:, 1]

    assert probabilities.shape == (len(valid_indices),)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
