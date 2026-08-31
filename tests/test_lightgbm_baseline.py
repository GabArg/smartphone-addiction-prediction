"""Parity and smoke checks for the migrated EXP-004 LightGBM baseline."""

from __future__ import annotations

import hashlib
import importlib
import subprocess
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

from src import train_lightgbm_exp004 as legacy
from src.models import lightgbm_baseline as migrated


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
    "objective": "binary",
    "n_estimators": 5000,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "random_state": 42,
    "n_jobs": -1,
}
HISTORICAL_OUTPUTS = (
    ROOT / "outputs" / "metrics" / "experiment_log.csv",
    ROOT / "outputs" / "metrics" / "experiment_log_v2.csv",
    ROOT / "outputs" / "metrics" / "exp004_lightgbm_metrics.txt",
    ROOT / "outputs" / "predictions" / "oof_exp004_lightgbm.csv",
    ROOT / "outputs" / "predictions" / "test_exp004_lightgbm.csv",
    ROOT / "outputs" / "submissions" / "submission_exp004_lightgbm.csv",
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


def test_lightgbm_wrapper_reexports_public_api() -> None:
    top_level_legacy = importlib.import_module("train_lightgbm_exp004")
    for name in (
        "prepare_aligned_categories",
        "validate_submission",
        "normalize_exp003_oof",
        "read_exp003_auc",
        "update_experiment_log",
        "main",
    ):
        assert getattr(legacy, name) is getattr(migrated, name)
        assert getattr(top_level_legacy, name) is getattr(migrated, name)

    assert legacy.MODEL_PARAMS is migrated.MODEL_PARAMS
    assert legacy.EXPERIMENT_ID == "EXP-004"
    assert legacy.TARGET == "addicted_label"
    assert legacy.ID_COLUMN == "id"
    assert legacy.N_SPLITS == 5
    assert legacy.RANDOM_STATE == 42


def test_exp004_productive_configuration_is_preserved() -> None:
    assert migrated.MODEL_PARAMS == EXPECTED_MODEL_PARAMS
    for default_only in ("metric", "min_child_samples", "verbosity", "max_bin"):
        assert default_only not in migrated.MODEL_PARAMS

    source = Path(migrated.__file__).read_text(encoding="utf-8")
    assert "StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)" in source
    assert 'eval_metric="auc"' in source
    assert "lgb.early_stopping(200, verbose=False)" in source
    assert "lgb.log_evaluation(0)" in source
    assert "categorical_feature=categorical_columns" in source
    assert "num_iteration=best_iteration" in source

    assert migrated.SUBMISSION_PATH.name == "submission_exp004_lightgbm.csv"
    assert migrated.METRICS_PATH.name == "exp004_lightgbm_metrics.txt"
    assert migrated.OOF_PATH.name == "oof_exp004_lightgbm.csv"
    assert migrated.TEST_PREDICTIONS_PATH.name == "test_exp004_lightgbm.csv"
    assert migrated.LOG_PATH.name == "experiment_log.csv"


def test_exp004_feature_order_and_aligned_categories_are_preserved() -> None:
    X_raw, _, X_test_raw, features = _load_samples()
    assert features == EXPECTED_FEATURES
    numeric = X_raw.select_dtypes(include=["number"]).columns.tolist()
    categorical = [column for column in features if column not in numeric]
    assert categorical == EXPECTED_CATEGORICAL

    X, X_test = migrated.prepare_aligned_categories(X_raw, X_test_raw, categorical)
    assert X.columns.tolist() == features
    assert X_test.columns.tolist() == features
    assert X[numeric].equals(X_raw[numeric])
    assert X_test[numeric].equals(X_test_raw[numeric])
    for column in categorical:
        train_values = X_raw[column].fillna("__MISSING__").astype(str)
        test_values = X_test_raw[column].fillna("__MISSING__").astype(str)
        expected_categories = pd.Index(
            pd.concat([train_values, test_values], ignore_index=True).unique()
        )
        assert isinstance(X[column].dtype, pd.CategoricalDtype)
        assert X[column].cat.categories.equals(expected_categories)
        assert X_test[column].cat.categories.equals(expected_categories)
        assert not X[column].isna().any()
        assert not X_test[column].isna().any()


def test_lightgbm_imports_have_no_output_side_effects() -> None:
    existing = [path for path in HISTORICAL_OUTPUTS if path.exists()]
    before = {path: _sha256(path) for path in existing}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import src.models.lightgbm_baseline; import src.train_lightgbm_exp004",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert {path: _sha256(path) for path in existing} == before


def test_lightgbm_small_fit_predict_smoke() -> None:
    X_raw, y, X_test_raw, features = _load_samples()
    assert y.nunique() == 2
    numeric = X_raw.select_dtypes(include=["number"]).columns.tolist()
    categorical = [column for column in features if column not in numeric]
    X, _ = migrated.prepare_aligned_categories(X_raw, X_test_raw, categorical)
    train_indices, valid_indices = train_test_split(
        np.arange(len(X)), test_size=0.2, random_state=42, stratify=y
    )

    smoke_params = {
        **migrated.MODEL_PARAMS,
        "n_estimators": 30,
        "n_jobs": 1,
    }
    model = LGBMClassifier(**smoke_params)
    model.fit(
        X.iloc[train_indices],
        y.iloc[train_indices],
        eval_set=[(X.iloc[valid_indices], y.iloc[valid_indices])],
        eval_metric="auc",
        categorical_feature=categorical,
        callbacks=[lgb.early_stopping(5, verbose=False), lgb.log_evaluation(0)],
    )
    assert isinstance(model.best_iteration_, int)
    assert model.best_iteration_ > 0
    probabilities = model.predict_proba(
        X.iloc[valid_indices], num_iteration=model.best_iteration_
    )[:, 1]

    assert probabilities.shape == (len(valid_indices),)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
