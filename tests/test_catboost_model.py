"""Parity and smoke checks for the migrated EXP-003 CatBoost model."""

from __future__ import annotations

import hashlib
import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

from src import train_catboost_exp003 as legacy
from src.models import catboost_model as migrated


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
    "iterations": 4000,
    "learning_rate": 0.05,
    "depth": 7,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "random_seed": 42,
    "verbose": False,
    "allow_writing_files": False,
}
HISTORICAL_OUTPUTS = (
    ROOT / "outputs" / "metrics" / "experiment_log.csv",
    ROOT / "outputs" / "metrics" / "experiment_log_v2.csv",
    ROOT / "outputs" / "metrics" / "exp003_catboost_metrics.txt",
    ROOT / "outputs" / "metrics" / "exp003_catboost_oof.csv",
    ROOT / "outputs" / "predictions" / "oof_exp003_catboost.csv",
    ROOT / "outputs" / "submissions" / "submission_exp003_catboost_4000.csv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_sample(rows: int = 1500) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    train_path = migrated.DATA_DIR / "train.csv"
    if not train_path.exists():
        pytest.skip("Competition train.csv is not available locally")
    train = pd.read_csv(train_path, nrows=rows)
    features = [
        column for column in train.columns if column not in {migrated.ID_COLUMN, migrated.TARGET}
    ]
    return train[features], train[migrated.TARGET], features


def test_wrapper_reexports_exp003_public_api() -> None:
    top_level_legacy = importlib.import_module("train_catboost_exp003")
    for name in (
        "prepare_categorical_missing",
        "parse_metrics",
        "validate_submission",
        "update_experiment_log",
        "main",
    ):
        assert getattr(legacy, name) is getattr(migrated, name)
        assert getattr(top_level_legacy, name) is getattr(migrated, name)

    assert legacy.MODEL_PARAMS is migrated.MODEL_PARAMS
    assert legacy.EXPERIMENT_ID == "EXP-003"
    assert legacy.TARGET == "addicted_label"
    assert legacy.ID_COLUMN == "id"
    assert legacy.N_SPLITS == 5
    assert legacy.RANDOM_SEED == 42


def test_exp003_productive_configuration_is_preserved() -> None:
    assert migrated.MODEL_PARAMS == EXPECTED_MODEL_PARAMS
    assert migrated.N_SPLITS == 5
    assert migrated.RANDOM_SEED == 42
    source = Path(migrated.__file__).read_text(encoding="utf-8")
    assert "StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)" in source
    assert "early_stopping_rounds=200" in source
    assert "cat_features=categorical_columns" in source

    assert migrated.SUBMISSION_PATH.name == "submission_exp003_catboost_4000.csv"
    assert migrated.METRICS_PATH.name == "exp003_catboost_metrics.txt"
    assert migrated.OOF_PATH.name == "exp003_catboost_oof.csv"
    assert migrated.EXP002_METRICS_PATH.name == "exp002_catboost_metrics.txt"
    assert migrated.LOG_PATH.name == "experiment_log.csv"


def test_feature_order_and_categorical_preparation_are_preserved() -> None:
    X_raw, _, features = _load_sample()
    assert features == EXPECTED_FEATURES
    numeric = X_raw.select_dtypes(include=["number"]).columns.tolist()
    categorical = [column for column in features if column not in numeric]
    assert categorical == EXPECTED_CATEGORICAL

    prepared = migrated.prepare_categorical_missing(X_raw, categorical)
    assert prepared.columns.tolist() == features
    assert prepared[numeric].equals(X_raw[numeric])
    assert not prepared[categorical].isna().any().any()
    for column in categorical:
        expected = X_raw[column].fillna("__MISSING__").astype(str)
        assert prepared[column].dtype == expected.dtype
        assert prepared[column].equals(expected)


def test_catboost_imports_have_no_output_side_effects() -> None:
    existing = [path for path in HISTORICAL_OUTPUTS if path.exists()]
    before = {path: _sha256(path) for path in existing}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import src.models.catboost_model; import src.train_catboost_exp003",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert {path: _sha256(path) for path in existing} == before


def test_catboost_small_fit_predict_smoke() -> None:
    X_raw, y, features = _load_sample(rows=1500)
    assert y.nunique() == 2
    numeric = X_raw.select_dtypes(include=["number"]).columns.tolist()
    categorical = [column for column in features if column not in numeric]
    X = migrated.prepare_categorical_missing(X_raw, categorical)
    train_indices, valid_indices = train_test_split(
        np.arange(len(X)), test_size=0.2, random_state=42, stratify=y
    )

    smoke_params = {
        **migrated.MODEL_PARAMS,
        "iterations": 12,
        "thread_count": 1,
    }
    model = CatBoostClassifier(**smoke_params)
    model.fit(
        X.iloc[train_indices],
        y.iloc[train_indices],
        cat_features=categorical,
        eval_set=(X.iloc[valid_indices], y.iloc[valid_indices]),
        early_stopping_rounds=5,
        verbose=False,
    )
    probabilities = model.predict_proba(X.iloc[valid_indices])[:, 1]

    assert probabilities.shape == (len(valid_indices),)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
