"""Compatibility, configuration and smoke checks for migrated EXP-039."""

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

from src.features.exact_values import ORIGINAL
from src.features.frequency import FREQUENCY_COLUMNS
from src.features.relational import RELATIONS, add_numeric_relations
from src.models import lightgbm_high_resolution as migrated
from src import train_exp039_high_bin_lgbm as legacy


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BASE_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "n_estimators": 10000,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}
FINAL_COLUMNS = [
    *ORIGINAL,
    "weekend_freq",
    "screen_freq",
    *RELATIONS,
]
HISTORICAL_OUTPUTS = (
    ROOT / "outputs" / "metrics" / "experiment_log.csv",
    ROOT / "outputs" / "metrics" / "experiment_log_v2.csv",
    ROOT / "outputs" / "metrics" / "exp039_highbin_metrics.txt",
    ROOT / "outputs" / "predictions" / "oof_exp039_lgbm_highbin_raw.csv",
    ROOT / "outputs" / "predictions" / "test_exp039_lgbm_highbin_raw.csv",
    ROOT / "outputs" / "predictions" / "oof_exp039_lgbm_highbin.csv",
    ROOT / "outputs" / "predictions" / "test_exp039_lgbm_highbin.csv",
    ROOT / "outputs" / "submissions" / "submission_exp039_highbin_lgbm_ensemble.csv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_sample(rows: int = 3000) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    if not (migrated.DATA / "train.csv").exists() or not (migrated.DATA / "test.csv").exists():
        pytest.skip("Competition train/test data is not available locally")
    train = pd.read_csv(migrated.DATA / "train.csv", nrows=rows)
    test = pd.read_csv(migrated.DATA / "test.csv", nrows=1000)
    return train, train[migrated.TARGET], test


def test_exp039_wrapper_reexports_public_api_and_feature_modules() -> None:
    top_level = importlib.import_module("train_exp039_high_bin_lgbm")
    for name in (
        "rss", "exact_key", "add_relations", "build_fold", "train_config",
        "summarize", "nested_two", "nested_triple", "main",
    ):
        assert getattr(legacy, name) is getattr(migrated, name)
        assert getattr(top_level, name) is getattr(migrated, name)
    assert migrated.ORIGINAL is ORIGINAL
    assert migrated.FREQ is FREQUENCY_COLUMNS
    assert migrated.RELATIONS is RELATIONS
    assert migrated.add_relations is add_numeric_relations


def test_exp039_final_configuration_and_historical_screen_are_preserved() -> None:
    assert migrated.BASE_PARAMS == EXPECTED_BASE_PARAMS
    assert migrated.FINAL_MAX_BIN == 4095
    assert migrated.FINAL_FEATURE_CFG == (("weekend_freq", "screen_freq"), True, tuple())
    assert migrated.FINAL_PARAM_UPDATES == {"num_leaves": 15, "min_child_samples": 100}
    source = Path(migrated.__file__).read_text(encoding="utf-8")
    assert "for mb in [255, 511, 1023, 2047, 4095]" in source
    assert "for leaves in [15, 31, 63]" in source
    assert "for child in [20, 50, 100]" in source
    assert "StratifiedKFold(5, shuffle=True, random_state=42)" in source
    assert "lgb.early_stopping(300, verbose=False)" in source
    assert "lgb.log_evaluation(0)" in source
    assert 'eval_metric="auc"' in source
    assert "num_iteration=best" in source
    assert set(migrated.CODES) == {
        "screen_value_code", "social_value_code", "weekend_value_code"
    }


def test_exp039_feature_parity_and_final_column_contract() -> None:
    report = pd.read_csv(
        ROOT / "outputs" / "reports" / "high_resolution_lgbm_feature_parity.csv"
    )
    assert len(report) == 60
    assert report.groupby("split").size().to_dict() == {
        "outer_train": 20, "outer_valid": 20, "test": 20
    }
    assert report["values_match"].all()
    assert report["nan_match"].all()
    assert report["dtype_match"].all()
    assert report["index_match"].all()
    assert report["order_match"].all()
    assert report["status"].eq("PASS").all()
    train_features = report[report["split"].eq("outer_train")].sort_values("position")
    assert train_features["feature"].tolist() == FINAL_COLUMNS
    assert "social_freq" not in FINAL_COLUMNS
    assert not any(name in FINAL_COLUMNS for name in migrated.CODES)


def test_exp040_remains_importable_and_does_not_depend_on_exp039_wrapper() -> None:
    exp040 = importlib.import_module("src.train_exp040_dual_repr_lgbm")
    assert exp040.FREQ == {"weekend_freq": "weekend_screen_time", "screen_freq": "daily_screen_time_hours"}
    source = Path(exp040.__file__).read_text(encoding="utf-8")
    assert "train_exp039_high_bin_lgbm" not in source


def test_exp039_import_from_arbitrary_cwd_has_no_side_effects(tmp_path: Path) -> None:
    existing = [path for path in HISTORICAL_OUTPUTS if path.exists()]
    before = {path: _sha256(path) for path in existing}
    script = ROOT / "src" / "train_exp039_high_bin_lgbm.py"
    code = (
        "import importlib.util; "
        f"s=importlib.util.spec_from_file_location('exp039_legacy', {str(script)!r}); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert {path: _sha256(path) for path in existing} == before


def test_exp039_high_bin_small_fit_predict_smoke() -> None:
    train, y, test = _load_sample()
    raw, raw_test = train[ORIGINAL].copy(), test[ORIGINAL].copy()
    X, X_test = migrated.prepare_aligned_categories(
        raw, raw_test, migrated.CATEGORICALS
    )
    train_indices, valid_indices = train_test_split(
        np.arange(len(X)), test_size=0.2, random_state=42, stratify=y
    )
    a, b, e = migrated.build_fold(
        X, X_test, raw, raw_test,
        train_indices, valid_indices, migrated.FINAL_FEATURE_CFG, True,
    )
    assert a.columns.tolist() == b.columns.tolist() == e.columns.tolist() == FINAL_COLUMNS
    assert a["weekend_freq"].dtype == b["screen_freq"].dtype == np.dtype("float32")
    assert all(pd.api.types.is_float_dtype(a[name]) for name in RELATIONS)

    params = {
        **migrated.BASE_PARAMS,
        "max_bin": migrated.FINAL_MAX_BIN,
        **migrated.FINAL_PARAM_UPDATES,
        "n_estimators": 40,
        "n_jobs": 1,
    }
    model = LGBMClassifier(**params)
    model.fit(
        a, y.iloc[train_indices],
        eval_set=[(b, y.iloc[valid_indices])],
        eval_metric="auc",
        categorical_feature=migrated.CATEGORICALS,
        callbacks=[lgb.early_stopping(5, verbose=False), lgb.log_evaluation(0)],
    )
    assert isinstance(model.best_iteration_, int) and model.best_iteration_ > 0
    probabilities = model.predict_proba(b, num_iteration=model.best_iteration_)[:, 1]
    assert probabilities.shape == (len(valid_indices),)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
