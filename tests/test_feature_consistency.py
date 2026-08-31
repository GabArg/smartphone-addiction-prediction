from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.frequency import add_frequency_features, apply_frequency_map, fit_frequency_map
from src.features.relational import RELATIONS, add_relational_features
from src.features.thresholds import THRESHOLD_FEATURES, add_threshold_features
from src.project_paths import DATA_DIR, REPORTS_DIR
from src.train_exp035_exact_value_logistic import stringify
from src.train_exp039_high_bin_lgbm import add_relations as original_add_relations
from src.train_xgboost_exp012_threshold_features import (
    NEW_FEATURES as ORIGINAL_THRESHOLD_FEATURES,
    add_threshold_features as original_add_threshold_features,
)

ROWS = 4096


def sample_data():
    paths = DATA_DIR / "train.csv", DATA_DIR / "test.csv"
    if not all(path.exists() for path in paths): pytest.skip("Competition data is not available locally")
    train = pd.read_csv(paths[0]).sample(ROWS, random_state=42).sort_index().reset_index(drop=True)
    test = pd.read_csv(paths[1]).sample(ROWS, random_state=42).sort_index().reset_index(drop=True)
    return train, test


def compare_series(feature, source, module, expected, actual):
    expected_values = expected.to_numpy()
    actual_values = actual.to_numpy()
    expected_nan, actual_nan = pd.isna(expected_values), pd.isna(actual_values)
    nan_match = bool(np.array_equal(expected_nan, actual_nan))
    valid = ~expected_nan & ~actual_nan
    diff = float(np.max(np.abs(expected_values[valid].astype(float) - actual_values[valid].astype(float)))) if valid.any() else 0.0
    return {"feature": feature, "source_script": source, "new_module": module,
            "rows_checked": len(expected), "max_abs_diff": diff,
            "nan_match": nan_match, "status": "PASS" if nan_match and diff <= 1e-12 else "FAIL"}


def parity_rows():
    train, test = sample_data()
    rows = []
    old_thresholds, new_thresholds = original_add_threshold_features(train), add_threshold_features(train)
    assert THRESHOLD_FEATURES == ORIGINAL_THRESHOLD_FEATURES
    for feature in THRESHOLD_FEATURES:
        rows.append(compare_series(feature, "train_xgboost_exp012_threshold_features.py",
                                   "src.features.thresholds", old_thresholds[feature], new_thresholds[feature]))
    base = train[["age"]].copy()
    old_relations = original_add_relations(base, train.reset_index(drop=True))
    new_relations = add_relational_features(train)
    for feature in RELATIONS:
        rows.append(compare_series(feature, "train_exp039_high_bin_lgbm.py",
                                   "src.features.relational", old_relations[feature], new_relations[feature]))
    for name, column in {"screen_freq": "daily_screen_time_hours", "weekend_freq": "weekend_screen_time"}.items():
        keys = stringify(train[column]).astype(str)
        original_map = keys.value_counts(dropna=False) / len(keys)
        expected = stringify(test[column]).astype(str).map(original_map).fillna(0).astype(np.float32)
        actual = apply_frequency_map(test[column], fit_frequency_map(train[column]))
        rows.append(compare_series(name, "train_exp039_high_bin_lgbm.py",
                                   "src.features.frequency", expected, actual))
    return rows


def test_feature_builders_have_train_test_consistency():
    train, test = sample_data()
    train_features = train.drop(columns="addicted_label")
    assert list(add_threshold_features(train_features).columns) == list(add_threshold_features(test).columns)
    assert list(add_relational_features(train_features).columns) == list(add_relational_features(test).columns)
    names = ["screen_freq", "weekend_freq"]
    assert list(add_frequency_features(train_features, train_features, names).columns) == list(add_frequency_features(train_features, test, names).columns)


def test_feature_parity_and_write_report():
    rows = parity_rows()
    report = REPORTS_DIR / "refactor_feature_parity.csv"
    pd.DataFrame(rows).to_csv(report, index=False)
    assert rows and all(row["status"] == "PASS" for row in rows)
