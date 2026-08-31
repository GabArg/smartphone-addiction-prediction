"""Parity and API checks for the centralized EXP-035 exact-value features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.exact_values import ORIGINAL, build_rep, safe_ratio, stringify


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ORIGINAL = [
    "age", "gender", "daily_screen_time_hours", "social_media_hours",
    "gaming_hours", "notifications_per_day", "app_opens_per_day",
    "sleep_hours", "work_study_hours", "weekend_screen_time",
    "stress_level", "academic_work_impact",
]
EXPECTED_COLUMNS = {
    "A": [*EXPECTED_ORIGINAL, "screen_social_exact"],
    "B": [
        *EXPECTED_ORIGINAL, "screen_social_exact",
        "screen_weekend_01", "social_weekend_01",
    ],
    "C": [
        *EXPECTED_ORIGINAL, "screen_social_exact",
        "screen_weekend_01", "social_weekend_01",
        "social_over_screen", "gaming_over_screen",
        "work_over_screen", "weekend_over_screen",
    ],
}


def _fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [20.0, np.nan, 40.0],
            "gender": ["Male", None, "Female"],
            "daily_screen_time_hours": [3.04, 0.0, np.nan],
            "social_media_hours": [1.26, np.nan, 4.0],
            "gaming_hours": [0.5, 2.0, np.nan],
            "notifications_per_day": [10.0, 20.0, np.nan],
            "app_opens_per_day": [5.0, 6.0, 7.0],
            "sleep_hours": [8.0, 7.5, 6.0],
            "work_study_hours": [4.0, 3.0, 2.0],
            "weekend_screen_time": [5.06, 4.0, 8.0],
            "stress_level": ["Low", "High", None],
            "academic_work_impact": ["No", None, "Yes"],
            "addicted_label": [0, 1, 1],
        },
        index=[8, 3, 1],
    )


def test_exact_value_variants_preserve_historical_names_and_order() -> None:
    frame = _fixture()
    assert ORIGINAL == EXPECTED_ORIGINAL
    for variant, expected in EXPECTED_COLUMNS.items():
        result = build_rep(frame, variant)
        assert result.columns.tolist() == expected
        assert result.index.equals(frame.index)
        assert "addicted_label" not in result
        assert all(str(dtype) == "string" for dtype in result.dtypes)


def test_string_missing_rounding_interactions_and_ratios_are_exact() -> None:
    frame = _fixture()
    result = build_rep(frame, "C")
    assert stringify(frame["age"]).tolist() == ["20", "__MISSING__", "40"]
    assert result.loc[8, "screen_social_exact"] == "3.04__1.26"
    assert result.loc[8, "screen_weekend_01"] == "3__5.1"
    assert result.loc[8, "social_weekend_01"] == "1.3__5.1"
    assert result.loc[8, "social_over_screen"] == "0.41"
    assert result.loc[3, "social_over_screen"] == "__MISSING__"
    assert result.loc[3, "gaming_over_screen"] == "__MISSING__"
    assert result.loc[1, "weekend_over_screen"] == "__MISSING__"
    ratio = safe_ratio(frame["social_media_hours"], frame["daily_screen_time_hours"])
    assert np.isfinite(ratio.loc[8])
    assert np.isnan(ratio.loc[3]) and np.isnan(ratio.loc[1])


def test_exact_value_feature_parity_report_is_complete() -> None:
    report = pd.read_csv(ROOT / "outputs" / "reports" / "exact_value_feature_parity.csv")
    assert len(report) == 47
    assert report.groupby("variant").size().to_dict() == {"A": 13, "B": 15, "C": 19}
    for column in (
        "values_match", "missing_token_match", "dtype_match", "index_match", "order_match"
    ):
        assert report[column].all()
    assert (report["original_cardinality"] == report["new_cardinality"]).all()
    assert report["status"].eq("PASS").all()
