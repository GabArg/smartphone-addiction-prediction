"""Parity and API checks for centralized EXP-036/037 relational features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.relational import (
    FINAL_RATIO_RELATIONS,
    FINAL_RELATIONAL_SPEC,
    RELATIONS,
    add_categorical_relations,
    build_base_representation,
    relational_value,
)


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_RATIOS = [
    "social_over_screen",
    "gaming_over_screen",
    "work_over_screen",
    "work_over_social",
    "gaming_over_social",
]


def _fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [20.0, 30.0, 40.0],
            "gender": ["Male", "Female", None],
            "daily_screen_time_hours": [4.0, 0.0, np.nan],
            "social_media_hours": [2.0, np.nan, 4.0],
            "gaming_hours": [1.0, 2.0, np.nan],
            "notifications_per_day": [10.0, 20.0, 30.0],
            "app_opens_per_day": [5.0, 6.0, 7.0],
            "sleep_hours": [8.0, 7.0, 6.0],
            "work_study_hours": [3.0, 4.0, 5.0],
            "weekend_screen_time": [5.0, 6.0, 7.0],
            "stress_level": ["Low", "High", None],
            "academic_work_impact": ["No", "Yes", None],
        },
        index=[9, 4, 1],
    )


def test_final_relational_api_has_exact_winners_and_order() -> None:
    assert list(FINAL_RATIO_RELATIONS) == EXPECTED_RATIOS
    assert FINAL_RELATIONAL_SPEC == [("screen_minus_social", 1)]
    assert list(RELATIONS) == [*EXPECTED_RATIOS, "screen_minus_social"]
    frame = _fixture()
    base = build_base_representation(frame)
    result = add_categorical_relations(base, frame)
    assert result.index.equals(frame.index)
    assert result.columns.tolist() == [
        *base.columns, *EXPECTED_RATIOS, "screen_minus_social"
    ]
    assert all(str(result[name].dtype) == "string" for name in [*EXPECTED_RATIOS, "screen_minus_social"])


def test_relational_rounding_and_missing_contract() -> None:
    frame = _fixture()
    result = add_categorical_relations(build_base_representation(frame), frame)
    assert result.loc[9, "social_over_screen"] == "0.5"
    assert result.loc[9, "work_over_social"] == "1.5"
    assert result.loc[9, "gaming_over_social"] == "0.5"
    assert result.loc[9, "screen_minus_social"] == "2"
    assert result.loc[4, "social_over_screen"] == "__MISSING__"
    assert result.loc[4, "gaming_over_screen"] == "__MISSING__"
    assert result.loc[1, "screen_minus_social"] == "__MISSING__"
    assert np.isnan(relational_value(frame, "social_over_screen").loc[4])


def test_relational_feature_parity_report_is_complete() -> None:
    report = pd.read_csv(ROOT / "outputs" / "reports" / "relational_feature_parity.csv")
    assert len(report) == 11
    assert report.groupby("experiment").size().to_dict() == {"EXP-036": 5, "EXP-037": 6}
    for column in ("values_match", "missing_match", "dtype_match", "index_match", "order_match"):
        assert report[column].all()
    assert (report["original_cardinality"] == report["new_cardinality"]).all()
    assert report["status"].eq("PASS").all()
