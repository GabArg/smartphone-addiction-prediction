"""Fold-safety and missing/unseen checks for EXP-039 frequency features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.frequency import (
    FREQUENCY_COLUMNS,
    add_frequency_features,
    apply_frequency_map,
    exact_key,
    fit_frequency_map,
)


def test_frequency_feature_catalog_preserves_historical_candidates() -> None:
    assert FREQUENCY_COLUMNS == {
        "screen_freq": "daily_screen_time_hours",
        "social_freq": "social_media_hours",
        "weekend_freq": "weekend_screen_time",
        "work_freq": "work_study_hours",
        "gaming_freq": "gaming_hours",
    }


def test_frequency_mapping_is_fit_only_on_supplied_outer_train() -> None:
    outer_train = pd.Series([1.0, 1.0, 2.0, np.nan], index=[10, 11, 12, 13])
    outer_valid = pd.Series([1.0, 2.0, 3.0, np.nan], index=[20, 21, 22, 23])
    mapping = fit_frequency_map(outer_train)
    transformed = apply_frequency_map(outer_valid, mapping)
    assert mapping.to_dict() == {"1": 0.5, "2": 0.25, "__MISSING__": 0.25}
    assert transformed.tolist() == [0.5, 0.25, 0.0, 0.25]
    assert transformed.dtype == np.dtype("float32")
    assert transformed.index.equals(outer_valid.index)
    assert "3" not in mapping


def test_frequency_builder_preserves_index_and_uses_fit_frame_only() -> None:
    fit = pd.DataFrame(
        {"daily_screen_time_hours": [1.0, 1.0, 2.0], "weekend_screen_time": [4.0, 5.0, 5.0]},
        index=[5, 6, 7],
    )
    transform = pd.DataFrame(
        {"daily_screen_time_hours": [1.0, 9.0], "weekend_screen_time": [5.0, 8.0]},
        index=[20, 30],
    )
    result = add_frequency_features(fit, transform, ["weekend_freq", "screen_freq"])
    assert result.index.equals(transform.index)
    assert result.columns.tolist() == [
        "daily_screen_time_hours", "weekend_screen_time", "weekend_freq", "screen_freq"
    ]
    assert np.allclose(result["weekend_freq"], [2 / 3, 0])
    assert np.allclose(result["screen_freq"], [2 / 3, 0])
    assert exact_key(pd.Series([1.0, np.nan])).tolist() == ["1", "__MISSING__"]
