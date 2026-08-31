"""Fold-safe frequency encoding used by EXP-039."""
from __future__ import annotations

import numpy as np
import pandas as pd

FREQUENCY_COLUMNS = {
    "screen_freq": "daily_screen_time_hours",
    "social_freq": "social_media_hours",
    "weekend_freq": "weekend_screen_time",
    "work_freq": "work_study_hours",
    "gaming_freq": "gaming_hours",
}


def exact_key(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        values = series.to_numpy(np.float64)
        return pd.Series(values, index=series.index).map(
            lambda value: "__MISSING__" if pd.isna(value)
            else np.format_float_positional(value, trim="-")
        ).astype(str)
    return series.fillna("__MISSING__").astype(str)


def fit_frequency_map(series: pd.Series) -> pd.Series:
    keys = exact_key(series)
    return keys.value_counts(dropna=False) / len(keys)


def apply_frequency_map(series: pd.Series, frequencies: pd.Series) -> pd.Series:
    return exact_key(series).map(frequencies).fillna(0).astype(np.float32)


def add_frequency_features(
    fit_frame: pd.DataFrame,
    transform_frame: pd.DataFrame,
    feature_names: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    result = transform_frame.copy()
    for name in feature_names:
        column = FREQUENCY_COLUMNS[name]
        result[name] = apply_frequency_map(transform_frame[column], fit_frequency_map(fit_frame[column]))
    return result
