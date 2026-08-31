"""Target-independent exact-value representations introduced in EXP-035."""

from __future__ import annotations

import numpy as np
import pandas as pd


ORIGINAL = [
    "age",
    "gender",
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "sleep_hours",
    "work_study_hours",
    "weekend_screen_time",
    "stress_level",
    "academic_work_impact",
]


def stringify(series: pd.Series, decimals: int | None = None) -> pd.Series:
    """Convert values to the exact string representation used by EXP-035."""
    if pd.api.types.is_numeric_dtype(series):
        values = series.to_numpy(np.float64)
        if decimals is not None:
            values = np.round(values, decimals)
        output = pd.Series(values, index=series.index).map(
            lambda value: (
                "__MISSING__"
                if pd.isna(value)
                else np.format_float_positional(value, trim="-")
            )
        )
        return output.astype("string")
    return series.fillna("__MISSING__").astype("string")


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide while preserving EXP-035 missing and zero-denominator behavior."""
    output = pd.Series(np.nan, index=numerator.index, dtype=float)
    valid = denominator.notna() & denominator.ne(0)
    output.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return output.replace([np.inf, -np.inf], np.nan)


def build_rep(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Build historical exact-value variant A, B or C without using the target."""
    representation = pd.DataFrame(index=frame.index)
    for column in ORIGINAL:
        representation[column] = stringify(frame[column])
    representation["screen_social_exact"] = (
        representation.daily_screen_time_hours
        + "__"
        + representation.social_media_hours
    )
    if variant in {"B", "C"}:
        screen = stringify(frame.daily_screen_time_hours, 1)
        weekend = stringify(frame.weekend_screen_time, 1)
        social = stringify(frame.social_media_hours, 1)
        representation["screen_weekend_01"] = screen + "__" + weekend
        representation["social_weekend_01"] = social + "__" + weekend
    if variant == "C":
        for name, numerator in [
            ("social_over_screen", frame.social_media_hours),
            ("gaming_over_screen", frame.gaming_hours),
            ("work_over_screen", frame.work_study_hours),
            ("weekend_over_screen", frame.weekend_screen_time),
        ]:
            representation[name] = stringify(
                safe_ratio(numerator, frame.daily_screen_time_hours), 2
            )
    return representation
