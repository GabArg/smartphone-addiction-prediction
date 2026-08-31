"""Relational features validated in EXP-036/037/039."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.exact_values import build_rep, stringify

RELATIONS = {
    "social_over_screen": ("social_media_hours", "daily_screen_time_hours", "ratio"),
    "gaming_over_screen": ("gaming_hours", "daily_screen_time_hours", "ratio"),
    "work_over_screen": ("work_study_hours", "daily_screen_time_hours", "ratio"),
    "work_over_social": ("work_study_hours", "social_media_hours", "ratio"),
    "gaming_over_social": ("gaming_hours", "social_media_hours", "ratio"),
    "screen_minus_social": ("daily_screen_time_hours", "social_media_hours", "difference"),
}

FINAL_RATIO_RELATIONS = {
    name: (numerator, denominator)
    for name, (numerator, denominator, kind) in RELATIONS.items()
    if kind == "ratio"
}
FINAL_RELATIONAL_SPEC = [("screen_minus_social", 1)]


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=numerator.index, dtype=np.float64)
    valid = denominator.notna() & denominator.ne(0)
    result.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return result.replace([np.inf, -np.inf], np.nan)


def build_base_representation(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the exact-value variant B used as the EXP-036/037 base."""
    return build_rep(frame, "B")


def relational_value(frame: pd.DataFrame, name: str) -> pd.Series:
    """Compute one of the six relational values selected by EXP-036/037."""
    numerator, denominator, kind = RELATIONS[name]
    if kind == "ratio":
        return safe_ratio(frame[numerator], frame[denominator])
    return frame[numerator] - frame[denominator]


def add_categorical_relations(
    base: pd.DataFrame,
    raw: pd.DataFrame,
    *,
    include_screen_difference: bool = True,
) -> pd.DataFrame:
    """Add the five winning ratios and optional final difference as strings."""
    result = base.copy()
    for name in FINAL_RATIO_RELATIONS:
        result[name] = stringify(relational_value(raw, name), 2)
    if include_screen_difference:
        result["screen_minus_social"] = stringify(
            relational_value(raw, "screen_minus_social"), 1
        )
    return result


def add_numeric_relations(base: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Add the six continuous EXP-039 relations to an existing feature frame."""
    result = base.copy()
    for name, (numerator, denominator, kind) in RELATIONS.items():
        value = (safe_ratio(raw[numerator], raw[denominator])
                 if kind == "ratio" else raw[numerator] - raw[denominator])
        result[name] = value.to_numpy(np.float64)
    return result


def add_relational_features(frame: pd.DataFrame) -> pd.DataFrame:
    return add_numeric_relations(frame, frame)
