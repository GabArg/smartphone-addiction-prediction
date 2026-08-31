"""Threshold and region features introduced in EXP-012."""
from __future__ import annotations

import numpy as np
import pandas as pd

THRESHOLD_FEATURES = [
    "screen_gt_8", "screen_le_6", "social_gt_4", "social_le_4",
    "clear_positive_zone", "clear_negative_zone", "ambiguous_zone",
    "screen_dist_to_6", "screen_dist_to_8", "social_dist_to_4",
    "screen_abs_dist_to_6", "screen_abs_dist_to_8", "social_abs_dist_to_4",
    "screen_x_social", "screen_plus_social", "screen_minus_social",
    "screen_over_social", "social_over_screen", "min_abs_threshold_distance",
    "screen_mid_band", "social_near_threshold", "screen_near_threshold",
    "region_code",
]


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=numerator.index, dtype=np.float64)
    valid = denominator.notna() & denominator.ne(0)
    result.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return result.replace([np.inf, -np.inf], np.nan)


def indicator(condition: pd.Series, known: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=condition.index, dtype=np.float64)
    result.loc[known] = condition.loc[known].astype(np.float64)
    return result


def add_threshold_features(frame: pd.DataFrame) -> pd.DataFrame:
    engineered = frame.copy()
    screen = engineered["daily_screen_time_hours"]
    social = engineered["social_media_hours"]
    screen_known, social_known = screen.notna(), social.notna()
    engineered["screen_gt_8"] = indicator(screen > 8, screen_known)
    engineered["screen_le_6"] = indicator(screen <= 6, screen_known)
    engineered["social_gt_4"] = indicator(social > 4, social_known)
    engineered["social_le_4"] = indicator(social <= 4, social_known)
    positive_true = (screen_known & screen.gt(8)) | (social_known & social.gt(4))
    positive_false = screen_known & screen.le(8) & social_known & social.le(4)
    clear_positive = pd.Series(np.nan, index=frame.index, dtype=np.float64)
    clear_positive.loc[positive_true], clear_positive.loc[positive_false] = 1.0, 0.0
    negative_true = screen_known & screen.le(6) & social_known & social.le(4)
    negative_false = (screen_known & screen.gt(6)) | (social_known & social.gt(4))
    clear_negative = pd.Series(np.nan, index=frame.index, dtype=np.float64)
    clear_negative.loc[negative_true], clear_negative.loc[negative_false] = 1.0, 0.0
    ambiguous = pd.Series(np.nan, index=frame.index, dtype=np.float64)
    ambiguous.loc[clear_positive.eq(1) | clear_negative.eq(1)] = 0.0
    ambiguous.loc[clear_positive.eq(0) & clear_negative.eq(0)] = 1.0
    engineered["clear_positive_zone"] = clear_positive
    engineered["clear_negative_zone"] = clear_negative
    engineered["ambiguous_zone"] = ambiguous
    engineered["screen_dist_to_6"] = screen - 6
    engineered["screen_dist_to_8"] = screen - 8
    engineered["social_dist_to_4"] = social - 4
    engineered["screen_abs_dist_to_6"] = (screen - 6).abs()
    engineered["screen_abs_dist_to_8"] = (screen - 8).abs()
    engineered["social_abs_dist_to_4"] = (social - 4).abs()
    engineered["screen_x_social"] = screen * social
    engineered["screen_plus_social"] = screen + social
    engineered["screen_minus_social"] = screen - social
    engineered["screen_over_social"] = safe_divide(screen, social)
    engineered["social_over_screen"] = safe_divide(social, screen)
    engineered["min_abs_threshold_distance"] = engineered[
        ["screen_abs_dist_to_6", "screen_abs_dist_to_8", "social_abs_dist_to_4"]
    ].min(axis=1, skipna=True)
    engineered["screen_mid_band"] = indicator((screen > 6) & (screen <= 8), screen_known)
    engineered["social_near_threshold"] = indicator((social >= 3.5) & (social <= 4.5), social_known)
    engineered["screen_near_threshold"] = indicator((screen >= 5.5) & (screen <= 8.5), screen_known)
    region_code = pd.Series(np.nan, index=frame.index, dtype=np.float64)
    region_code.loc[clear_negative.eq(1)] = 0.0
    region_code.loc[ambiguous.eq(1)] = 1.0
    region_code.loc[clear_positive.eq(1)] = 2.0
    engineered["region_code"] = region_code
    return engineered
