"""Compatibility wrapper for the migrated EXP-039 high-resolution LightGBM."""

from pathlib import Path
import sys

_PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_FOR_IMPORTS))

from src.models.lightgbm_high_resolution import (
    BASE_PARAMS,
    CATEGORICALS,
    CODES,
    DATA,
    EXP037_ENS,
    FINAL_FEATURE_CFG,
    FINAL_MAX_BIN,
    FINAL_PARAM_UPDATES,
    FREQ,
    ID,
    METRICS,
    ORIGINAL,
    OUT,
    PRED,
    RELATIONS,
    REPORTS,
    ROOT,
    SUB,
    TARGET,
    add_relations,
    apply_frequency_map,
    build_fold,
    exact_key,
    fit_frequency_map,
    load_oof,
    main,
    nested_triple,
    nested_two,
    prepare_aligned_categories,
    rank,
    rss,
    summarize,
    train_config,
)

__all__ = [
    "BASE_PARAMS",
    "CATEGORICALS",
    "CODES",
    "DATA",
    "EXP037_ENS",
    "FINAL_FEATURE_CFG",
    "FINAL_MAX_BIN",
    "FINAL_PARAM_UPDATES",
    "FREQ",
    "ID",
    "METRICS",
    "ORIGINAL",
    "OUT",
    "PRED",
    "RELATIONS",
    "REPORTS",
    "ROOT",
    "SUB",
    "TARGET",
    "add_relations",
    "apply_frequency_map",
    "build_fold",
    "exact_key",
    "fit_frequency_map",
    "load_oof",
    "main",
    "nested_triple",
    "nested_two",
    "prepare_aligned_categories",
    "rank",
    "rss",
    "summarize",
    "train_config",
]


if __name__ == "__main__":
    main()
