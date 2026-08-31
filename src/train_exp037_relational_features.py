"""Compatibility wrapper for the migrated EXP-037 relational Logistic model."""

from pathlib import Path
import sys

_PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_FOR_IMPORTS))

from src.models.logistic_relational import (
    BASE_R,
    DATA,
    DIFF,
    ENS036,
    EXP027,
    EXP028,
    EXP036,
    INV,
    METRICS,
    ORIGINAL,
    OUT,
    PRED,
    REPORTS,
    ROOT,
    SUB,
    WEIGHTS,
    add_features,
    base_rep,
    ckey,
    feature_value,
    fit_cv,
    fs,
    load_oof,
    main,
    nested_blend,
    nested_triple,
    rank,
    rss,
    safe_ratio,
    stringify,
)

__all__ = [
    "BASE_R",
    "DATA",
    "DIFF",
    "ENS036",
    "EXP027",
    "EXP028",
    "EXP036",
    "INV",
    "METRICS",
    "ORIGINAL",
    "OUT",
    "PRED",
    "REPORTS",
    "ROOT",
    "SUB",
    "WEIGHTS",
    "add_features",
    "base_rep",
    "ckey",
    "feature_value",
    "fit_cv",
    "fs",
    "load_oof",
    "main",
    "nested_blend",
    "nested_triple",
    "rank",
    "rss",
    "safe_ratio",
    "stringify",
]


if __name__ == "__main__":
    main()
