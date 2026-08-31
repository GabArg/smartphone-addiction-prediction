"""Compatibility wrapper for the migrated EXP-035 exact-value Logistic model."""

from pathlib import Path
import sys

_PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_FOR_IMPORTS))

from src.models.logistic_exact_values import (
    CS,
    DATA,
    EXP027,
    EXP028,
    METRICS,
    ORIGINAL,
    OUT,
    PRED,
    REPORTS,
    ROOT,
    SUB,
    TRIPLE_WEIGHTS,
    WEIGHTS,
    build_rep,
    finalize_existing,
    fit_fold,
    folds_auc,
    full_variant,
    load_oof,
    main,
    nested_blend,
    nested_choice,
    rank,
    rss,
    safe_ratio,
    stringify,
)

__all__ = [
    "CS",
    "DATA",
    "EXP027",
    "EXP028",
    "METRICS",
    "ORIGINAL",
    "OUT",
    "PRED",
    "REPORTS",
    "ROOT",
    "SUB",
    "TRIPLE_WEIGHTS",
    "WEIGHTS",
    "build_rep",
    "finalize_existing",
    "fit_fold",
    "folds_auc",
    "full_variant",
    "load_oof",
    "main",
    "nested_blend",
    "nested_choice",
    "rank",
    "rss",
    "safe_ratio",
    "stringify",
]


if __name__ == "__main__":
    if "--finalize-existing" in sys.argv:
        finalize_existing()
    else:
        main()
