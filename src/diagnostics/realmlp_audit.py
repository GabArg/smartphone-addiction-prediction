"""EXP-041B: read-only audit of the pytabkit 1.7.3 RealMLP setup.

This script never calls fit() or predict(). It can still audit the data and the
requested EXP-041 parameters when pytabkit is not installed locally.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
from pathlib import Path

import pandas as pd

TARGET = "addicted_label"
CATEGORICALS = ["gender", "stress_level", "academic_work_impact"]
REQUESTED = {
    "n_cv": 1,
    "n_refit": 0,
    "n_epochs": 256,
    "batch_size": 1024,
    "val_metric_name": "1-auc_ovr",
    "use_ls": False,
    "verbosity": 1,
    "device": "cuda",
    "random_state": 42,
}

# Pinned official defaults from pytabkit 1.7.3 DefaultParams.RealMLP_TD_CLASS.
TD_173 = {
    "hidden_sizes": [256, 256, 256], "max_one_hot_cat_size": 9,
    "embedding_size": 8, "weight_param": "ntk", "bias_lr_factor": 0.1,
    "act": "selu", "use_parametric_act": True, "act_lr_factor": 0.1,
    "block_str": "w-b-a-d", "p_drop": 0.15, "p_drop_sched": "flat_cos",
    "add_front_scale": True, "scale_lr_factor": 6.0,
    "bias_init_mode": "he+5", "weight_init_mode": "std", "wd": 0.02,
    "wd_sched": "flat_cos", "bias_wd_factor": 0.0, "use_ls": True,
    "ls_eps": 0.1, "num_emb_type": "pbld", "plr_sigma": 0.1,
    "plr_hidden_1": 16, "plr_hidden_2": 4, "plr_lr_factor": 0.1,
    "lr": 0.04,
    "tfms": ["one_hot", "median_center", "robust_scale", "smooth_clip", "embedding"],
    "n_epochs": 256, "lr_sched": "coslog4", "opt": "adam", "sq_mom": 0.95,
}
INTERFACE_DEFAULTS = {
    "n_cv": 1, "n_refit": 0, "n_repeats": 1, "val_fraction": 0.2,
    "verbosity": 0, "n_ens": 1, "ens_av_before_softmax": False,
    "calibration_method": None, "use_early_stopping": False,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=Path("data"))
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    requested = {**REQUESTED, "device": args.device}
    report = {
        "audit_only_no_training": True,
        "class": "pytabkit.RealMLP_TD_Classifier",
        "requested_constructor_params": requested,
        "official_td_1_7_3_defaults": TD_173,
        "effective_known_params": {**INTERFACE_DEFAULTS, **TD_173, **requested},
        "validation": {
            "fit_call": "fit(X_outer_train, y_outer_train, X_outer_val, y_outer_val, cat_col_names=CATEGORICALS)",
            "explicit_outer_validation": True,
            "internal_holdout_val_fraction_used": False,
            "best_epoch_selection": True,
            "early_stopping": False,
            "note": "All 256 epochs run, but prediction uses the best validation checkpoint.",
        },
        "ensemble_refit": {"n_cv": 1, "n_refit": 0, "n_repeats": 1, "n_ens": 1,
                           "models_per_outer_fold": 1, "calibration_method": None},
        "preprocessing": {
            "external_numeric": "outer-train median imputation, float32",
            "external_categorical": "category dtype with __MISSING__/__UNSEEN__ tokens",
            "internal_tfms": TD_173["tfms"], "num_emb_type": "pbld",
        },
        "available_official_variants": ["RealMLP_TD", "RealMLP_TD_S", "RealMLP_HPO"],
    }

    train_path = args.input_dir / "train.csv"
    if train_path.exists():
        df = pd.read_csv(train_path)
        feature_cols = [c for c in df.columns if c not in {"id", TARGET}]
        report["data"] = {
            "rows": len(df), "raw_feature_count": len(feature_cols),
            "continuous_count": len([c for c in feature_cols if c not in CATEGORICALS]),
            "categorical_count": len(CATEGORICALS),
            "categorical_columns": CATEGORICALS,
            "categorical_dtypes_on_csv_load": {c: str(df[c].dtype) for c in CATEGORICALS},
            "categorical_cardinalities_including_missing": {
                c: int(df[c].nunique(dropna=False)) for c in CATEGORICALS},
            "missing_by_column": {c: int(v) for c, v in df[feature_cols].isna().sum().items()},
            "target_dtype": str(df[TARGET].dtype),
            "target_values": sorted(df[TARGET].unique().tolist()),
        }

    try:
        from pytabkit import RealMLP_TD_Classifier
        from pytabkit.models.sklearn.default_params import DefaultParams

        sig = inspect.signature(RealMLP_TD_Classifier)
        accepted = set(sig.parameters)
        model = RealMLP_TD_Classifier(**requested)
        report["installed"] = {
            "version": importlib.metadata.version("pytabkit"),
            "signature": str(sig),
            "object_repr": repr(model),
            "get_params": model.get_params(deep=False),
            "runtime_td_defaults": DefaultParams.RealMLP_TD_CLASS,
            "unknown_requested_params": sorted(set(requested) - accepted),
            "ignored_or_unknown_params": [],
        }
        if hasattr(model, "get_config"):
            report["installed"]["get_config_effective"] = model.get_config()
    except ImportError as exc:
        report["installed"] = {
            "available": False, "reason": str(exc),
            "note": "Run this same script in the Kaggle environment after installing pytabkit==1.7.3.",
            "unknown_requested_params_from_official_signature": [],
        }

    print(json.dumps(report, indent=2, default=str, sort_keys=True))


if __name__ == "__main__":
    main()
