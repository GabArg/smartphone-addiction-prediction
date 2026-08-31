"""EXP-031: ExtraTrees diversity experiment on exact EXP-016 threshold features."""

from __future__ import annotations

import gc
import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from joblib import parallel_config
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from train_xgboost_exp012_threshold_features import NEW_FEATURES, add_threshold_features


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
PRED = OUT / "predictions"
REPORTS = OUT / "reports"
METRICS = OUT / "metrics"
SUB = OUT / "submissions"

CONFIGS = {
    "baseline": dict(min_samples_leaf=2, max_features="sqrt"),
    "ET-A": dict(min_samples_leaf=5, max_features="sqrt"),
    "ET-B": dict(min_samples_leaf=2, max_features=.7),
}
BASE_PARAMS = dict(n_estimators=800, max_depth=None, min_samples_split=2, bootstrap=False,
                   class_weight=None, random_state=42, n_jobs=-1)
WEIGHTS = [.90, .925, .95, .975, .98, .99]
EXP027 = .965919188052602
EXP028 = .9659307051390922


def to_float32(x):
    return x.astype(np.float32, copy=False)


def load_oof(path, train):
    df = pd.read_csv(path)
    if not df.id.equals(train.id) or not df.y_true.equals(train.addicted_label) or df.oof_prediction.isna().any():
        raise ValueError(f"OOF invalida: {path.name}")
    return df.oof_prediction.to_numpy(np.float64)


def normalized_rank(values):
    r = pd.Series(values).rank(method="average").to_numpy(np.float64)
    return (r - 1) / (len(r) - 1)


def fold_aucs(y, p, splits):
    return [float(roc_auc_score(y[va], p[va])) for _, va in splits]


def aggregate_importance(names, values):
    agg = {}
    for name, value in zip(names, values):
        clean = name.split("__", 1)[-1]
        if name.startswith("cat__"):
            matches = [c for c in ["gender", "stress_level", "academic_work_impact"] if clean.startswith(c + "_")]
            base = matches[0] if matches else clean
        else:
            base = clean
        agg[base] = agg.get(base, 0.) + float(value)
    return agg


def nested_blend(y, base, extra, splits, base_name):
    candidates = []
    rank_base = normalized_rank(base)
    rank_extra = normalized_rank(extra)
    for method in ["probability", "rank"]:
        b = base if method == "probability" else rank_base
        e = extra if method == "probability" else rank_extra
        for w in WEIGHTS:
            p = w * b + (1 - w) * e
            candidates.append({"base": base_name, "method": method, "base_weight": w,
                               "global_auc_diagnostic": roc_auc_score(y, p)})
    nested = np.zeros(len(y), dtype=np.float64)
    choices = []
    for fold, (tr, va) in enumerate(splits, 1):
        best = None
        for method in ["probability", "rank"]:
            b = base if method == "probability" else rank_base
            e = extra if method == "probability" else rank_extra
            for w in WEIGHTS:
                p = w * b + (1 - w) * e
                auc = roc_auc_score(y[tr], p[tr])
                item = (auc, w, method, p)
                if best is None or item[0] > best[0] + 1e-15 or (abs(item[0] - best[0]) <= 1e-15 and w > best[1]):
                    best = item
        nested[va] = best[3][va]
        choices.append({"fold": fold, "base": base_name, "method": best[2], "base_weight": best[1],
                        "selection_auc": best[0]})
    fs = fold_aucs(y, nested, splits)
    return nested, choices, {"base": base_name, "auc": float(roc_auc_score(y, nested)), "folds": fs,
                             "std": float(np.std(fs)), "improved_folds": int(np.sum(np.asarray(fs) > np.asarray(fold_aucs(y, base, splits)))),
                             "choices": choices}, candidates


def update_log(auc, base_name, choices):
    path = METRICS / "experiment_log.csv"
    cols = ["experiment_id", "datetime", "model", "features", "cv_strategy", "cv_roc_auc", "kaggle_score", "notes"]
    log = pd.read_csv(path, dtype=str, keep_default_na=False)
    if log.columns.tolist() != cols:
        raise ValueError("experiment_log.csv inesperado")
    log = log.loc[~log.experiment_id.eq("EXP-031")]
    summary = ";".join(f"F{x['fold']}:{x['method']}/w{x['base_weight']}" for x in choices)
    row = {"experiment_id": "EXP-031", "datetime": datetime.now().astimezone().isoformat(timespec="seconds"),
           "model": "ExtraTrees_Ensemble", "features": "exp012_threshold_region_features",
           "cv_strategy": "Nested_OOF_blend", "cv_roc_auc": f"{auc:.8f}", "kaggle_score": "",
           "notes": f"base={base_name}; {summary}"}
    pd.concat([log, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)


def main():
    total_start = perf_counter()
    for d in [PRED, REPORTS, METRICS, SUB]:
        d.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    original = [c for c in train if c not in {"id", "addicted_label"}]
    raw = add_threshold_features(train[original])
    raw_test = add_threshold_features(test[original])
    if raw.columns.tolist() != raw_test.columns.tolist() or len(raw.columns) != len(original) + len(NEW_FEATURES):
        raise ValueError("Features distintas de EXP-016")
    cats = [c for c in original if not pd.api.types.is_numeric_dtype(raw[c])]
    nums = [c for c in raw if c not in cats]
    y = train.addicted_label.to_numpy()
    splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(raw, y))

    oofs = {name: np.zeros(len(train), np.float64) for name in CONFIGS}
    tests = {name: np.zeros(len(test), np.float64) for name in CONFIGS}
    fold_rows, importances = [], {name: [] for name in CONFIGS}
    for fold, (tr, va) in enumerate(splits, 1):
        prep = ColumnTransformer([
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("float32", FunctionTransformer(to_float32, feature_names_out="one-to-one"))]), nums),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32))]), cats),
        ], sparse_threshold=0.0)
        tprep = perf_counter()
        Xtr = prep.fit_transform(raw.iloc[tr])
        Xva = prep.transform(raw.iloc[va])
        Xt = prep.transform(raw_test)
        prep_seconds = perf_counter() - tprep
        names = prep.get_feature_names_out()
        for config, changes in CONFIGS.items():
            start = perf_counter()
            params = dict(BASE_PARAMS); params.update(changes)
            model = ExtraTreesClassifier(**params)
            # Preserve n_jobs=-1 while sharing X and forest memory between workers.
            with parallel_config(backend="threading", n_jobs=-1, require="sharedmem"):
                model.fit(Xtr, y[tr])
                pv = model.predict_proba(Xva)[:, 1].astype(np.float64)
                pt = model.predict_proba(Xt)[:, 1].astype(np.float64)
            oofs[config][va] = pv; tests[config] += pt / 5
            auc = float(roc_auc_score(y[va], pv))
            seconds = perf_counter() - start
            importances[config].append(aggregate_importance(names, model.feature_importances_))
            fold_rows.append({"config": config, "fold": fold, "auc": auc, "training_seconds": seconds,
                              "preprocessing_seconds": prep_seconds})
            pd.DataFrame(fold_rows).to_csv(REPORTS / "exp031_extratrees_fold_progress.csv", index=False)
            print(f"fold={fold} config={config} auc={auc:.8f} seconds={seconds:.1f}", flush=True)
            del model, pv, pt
            gc.collect()
        del Xtr, Xva, Xt, prep
        gc.collect()

    variant_rows = []
    for config in CONFIGS:
        fs = fold_aucs(y, oofs[config], splits)
        variant_rows.append({"config": config, "oof_auc": roc_auc_score(y, oofs[config]), "fold1": fs[0],
                             "fold2": fs[1], "fold3": fs[2], "fold4": fs[3], "fold5": fs[4],
                             "std": np.std(fs), "training_seconds": sum(r["training_seconds"] for r in fold_rows if r["config"] == config)})
        safe = config.lower().replace("-", "_")
        pd.DataFrame({"id": train.id, "y_true": y, "oof_prediction": oofs[config]}).to_csv(PRED / f"oof_exp031_{safe}.csv", index=False)
        pd.DataFrame({"id": test.id, "prediction": tests[config]}).to_csv(PRED / f"test_exp031_{safe}.csv", index=False)
    variants = pd.DataFrame(variant_rows).sort_values("oof_auc", ascending=False)
    variants.to_csv(REPORTS / "exp031_extratrees_variants.csv", index=False)
    best_name = variants.iloc[0].config
    et = oofs[best_name]; et_test = tests[best_name]
    pd.DataFrame({"id": train.id, "y_true": y, "oof_prediction": et}).to_csv(PRED / "oof_exp031_extratrees.csv", index=False)
    pd.DataFrame({"id": test.id, "prediction": et_test}).to_csv(PRED / "test_exp031_extratrees.csv", index=False)

    # Existing controls and exactly reconstructed persisted ensembles.
    p16 = load_oof(PRED / "oof_exp016_xgboost_depth5_9000.csv", train)
    p22 = load_oof(PRED / "oof_exp022_catboost_thresholds_9000.csv", train)
    s42 = load_oof(PRED / "oof_exp027_seed42.csv", train); s2026 = load_oof(PRED / "oof_exp027_seed2026.csv", train)
    s777 = load_oof(PRED / "oof_exp027_seed777.csv", train); s31415 = load_oof(PRED / "oof_exp028_seed31415.csv", train)
    s1234 = load_oof(PRED / "oof_exp028_seed1234.csv", train)
    p27 = .75 * (.4 * s42 + .3 * s2026 + .3 * s777) + .25 * p22
    p28 = .75 * ((s42 + s2026 + s777 + s31415 + s1234) / 5) + .25 * p22
    p6 = load_oof(PRED / "oof_exp006_lightgbm_features.csv", train)
    controls = {"EXP-016": p16, "EXP-022": p22, "EXP-027": p27, "EXP-028": p28, "EXP-006": p6}
    corr_rows = []
    for name, p in controls.items():
        corr_rows.append({"model": name, "pearson": pd.Series(et).corr(pd.Series(p)),
                          "spearman": pd.Series(et).corr(pd.Series(p), method="spearman")})
    residual = y - p27
    corr_rows.append({"model": "residual_EXP027", "pearson": pd.Series(et).corr(pd.Series(residual)),
                      "spearman": pd.Series(et).corr(pd.Series(residual), method="spearman")})
    correlations = pd.DataFrame(corr_rows)
    correlations.to_csv(REPORTS / "exp031_extratrees_correlations.csv", index=False)

    quality = float(roc_auc_score(y, et))
    pearson27 = float(correlations.loc[correlations.model.eq("EXP-027"), "pearson"].iloc[0])
    allow_blend = quality >= .955 or (.945 <= quality < .955 and pearson27 < .97)
    blend_rows, nested_results = [], {}
    if allow_blend:
        for base_name, base in [("EXP-027", p27), ("EXP-028", p28)]:
            nested, choices, result, diagnostics = nested_blend(y, base, et, splits, base_name)
            nested_results[base_name] = (nested, choices, result)
            for d in diagnostics: blend_rows.append({"kind": "global_diagnostic", **d})
            for c in choices: blend_rows.append({"kind": "nested_choice", **c})
            blend_rows.append({"kind": "nested_result", "base": base_name, "method": "mixed_by_fold",
                               "base_weight": np.nan, "global_auc_diagnostic": result["auc"]})
    pd.DataFrame(blend_rows).to_csv(REPORTS / "exp031_extratrees_blends.csv", index=False)

    # Regional and EXP-016 score-band diagnostics.
    screen = train.daily_screen_time_hours; social = train.social_media_hours; valid = screen.notna() & social.notna()
    cp = valid & ((screen > 8) | (social > 4)); cn = valid & screen.le(6) & social.le(4); amb = valid & ~cp & ~cn
    regional_rows = []
    for label, mask in [("clear_positive", cp), ("clear_negative", cn), ("ambiguous", amb)]:
        for name, p in [("ExtraTrees", et), ("EXP-016", p16)]:
            regional_rows.append({"analysis": "region", "segment": label, "model": name, "rows": int(mask.sum()),
                                  "auc": roc_auc_score(y[mask], p[mask])})
    for label, lo, hi in [("0.30-0.40", .3, .4), ("0.40-0.50", .4, .5), ("0.50-0.60", .5, .6),
                          ("0.60-0.70", .6, .7), ("0.70-0.80", .7, .8)]:
        mask = (p16 >= lo) & (p16 < hi)
        for name, p in [("ExtraTrees", et), ("EXP-016", p16)]:
            regional_rows.append({"analysis": "score_band", "segment": label, "model": name, "rows": int(mask.sum()),
                                  "auc": roc_auc_score(y[mask], p[mask])})
    regional = pd.DataFrame(regional_rows)
    regional.to_csv(REPORTS / "exp031_extratrees_regional.csv", index=False)

    # Average base-feature importance for the winning variant.
    imp_keys = sorted(set().union(*[x.keys() for x in importances[best_name]]))
    importance = pd.DataFrame({"feature": imp_keys,
                               "importance_mean": [np.mean([x.get(k, 0) for x in importances[best_name]]) for k in imp_keys],
                               "importance_std": [np.std([x.get(k, 0) for x in importances[best_name]]) for k in imp_keys]}).sort_values("importance_mean", ascending=False)

    submission_generated = False; submission_path = "none"; selected = None
    if allow_blend:
        eligible = []
        for base_name, (nested, choices, result) in nested_results.items():
            base_ref = EXP027 if base_name == "EXP-027" else EXP028
            base_folds = fold_aucs(y, controls[base_name], splits)
            deltas = np.asarray(result["folds"]) - np.asarray(base_folds)
            if result["auc"] - base_ref >= .00003 and np.sum(deltas > 0) >= 4 and not np.any(deltas < -.00003):
                eligible.append((result["auc"], base_name, nested, choices, result))
        if eligible:
            selected = max(eligible, key=lambda x: x[0])
            _, base_name, nested, choices, result = selected
            # Apply fold choices to test. EXP-027 test exists; reconstruct EXP-028 test exactly.
            t22 = pd.read_csv(PRED / "test_exp022_catboost_thresholds_9000.csv").prediction.to_numpy(np.float64)
            if base_name == "EXP-027":
                base_test = pd.read_csv(SUB / "submission_exp027_seed_ensemble.csv").addicted_label.to_numpy(np.float64)
            else:
                t42 = pd.read_csv(PRED / "test_exp016_xgboost_depth5_9000.csv").prediction.to_numpy(np.float64)
                oldbag = (pd.read_csv(SUB / "submission_exp027_seed_ensemble.csv").addicted_label.to_numpy(np.float64) - .25*t22)/.75
                othermean = (oldbag - .4*t42)/.6
                t31415 = pd.read_csv(PRED / "test_exp028_seed31415.csv").prediction.to_numpy(np.float64)
                t1234 = pd.read_csv(PRED / "test_exp028_seed1234.csv").prediction.to_numpy(np.float64)
                base_test = .75*(.2*t42 + .4*othermean + .2*t31415 + .2*t1234) + .25*t22
            test_blends = []
            for c in choices:
                if c["method"] == "probability": test_blends.append(c["base_weight"]*base_test + (1-c["base_weight"])*et_test)
                else: test_blends.append(c["base_weight"]*normalized_rank(base_test)+(1-c["base_weight"])*normalized_rank(et_test))
            final_test = np.mean(test_blends, axis=0)
            sub = pd.DataFrame({"id": sample.id, "addicted_label": final_test})
            if len(sub) != 296302 or not sub.id.equals(sample.id) or sub.isna().any().any() or not sub.addicted_label.between(0, 1).all():
                raise ValueError("Submission invalida")
            path = SUB / "submission_exp031_extratrees_ensemble.csv"; sub.to_csv(path, index=False)
            submission_generated = True; submission_path = str(path); update_log(result["auc"], base_name, choices)

    elapsed = perf_counter() - total_start
    lines = ["EXP-031 ExtraTrees diversity", f"configs={CONFIGS}; base_params={BASE_PARAMS}; features={NEW_FEATURES}",
             "variants:\n" + variants.to_string(index=False), f"best_variant={best_name}",
             "correlations:\n" + correlations.to_string(index=False), f"allow_blend={allow_blend}",
             "regional_bands:\n" + regional.to_string(index=False), "top25_importance:\n" + importance.head(25).to_string(index=False),
             f"nested_results={{{', '.join(f'{k}:{v[2]}' for k,v in nested_results.items())}}}",
             f"submission_generated={submission_generated}; path={submission_path}", f"total_seconds={elapsed:.2f}", "problems=none"]
    (METRICS / "exp031_extratrees_metrics.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"variants=\n{variants.to_string(index=False)}\nbest={best_name}; correlations=\n{correlations.to_string(index=False)}")
    print(f"nested={nested_results}; submission={submission_generated}; path={submission_path}; seconds={elapsed:.2f}")


if __name__ == "__main__":
    main()
