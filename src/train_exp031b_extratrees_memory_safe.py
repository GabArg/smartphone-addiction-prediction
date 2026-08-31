"""EXP-031B: memory-safe ExtraTrees diversity experiment."""

from __future__ import annotations

import gc
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from train_xgboost_exp012_threshold_features import NEW_FEATURES, add_threshold_features

try:
    import psutil
except ImportError:
    psutil = None


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
PRED = OUT / "predictions"
REPORTS = OUT / "reports"
METRICS = OUT / "metrics"
SUB = OUT / "submissions"
MODEL_PARAMS = dict(n_estimators=300, max_depth=18, min_samples_split=4, min_samples_leaf=5,
                    max_features="sqrt", bootstrap=False, class_weight=None, random_state=42, n_jobs=1)
ET_WEIGHTS = [.01, .02, .03, .05, .075, .10]
EXP027 = .965919188052602
EXP028 = .9659307051390922


def as_float32(x):
    return x.astype(np.float32, copy=False)


def rss_gb():
    return psutil.Process().memory_info().rss / 1024**3 if psutil is not None else np.nan


def load_oof(path, train):
    df = pd.read_csv(path)
    if not df.id.equals(train.id) or not df.y_true.equals(train.addicted_label) or df.oof_prediction.isna().any():
        raise ValueError(f"OOF invalida: {path.name}")
    return df.oof_prediction.to_numpy(np.float64)


def ranks(x):
    r = pd.Series(x).rank(method="average").to_numpy(np.float64)
    return (r - 1) / (len(r) - 1)


def fold_aucs(y, p, splits):
    return [float(roc_auc_score(y[va], p[va])) for _, va in splits]


def aggregate(names, values):
    out = {}
    cats = ["gender", "stress_level", "academic_work_impact"]
    for name, value in zip(names, values):
        clean = name.split("__", 1)[-1]
        if name.startswith("cat__"):
            match = [c for c in cats if clean.startswith(c + "_")]
            base = match[0] if match else clean
        else:
            base = clean
        out[base] = out.get(base, 0.) + float(value)
    return out


def nested_blend(y, base, et, splits, base_name):
    rb, re = ranks(base), ranks(et)
    nested = np.zeros(len(y), np.float64)
    choices = []
    diagnostics = []
    for method in ["probability", "rank"]:
        b, e = (base, et) if method == "probability" else (rb, re)
        for ew in ET_WEIGHTS:
            p = (1 - ew) * b + ew * e
            diagnostics.append({"base": base_name, "kind": "global_diagnostic", "method": method,
                                "et_weight": ew, "auc": roc_auc_score(y, p)})
    for fold, (tr, va) in enumerate(splits, 1):
        options = []
        for method in ["probability", "rank"]:
            b, e = (base, et) if method == "probability" else (rb, re)
            for ew in ET_WEIGHTS:
                p = (1 - ew) * b + ew * e
                options.append((roc_auc_score(y[tr], p[tr]), -ew, method, p, ew))
        best = max(options, key=lambda x: (x[0], x[1]))
        nested[va] = best[3][va]
        choices.append({"base": base_name, "kind": "nested_choice", "fold": fold,
                        "method": best[2], "et_weight": best[4], "selection_auc": best[0]})
    fs = fold_aucs(y, nested, splits)
    result = {"base": base_name, "kind": "nested_result", "method": "fold_specific",
              "et_weight": np.nan, "auc": roc_auc_score(y, nested), "fold1": fs[0], "fold2": fs[1],
              "fold3": fs[2], "fold4": fs[3], "fold5": fs[4], "std": np.std(fs)}
    return nested, choices, result, diagnostics


def main():
    start_all = perf_counter()
    for d in [PRED, REPORTS, METRICS, SUB]:
        d.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    original = [c for c in train if c not in {"id", "addicted_label"}]
    raw = add_threshold_features(train[original])
    raw_test = add_threshold_features(test[original])
    if raw.columns.tolist() != raw_test.columns.tolist() or len(raw.columns) != len(original) + len(NEW_FEATURES):
        raise ValueError("Features no coinciden con EXP-016")
    cats = [c for c in original if not pd.api.types.is_numeric_dtype(raw[c])]
    nums = [c for c in raw if c not in cats]
    y = train.addicted_label.to_numpy()
    splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(raw, y))
    oof = np.zeros(len(train), np.float64)
    test_pred = np.zeros(len(test), np.float64)
    rows, importance_folds = [], []
    peak_observed = rss_gb()
    fold1_memory_ok = False

    for fold, (tr, va) in enumerate(splits, 1):
        fold_start = perf_counter()
        prep = ColumnTransformer([
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("float32", FunctionTransformer(as_float32, feature_names_out="one-to-one"))]), nums),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32))]), cats),
        ], sparse_threshold=0.0)
        try:
            Xtr = prep.fit_transform(raw.iloc[tr])
            Xva = prep.transform(raw.iloc[va])
            Xt = prep.transform(raw_test)
            peak_observed = np.nanmax([peak_observed, rss_gb()])
            model = ExtraTreesClassifier(**MODEL_PARAMS)
            model.fit(Xtr, y[tr])
            peak_observed = np.nanmax([peak_observed, rss_gb()])
            pv = model.predict_proba(Xva)[:, 1].astype(np.float64)
            pt = model.predict_proba(Xt)[:, 1].astype(np.float64)
        except MemoryError:
            if fold == 1:
                (METRICS / "exp031b_extratrees_metrics.txt").write_text(
                    "EXP-031B status=blocked_fold1_memory_error\nfold1_in_memory=false\nmodel=" + str(MODEL_PARAMS) +
                    f"\npeak_observed_rss_gb={peak_observed}\n", encoding="utf-8")
            raise
        oof[va] = pv; test_pred += pt / 5
        auc = float(roc_auc_score(y[va], pv))
        importance_folds.append(aggregate(prep.get_feature_names_out(), model.feature_importances_))
        seconds = perf_counter() - fold_start
        rows.append({"fold": fold, "auc": auc, "seconds": seconds, "observed_rss_gb": rss_gb()})
        pd.DataFrame(rows).to_csv(REPORTS / "exp031b_fold_progress.csv", index=False)
        print(f"fold={fold} auc={auc:.8f} seconds={seconds:.1f} rss_gb={rss_gb():.2f}", flush=True)
        if fold == 1:
            fold1_memory_ok = True
        del model, Xtr, Xva, Xt, prep, pv, pt
        gc.collect()

    pd.DataFrame({"id": train.id, "y_true": y, "oof_prediction": oof}).to_csv(PRED / "oof_exp031b_extratrees.csv", index=False)
    pd.DataFrame({"id": test.id, "prediction": test_pred}).to_csv(PRED / "test_exp031b_extratrees.csv", index=False)
    fs = fold_aucs(y, oof, splits)
    oof_auc = float(roc_auc_score(y, oof)); std = float(np.std(fs))

    p16 = load_oof(PRED / "oof_exp016_xgboost_depth5_9000.csv", train)
    p22 = load_oof(PRED / "oof_exp022_catboost_thresholds_9000.csv", train)
    s42 = load_oof(PRED / "oof_exp027_seed42.csv", train); s2026 = load_oof(PRED / "oof_exp027_seed2026.csv", train)
    s777 = load_oof(PRED / "oof_exp027_seed777.csv", train); s31415 = load_oof(PRED / "oof_exp028_seed31415.csv", train)
    s1234 = load_oof(PRED / "oof_exp028_seed1234.csv", train)
    p27 = .75*(.4*s42+.3*s2026+.3*s777)+.25*p22
    p28 = .75*((s42+s2026+s777+s31415+s1234)/5)+.25*p22
    p6 = load_oof(PRED / "oof_exp006_lightgbm_features.csv", train)
    controls = {"EXP-016": p16, "EXP-022": p22, "EXP-027": p27, "EXP-028": p28, "EXP-006": p6}
    corr = pd.DataFrame([{"model": name, "pearson": pd.Series(oof).corr(pd.Series(p)),
                          "spearman": pd.Series(oof).corr(pd.Series(p), method="spearman")}
                         for name, p in controls.items()])
    corr.to_csv(REPORTS / "exp031b_correlations.csv", index=False)

    screen, social = train.daily_screen_time_hours, train.social_media_hours
    valid = screen.notna() & social.notna(); cp = valid & ((screen > 8) | (social > 4))
    cn = valid & screen.le(6) & social.le(4); amb = valid & ~cp & ~cn
    reg_rows = []
    for label, mask in [("clear_positive", cp), ("clear_negative", cn), ("ambiguous", amb)]:
        for name, p in [("ExtraTrees", oof), ("EXP-016", p16)]:
            reg_rows.append({"analysis": "region", "segment": label, "model": name, "rows": mask.sum(),
                             "auc": roc_auc_score(y[mask], p[mask])})
    for label, lo, hi in [("0.30-0.40",.3,.4),("0.40-0.50",.4,.5),("0.50-0.60",.5,.6),
                          ("0.60-0.70",.6,.7),("0.70-0.80",.7,.8)]:
        mask = (p16 >= lo) & (p16 < hi)
        for name, p in [("ExtraTrees", oof), ("EXP-016", p16)]:
            reg_rows.append({"analysis": "score_band", "segment": label, "model": name, "rows": mask.sum(),
                             "auc": roc_auc_score(y[mask], p[mask])})
    regional = pd.DataFrame(reg_rows); regional.to_csv(REPORTS / "exp031b_regional.csv", index=False)

    keys = sorted(set().union(*[x.keys() for x in importance_folds]))
    importance = pd.DataFrame({"feature": keys,
                               "importance_mean": [np.mean([x.get(k,0) for x in importance_folds]) for k in keys],
                               "importance_std": [np.std([x.get(k,0) for x in importance_folds]) for k in keys]}).sort_values("importance_mean", ascending=False)

    blend_rows, nested_results = [], {}
    if oof_auc >= .945:
        for name, base in [("EXP-027", p27), ("EXP-028", p28)]:
            nested, choices, result, diagnostics = nested_blend(y, base, oof, splits, name)
            nested_results[name] = (nested, choices, result)
            blend_rows.extend(diagnostics); blend_rows.extend(choices); blend_rows.append(result)
    pd.DataFrame(blend_rows).to_csv(REPORTS / "exp031b_blends.csv", index=False)

    useful = ((oof_auc >= .955 and float(corr.loc[corr.model.eq("EXP-027"),"pearson"].iloc[0]) <= .985) or
              any(v[2]["auc"] - (EXP027 if k == "EXP-027" else EXP028) >= .00002 for k,v in nested_results.items()))
    submission_generated = False
    # No submission unless a nested blend clears +0.00003 vs EXP-027.
    if "EXP-027" in nested_results and nested_results["EXP-027"][2]["auc"] - EXP027 >= .00003:
        # This branch is intentionally conservative; fold-specific test blending is possible from saved test predictions.
        nested, choices, result = nested_results["EXP-027"]
        base_test = pd.read_csv(SUB / "submission_exp027_seed_ensemble.csv").addicted_label.to_numpy(np.float64)
        test_parts = []
        for c in choices:
            test_parts.append((1-c["et_weight"])*base_test+c["et_weight"]*test_pred if c["method"]=="probability"
                              else (1-c["et_weight"])*ranks(base_test)+c["et_weight"]*ranks(test_pred))
        sample = pd.read_csv(DATA / "sample_submission.csv")
        sub = pd.DataFrame({"id":sample.id,"addicted_label":np.mean(test_parts,axis=0)})
        if len(sub)==296302 and sub.id.equals(sample.id) and not sub.isna().any().any() and sub.addicted_label.between(0,1).all():
            sub.to_csv(SUB / "submission_exp031b_extratrees_ensemble.csv",index=False); submission_generated=True

    elapsed = perf_counter()-start_all
    lines = ["EXP-031B memory-safe ExtraTrees",f"fold1_in_memory={fold1_memory_ok}",f"model={MODEL_PARAMS}",
             f"fold_aucs={fs}; oof_auc={oof_auc:.12f}; std={std:.12f}",f"fold_rows={rows}",
             f"peak_observed_rss_gb={peak_observed}","correlations:\n"+corr.to_string(index=False),
             "regional_bands:\n"+regional.to_string(index=False),"top25_importance:\n"+importance.head(25).to_string(index=False),
             f"nested_results={{{', '.join(f'{k}:{v[2]}' for k,v in nested_results.items())}}}",
             f"useful_diversity={useful}; submission_generated={submission_generated}",f"total_seconds={elapsed:.2f}","problems=none"]
    (METRICS / "exp031b_extratrees_metrics.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"oof={oof_auc:.12f}; std={std:.8f}; useful={useful}; nested={nested_results}; seconds={elapsed:.2f}")


if __name__ == "__main__":
    main()
