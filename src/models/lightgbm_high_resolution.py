"""EXP-039: high-resolution LightGBM with fold-safe non-target encodings."""
from __future__ import annotations

import gc
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.features.exact_values import ORIGINAL
from src.features.frequency import (
    FREQUENCY_COLUMNS as FREQ,
    apply_frequency_map,
    exact_key,
    fit_frequency_map,
)
from src.features.relational import RELATIONS, add_numeric_relations as add_relations
from src.models.lightgbm_baseline import prepare_aligned_categories
from src.models.logistic_exact_values import load_oof, rank
from src.project_paths import (
    DATA_DIR as DATA,
    METRICS_DIR as METRICS,
    OUTPUTS_DIR as OUT,
    PREDICTIONS_DIR as PRED,
    PROJECT_ROOT as ROOT,
    REPORTS_DIR as REPORTS,
    SUBMISSIONS_DIR as SUB,
)

ID, TARGET = "id", "addicted_label"
CATEGORICALS = ["gender", "stress_level", "academic_work_impact"]
CODES = {
    "screen_value_code": "daily_screen_time_hours",
    "social_value_code": "social_media_hours",
    "weekend_value_code": "weekend_screen_time",
}
BASE_PARAMS = {
    "objective": "binary", "metric": "auc", "learning_rate": .03,
    "n_estimators": 10000, "num_leaves": 31, "max_depth": -1,
    "min_child_samples": 20, "subsample": .9, "colsample_bytree": .9,
    "reg_alpha": 0., "reg_lambda": 1., "random_state": 42,
    "n_jobs": -1, "verbosity": -1,
}
FINAL_FEATURE_CFG = (("weekend_freq", "screen_freq"), True, tuple())
FINAL_MAX_BIN = 4095
FINAL_PARAM_UPDATES = {"num_leaves": 15, "min_child_samples": 100}
EXP037_ENS = .967068329297671


def rss():
    return psutil.Process().memory_info().rss / 1024**3


def build_fold(X, Xt, raw, rawt, tr, va, feature_cfg, need_test):
    a, b = X.iloc[tr].copy(), X.iloc[va].copy()
    e = Xt.copy() if need_test else None
    freq_names, use_rel, code_names = feature_cfg
    for name in freq_names:
        col = FREQ[name]
        counts = fit_frequency_map(raw[col].iloc[tr])
        a[name] = apply_frequency_map(raw[col].iloc[tr], counts).to_numpy(np.float32)
        b[name] = apply_frequency_map(raw[col].iloc[va], counts).to_numpy(np.float32)
        if need_test:
            e[name] = apply_frequency_map(rawt[col], counts).to_numpy(np.float32)
    if use_rel:
        a = add_relations(a, raw.iloc[tr].reset_index(drop=True))
        b = add_relations(b, raw.iloc[va].reset_index(drop=True))
        if need_test:
            e = add_relations(e, rawt.reset_index(drop=True))
    for name in code_names:
        col = CODES[name]
        ka, kb = exact_key(raw[col].iloc[tr]), exact_key(raw[col].iloc[va])
        mapping = {v: i for i, v in enumerate(sorted(ka.unique()))}
        a[name] = ka.map(mapping).fillna(-1).to_numpy(np.int32)
        b[name] = kb.map(mapping).fillna(-1).to_numpy(np.int32)
        if need_test:
            e[name] = exact_key(rawt[col]).map(mapping).fillna(-1).to_numpy(np.int32)
    return a, b, e


def train_config(X, Xt, raw, rawt, y, splits, fold_ids, max_bin, feature_cfg,
                 param_updates, label, need_test=False, abort_limits=None):
    oof = np.full(len(y), np.nan, np.float64); tests, rows = [], []
    params = {**BASE_PARAMS, "max_bin": int(max_bin), **param_updates}
    for fold in fold_ids:
        tr, va = splits[fold]
        start = perf_counter()
        a, b, e = build_fold(X, Xt, raw, rawt, tr, va, feature_cfg, need_test)
        model = LGBMClassifier(**params)
        model.fit(a, y[tr], eval_set=[(b, y[va])], eval_metric="auc",
                  categorical_feature=CATEGORICALS,
                  callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(0)])
        best = int(model.best_iteration_)
        oof[va] = model.predict_proba(b, num_iteration=best)[:, 1]
        if need_test:
            tests.append(model.predict_proba(e, num_iteration=best)[:, 1].astype(np.float64))
        elapsed, memory = perf_counter()-start, rss()
        rows.append({"label": label, "fold": fold+1, "max_bin": max_bin,
                     "feature_cfg": str(feature_cfg), "num_leaves": params["num_leaves"],
                     "min_child_samples": params["min_child_samples"],
                     "auc": roc_auc_score(y[va], oof[va]), "best_iteration": best,
                     "seconds": elapsed, "rss_gb": memory})
        print(f"{label} fold={fold+1} auc={rows[-1]['auc']:.8f} best={best} sec={elapsed:.1f} ram={memory:.2f}", flush=True)
        del a, b, e, model; gc.collect()
        if abort_limits and (elapsed > abort_limits[0] or memory > abort_limits[1]):
            rows[-1]["aborted_after_fold"] = True
            break
    return oof, np.asarray(tests), rows


def summarize(label, oof, y, splits, fold_ids, rows, baseline=None):
    valid = np.concatenate([splits[f][1] for f in fold_ids])
    auc = roc_auc_score(y[valid], oof[valid])
    fs = np.asarray([roc_auc_score(y[splits[f][1]], oof[splits[f][1]]) for f in fold_ids])
    result = {"label": label, "auc": auc, "mean_auc": fs.mean(), "std": fs.std(),
              "folds": str(fs.tolist()), "best_iterations": str([r["best_iteration"] for r in rows]),
              "seconds": sum(r["seconds"] for r in rows),
              "peak_rss_gb": max(r["rss_gb"] for r in rows), "completed_folds": len(rows)}
    if baseline is not None:
        result["delta"] = auc - baseline
    return result


def nested_two(y, base, other, splits, weights, label):
    rb, ro = rank(base), rank(other); out = np.zeros(len(y)); rows = []
    for f, (tr, va) in enumerate(splits, 1):
        opts = []
        for method in ["probability", "rank"]:
            for w in weights:
                p = (1-w)*base+w*other if method == "probability" else (1-w)*rb+w*ro
                opts.append((roc_auc_score(y[tr], p[tr]), method, w, p))
        z = max(opts, key=lambda q: q[0]); out[va] = z[3][va]
        rows.append({"candidate": label, "fold": f, "method": z[1], "lgbm_weight": z[2],
                     "selection_auc": z[0]})
    fs = np.asarray([roc_auc_score(y[v], out[v]) for _, v in splits])
    rows.append({"candidate": label, "fold": 0, "auc": roc_auc_score(y, out),
                 "std": fs.std(), "folds": str(fs.tolist())})
    return out, rows


def nested_triple(y, p27, p37, p39, splits):
    r27, r37, r39 = rank(p27), rank(p37), rank(p39)
    out = np.zeros(len(y)); rows = []
    for f, (tr, va) in enumerate(splits, 1):
        opts = []
        for method in ["probability", "rank"]:
            for w in [.05, .10, .15, .20, .25, .30, .35, .40, .45]:
                a, b = .625*(1-w), .375*(1-w)
                p = a*p27+b*p37+w*p39 if method == "probability" else a*r27+b*r37+w*r39
                opts.append((roc_auc_score(y[tr], p[tr]), method, w, a, b, p))
        z = max(opts, key=lambda q: q[0]); out[va] = z[5][va]
        rows.append({"candidate": "TRIPLE", "fold": f, "method": z[1],
                     "lgbm_weight": z[2], "exp027_weight": z[3], "exp037_weight": z[4],
                     "selection_auc": z[0]})
    fs = np.asarray([roc_auc_score(y[v], out[v]) for _, v in splits])
    rows.append({"candidate": "TRIPLE", "fold": 0, "auc": roc_auc_score(y, out),
                 "std": fs.std(), "folds": str(fs.tolist())})
    return out, rows


def main():
    total_start = perf_counter()
    for d in [PRED, REPORTS, METRICS, SUB]: d.mkdir(parents=True, exist_ok=True)
    train, test = pd.read_csv(DATA/"train.csv"), pd.read_csv(DATA/"test.csv")
    y = train[TARGET].to_numpy()
    raw, rawt = train[ORIGINAL].copy(), test[ORIGINAL].copy()
    X, Xt = prepare_aligned_categories(raw, rawt, CATEGORICALS)
    splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
    screen_folds = [0, 1, 2]; full_folds = [0, 1, 2, 3, 4]

    numeric = [c for c in ORIGINAL if c not in CATEGORICALS]
    cardinality = pd.DataFrame([{"feature": c, "train_unique": raw[c].nunique(dropna=False),
                                 "test_unique": rawt[c].nunique(dropna=False), "min": raw[c].min(),
                                 "max": raw[c].max(), "dtype": str(raw[c].dtype)} for c in numeric])
    max_card = int(cardinality.train_unique.max())
    recommended_bin = int(2**np.ceil(np.log2(max_card+1))-1)
    print(cardinality.to_string(index=False), flush=True)
    print(f"max_numeric_cardinality={max_card}; recommended_max_bin={recommended_bin}", flush=True)

    progress = []
    def save_progress(rows):
        progress.extend(rows); pd.DataFrame(progress).to_csv(REPORTS/"exp039_all_fold_progress.csv", index=False)

    raw_cfg = (tuple(), False, tuple())
    maxbin_rows, maxbin_cache = [], {}
    baseline_fold_time = None
    for mb in [255, 511, 1023, 2047, 4095]:
        abort = (baseline_fold_time*4, 12.5) if baseline_fold_time else None
        oof, _, rows = train_config(X, Xt, raw, rawt, y, splits, screen_folds, mb, raw_cfg, {},
                                    f"maxbin_{mb}", abort_limits=abort)
        save_progress(rows)
        if mb == 255 and rows: baseline_fold_time = np.mean([r["seconds"] for r in rows])
        summary = summarize(f"maxbin_{mb}", oof, y, splits, list(range(len(rows))), rows)
        summary["max_bin"] = mb; maxbin_rows.append(summary); maxbin_cache[mb] = oof
        if len(rows) < 3:
            print(f"max_bin={mb} stopped as excessive", flush=True)
    maxbin_df = pd.DataFrame(maxbin_rows); maxbin_df.to_csv(REPORTS/"exp039_maxbin_screen.csv", index=False)
    completed = maxbin_df[maxbin_df.completed_folds == 3]
    best_bin = int(completed.sort_values(["mean_auc", "max_bin"], ascending=[False, True]).iloc[0].max_bin)
    raw_screen_auc = float(completed.loc[completed.max_bin == best_bin, "auc"].iloc[0])

    # Required full raw model.
    raw_oof, raw_tests, rows = train_config(X, Xt, raw, rawt, y, splits, full_folds, best_bin,
                                            raw_cfg, {}, "raw_full", need_test=True)
    save_progress(rows)
    raw_full_summary = summarize("raw_full", raw_oof, y, splits, full_folds, rows)
    raw_test = raw_tests.mean(axis=0, dtype=np.float64)
    pd.DataFrame({ID: train[ID], "y_true": y, "oof_prediction": raw_oof}).to_csv(PRED/"oof_exp039_lgbm_highbin_raw.csv", index=False)
    pd.DataFrame({ID: test[ID], "prediction": raw_test}).to_csv(PRED/"test_exp039_lgbm_highbin_raw.csv", index=False)

    # Fold-safe frequency screening.
    freq_rows, freq_cache = [], {}
    for name in FREQ:
        cfg = ((name,), False, tuple())
        oof, _, rows = train_config(X, Xt, raw, rawt, y, splits, screen_folds, best_bin, cfg, {}, f"freq_{name}")
        save_progress(rows); s = summarize(name, oof, y, splits, screen_folds, rows, raw_screen_auc)
        freq_rows.append(s); freq_cache[(name,)] = oof
    order = [r["label"] for r in sorted(freq_rows, key=lambda q: q["auc"], reverse=True)]
    freq_sets = [tuple(order[:1]), tuple(order[:2]), tuple(order[:3]), tuple(order)]
    seen_sets = set()
    for names in freq_sets:
        if names in seen_sets: continue
        seen_sets.add(names)
        if len(names) == 1: oof = freq_cache[names]
        else:
            cfg = (names, False, tuple())
            oof, _, rows = train_config(X, Xt, raw, rawt, y, splits, screen_folds, best_bin, cfg, {}, f"freq_set_{len(names)}")
            save_progress(rows); freq_cache[names] = oof
        # recover rows only for summary bookkeeping where necessary
        valid = np.concatenate([splits[f][1] for f in screen_folds])
        fs = [roc_auc_score(y[splits[f][1]], oof[splits[f][1]]) for f in screen_folds]
        freq_rows.append({"label": "+".join(names), "auc": roc_auc_score(y[valid], oof[valid]),
                          "mean_auc": np.mean(fs), "std": np.std(fs), "folds": str(fs),
                          "delta": roc_auc_score(y[valid], oof[valid])-raw_screen_auc})
    freq_df = pd.DataFrame(freq_rows); freq_df.to_csv(REPORTS/"exp039_frequency_screen.csv", index=False)
    best_freq_tuple = max(freq_cache, key=lambda k: roc_auc_score(y[np.concatenate([splits[f][1] for f in screen_folds])], freq_cache[k][np.concatenate([splits[f][1] for f in screen_folds])]))

    # Feature families.
    family_cfgs = {
        "raw": raw_cfg, "frequency": (best_freq_tuple, False, tuple()),
        "relations": (tuple(), True, tuple()),
        "frequency_relations": (best_freq_tuple, True, tuple()),
    }
    family_rows, family_cache = [], {"raw": maxbin_cache[best_bin], "frequency": freq_cache[best_freq_tuple]}
    for name, cfg in family_cfgs.items():
        if name not in family_cache:
            oof, _, rows = train_config(X, Xt, raw, rawt, y, splits, screen_folds, best_bin, cfg, {}, f"family_{name}")
            save_progress(rows); family_cache[name] = oof
        oof = family_cache[name]; valid = np.concatenate([splits[f][1] for f in screen_folds])
        fs = [roc_auc_score(y[splits[f][1]], oof[splits[f][1]]) for f in screen_folds]
        auc = roc_auc_score(y[valid], oof[valid])
        family_rows.append({"feature_set": name, "feature_cfg": str(cfg), "auc": auc,
                            "mean_auc": np.mean(fs), "std": np.std(fs), "delta_vs_raw": auc-raw_screen_auc,
                            "folds": str(fs), "eligible_full": auc-raw_screen_auc >= .00010})
    family_df = pd.DataFrame(family_rows); family_df.to_csv(REPORTS/"exp039_feature_sets.csv", index=False)

    # Exact value codes.
    code_rows = []
    for names in [(x,) for x in CODES] + [tuple(CODES)]:
        cfg = (tuple(), False, names)
        oof, _, rows = train_config(X, Xt, raw, rawt, y, splits, screen_folds, best_bin, cfg, {}, "codes_"+str(len(names)))
        save_progress(rows); s = summarize("+".join(names), oof, y, splits, screen_folds, rows, raw_screen_auc)
        code_rows.append(s)
    pd.DataFrame(code_rows).to_csv(REPORTS/"exp039_value_codes.csv", index=False)

    # Choose best screened family, then sequential mini-screen.
    eligible_families = family_df[(family_df.feature_set == "raw") | family_df.eligible_full]
    best_family = eligible_families.sort_values("mean_auc", ascending=False).iloc[0]
    best_cfg = family_cfgs[best_family.feature_set]
    param_rows, param_cache = [], {}
    for leaves in [15, 31, 63]:
        key = (leaves, 20)
        if leaves == 31:
            oof = family_cache[best_family.feature_set]
            valid = np.concatenate([splits[f][1] for f in screen_folds]); fs = [roc_auc_score(y[splits[f][1]], oof[splits[f][1]]) for f in screen_folds]
            row = {"stage": "num_leaves", "num_leaves": leaves, "min_child_samples": 20,
                   "auc": roc_auc_score(y[valid], oof[valid]), "mean_auc": np.mean(fs), "std": np.std(fs), "folds": str(fs)}
        else:
            oof, _, rows = train_config(X, Xt, raw, rawt, y, splits, screen_folds, best_bin, best_cfg,
                                        {"num_leaves": leaves}, f"leaves_{leaves}")
            save_progress(rows); row = summarize(f"leaves_{leaves}", oof, y, splits, screen_folds, rows); row["stage"]="num_leaves"; row["num_leaves"]=leaves; row["min_child_samples"]=20
        param_cache[key] = oof; param_rows.append(row)
    best_leaves = int(max([r for r in param_rows if r["stage"] == "num_leaves"], key=lambda q:q["mean_auc"])["num_leaves"])
    for child in [20, 50, 100]:
        if child == 20: continue
        oof, _, rows = train_config(X, Xt, raw, rawt, y, splits, screen_folds, best_bin, best_cfg,
                                    {"num_leaves": best_leaves, "min_child_samples": child}, f"child_{child}")
        save_progress(rows); row = summarize(f"child_{child}", oof, y, splits, screen_folds, rows); row["stage"]="min_child_samples"; row["num_leaves"]=best_leaves; row["min_child_samples"]=child
        param_cache[(best_leaves, child)] = oof; param_rows.append(row)
    param_df = pd.DataFrame(param_rows); param_df.to_csv(REPORTS/"exp039_param_screen.csv", index=False)
    child_candidates = [r for r in param_rows if r.get("num_leaves") == best_leaves]
    best_param = max(child_candidates, key=lambda q:q["mean_auc"])
    best_child = int(best_param["min_child_samples"])

    final_updates = {"num_leaves": best_leaves, "min_child_samples": best_child}
    if best_cfg == raw_cfg and best_leaves == 31 and best_child == 20:
        final_oof, final_tests = raw_oof, raw_tests
        final_rows = []
    else:
        final_oof, final_tests, final_rows = train_config(X, Xt, raw, rawt, y, splits, full_folds, best_bin,
                                                         best_cfg, final_updates, "FINAL", need_test=True)
        save_progress(final_rows)
    final_test = final_tests.mean(axis=0, dtype=np.float64)
    final_auc = roc_auc_score(y, final_oof)
    final_fs = np.asarray([roc_auc_score(y[v], final_oof[v]) for _,v in splits])
    pd.DataFrame({ID:train[ID],"y_true":y,"oof_prediction":final_oof}).to_csv(PRED/"oof_exp039_lgbm_highbin.csv",index=False)
    pd.DataFrame({ID:test[ID],"prediction":final_test}).to_csv(PRED/"test_exp039_lgbm_highbin.csv",index=False)

    # Diversity and diagnostics.
    refs = {
        "EXP-016": load_oof(PRED/"oof_exp016_xgboost_depth5_9000.csv",train),
        "EXP-022": load_oof(PRED/"oof_exp022_catboost_thresholds_9000.csv",train),
        "EXP-037": load_oof(PRED/"oof_exp037_relational_logistic.csv",train),
    }
    seeds=[load_oof(PRED/f"oof_exp027_seed{x}.csv",train) for x in [42,2026,777]]
    refs["EXP-027"] = .75*(.4*seeds[0]+.3*seeds[1]+.3*seeds[2])+.25*refs["EXP-022"]
    correlations = pd.DataFrame([{"model":n,"pearson":pd.Series(final_oof).corr(pd.Series(p)),"spearman":pd.Series(final_oof).corr(pd.Series(p),method="spearman")} for n,p in {**refs,"residual_EXP027":y-refs["EXP-027"]}.items()])
    correlations.to_csv(REPORTS/"exp039_correlations.csv",index=False)
    screen,social=train.daily_screen_time_hours,train.social_media_hours;known=screen.notna()&social.notna();cp=known&((screen>8)|(social>4));cn=known&(screen<=6)&(social<=4);amb=known&~cp&~cn
    regional=[]
    for name,mask in [("clear_positive",cp),("clear_negative",cn),("ambiguous",amb)]:
        for model,p in [("EXP-039",final_oof),("EXP-027",refs["EXP-027"])]: regional.append({"analysis":"region","segment":name,"model":model,"rows":mask.sum(),"auc":roc_auc_score(y[mask],p[mask])})
    for name,lo,hi in [("0.30-0.40",.3,.4),("0.40-0.50",.4,.5),("0.50-0.60",.5,.6),("0.60-0.70",.6,.7),("0.70-0.80",.7,.8)]:
        mask=(refs["EXP-016"]>=lo)&(refs["EXP-016"]<hi)
        for model,p in [("EXP-039",final_oof),("EXP-027",refs["EXP-027"])]:regional.append({"analysis":"score_band","segment":name,"model":model,"rows":mask.sum(),"auc":roc_auc_score(y[mask],p[mask])})
    regional_df=pd.DataFrame(regional);regional_df.to_csv(REPORTS/"exp039_regional.csv",index=False)

    blend_rows=[];blend_oofs={}
    if final_auc>=.9650:
        z,rows=nested_two(y,refs["EXP-027"],final_oof,splits,[.05,.10,.15,.20,.25,.30,.35,.40],"EXP027_EXP039");blend_oofs["two"]=z;blend_rows+=rows
        z,rows=nested_triple(y,refs["EXP-027"],refs["EXP-037"],final_oof,splits);blend_oofs["triple"]=z;blend_rows+=rows
    blend_df=pd.DataFrame(blend_rows);blend_df.to_csv(REPORTS/"exp039_blends.csv",index=False)

    generated=False;subpath="";best_blend=None
    if blend_oofs:
        best_blend=max(blend_oofs,key=lambda k:roc_auc_score(y,blend_oofs[k]));z=blend_oofs[best_blend];zauc=roc_auc_score(y,z);zfs=np.asarray([roc_auc_score(y[v],z[v]) for _,v in splits]);ref=.625*rank(refs["EXP-027"])+.375*rank(refs["EXP-037"]);rfs=np.asarray([roc_auc_score(y[v],ref[v]) for _,v in splits])
        if zauc>=EXP037_ENS+.00010 and sum(zfs>rfs)>=4 and np.min(zfs-rfs)>=-.00005:
            choices=blend_df[(blend_df.candidate==( "TRIPLE" if best_blend=="triple" else "EXP027_EXP039"))&(blend_df.fold>0)]
            t27=pd.read_csv(SUB/"submission_exp027_seed_ensemble.csv").addicted_label.to_numpy(np.float64);t37=pd.read_csv(PRED/"test_exp037_relational_logistic.csv").prediction.to_numpy(np.float64);parts=[]
            for choice,tl in zip(choices.itertuples(),final_tests):
                if best_blend=="two": comps=[1-choice.lgbm_weight,choice.lgbm_weight];vals=[t27,tl]
                else: comps=[choice.exp027_weight,choice.exp037_weight,choice.lgbm_weight];vals=[t27,t37,tl]
                parts.append(sum(w*(rank(v) if choice.method=="rank" else v) for w,v in zip(comps,vals)))
            sample=pd.read_csv(DATA/"sample_submission.csv");submission=pd.DataFrame({ID:sample[ID],TARGET:np.mean(parts,axis=0,dtype=np.float64)});subpath=SUB/"submission_exp039_highbin_lgbm_ensemble.csv";submission.to_csv(subpath,index=False);generated=True
            logpath=METRICS/"experiment_log.csv";log=pd.read_csv(logpath)
            if not log.experiment_id.astype(str).eq("EXP-039").any():
                log=pd.concat([log,pd.DataFrame([{"experiment_id":"EXP-039","datetime":pd.Timestamp.now().isoformat(timespec="seconds"),"model":"HighBinLightGBM_Ensemble","features":str(best_cfg),"cv_strategy":"Nested_OOF_ensemble_optimization","cv_roc_auc":zauc,"kaggle_score":"","notes":f"{best_blend}; max_bin={best_bin}; leaves={best_leaves}; min_child={best_child}"}])],ignore_index=True);log.to_csv(logpath,index=False)

    total=perf_counter()-total_start
    lines=["EXP-039 high-bin LightGBM",f"cardinality=\n{cardinality.to_string(index=False)}",f"max_cardinality={max_card};recommended_max_bin={recommended_bin}",f"maxbin=\n{maxbin_df.to_string(index=False)}",f"raw_full={raw_full_summary}",f"frequency=\n{freq_df.to_string(index=False)}",f"feature_sets=\n{family_df.to_string(index=False)}",f"value_codes=\n{pd.DataFrame(code_rows).to_string(index=False)}",f"params=\n{param_df.to_string(index=False)}",f"final_feature_cfg={best_cfg};params={final_updates};max_bin={best_bin}",f"final_auc={final_auc};folds={final_fs.tolist()};std={final_fs.std()}",f"comparisons=EXP004:{final_auc-.963537};EXP006:{final_auc-.963664};EXP016:{final_auc-.96570173};EXP027:{final_auc-.965919188052602};EXP037_logistic:{final_auc-.9635752939431477}",f"correlations=\n{correlations.to_string(index=False)}",f"regional=\n{regional_df.to_string(index=False)}",f"blends=\n{blend_df.to_string(index=False)}",f"best_blend={best_blend}",f"submission={generated};path={subpath}",f"total_seconds={total}"]
    (METRICS/"exp039_highbin_metrics.txt").write_text("\n".join(lines)+"\n",encoding="utf8")
    print(f"FINAL auc={final_auc:.10f} best_bin={best_bin} submission={generated} seconds={total:.1f}",flush=True)


if __name__=="__main__":main()
