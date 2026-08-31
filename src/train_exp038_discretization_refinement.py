"""EXP-038: unsupervised discretization refinement of EXP-037 relations."""
from __future__ import annotations

import gc
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

from diagnose_exp029_pairwise_ranking import sample_pairs
from train_exp035_exact_value_logistic import load_oof, rank, safe_ratio, stringify
from train_exp036_ratio_ablation import base_rep

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
PRED = OUT / "predictions"
REPORTS = OUT / "reports"
METRICS = OUT / "metrics"
SUB = OUT / "submissions"

EXP037_LOGISTIC = 0.9635752939431477
EXP037_ENSEMBLE = 0.967068329297671
REL = {
    "social_over_screen": ("ratio", "social_media_hours", "daily_screen_time_hours"),
    "gaming_over_screen": ("ratio", "gaming_hours", "daily_screen_time_hours"),
    "work_over_screen": ("ratio", "work_study_hours", "daily_screen_time_hours"),
    "work_over_social": ("ratio", "work_study_hours", "social_media_hours"),
    "gaming_over_social": ("ratio", "gaming_hours", "social_media_hours"),
    "screen_minus_social": ("difference", "daily_screen_time_hours", "social_media_hours"),
}
BASE_CFG = {
    "social_over_screen": ("round", 2.0),
    "gaming_over_screen": ("round", 2.0),
    "work_over_screen": ("round", 2.0),
    "work_over_social": ("round", 2.0),
    "gaming_over_social": ("round", 2.0),
    "screen_minus_social": ("round", 1.0),
}
TOP = ["work_over_screen", "social_over_screen", "gaming_over_screen"]
SECONDARY = ["work_over_social", "gaming_over_social"]
WEIGHTS = [.30, .325, .35, .375, .40, .425, .45]


def rss():
    return psutil.Process().memory_info().rss / 1024**3


def values(raw: pd.DataFrame, feature: str) -> pd.Series:
    kind, a, b = REL[feature]
    if kind == "ratio":
        return safe_ratio(raw[a], raw[b])
    return raw[a] - raw[b]


def transform_nonquantile(s: pd.Series, spec: tuple[str, float]) -> pd.Series:
    method, arg = spec
    x = s.to_numpy(np.float64)
    if method == "round":
        x = np.round(x, int(arg))
    elif method == "step":
        x = np.round(x / arg) * arg
    else:
        raise ValueError(spec)
    return stringify(pd.Series(x, index=s.index))


def quantile_transform(train_s, valid_s, test_s, bins):
    tr = train_s.to_numpy(np.float64)
    va = valid_s.to_numpy(np.float64)
    te = test_s.to_numpy(np.float64) if test_s is not None else None
    mt, mv = np.isfinite(tr), np.isfinite(va)
    me = np.isfinite(te) if te is not None else None
    disc = KBinsDiscretizer(
        n_bins=bins, encode="ordinal", strategy="quantile", subsample=None,
        quantile_method="averaged_inverted_cdf",
    )
    disc.fit(tr[mt, None])

    def apply(x, mask, index):
        out = pd.Series("__MISSING__", index=index, dtype="string")
        if mask.any():
            z = disc.transform(x[mask, None]).ravel().astype(np.int32)
            out.iloc[np.flatnonzero(mask)] = pd.Series(z).astype(str).to_numpy()
        return out

    a = apply(tr, mt, train_s.index)
    b = apply(va, mv, valid_s.index)
    e = apply(te, me, test_s.index) if test_s is not None else None
    return a, b, e, int(sum(len(x) - 1 for x in disc.bin_edges_))


def build_fold(base, raw, indices, cfg, quantile_fit=None, test=None, basetest=None):
    x = base.iloc[indices].copy().reset_index(drop=True)
    xt = basetest.copy().reset_index(drop=True) if test is not None else None
    effective = {}
    for feature, spec in cfg.items():
        s = values(raw, feature)
        if spec[0] != "quantile":
            x[feature] = transform_nonquantile(s.iloc[indices].reset_index(drop=True), spec)
            if test is not None:
                xt[feature] = transform_nonquantile(values(test, feature).reset_index(drop=True), spec)
        else:
            if quantile_fit is None:
                raise ValueError("quantile_fit indices required")
            fit_s = s.iloc[quantile_fit].reset_index(drop=True)
            val_s = s.iloc[indices].reset_index(drop=True)
            test_s = values(test, feature).reset_index(drop=True) if test is not None else None
            _, xv, xe, n = quantile_transform(fit_s, val_s, test_s, int(spec[1]))
            x[feature], effective[feature] = xv, n
            if test is not None:
                xt[feature] = xe
    return x, xt, effective


def fit_cv(train, test, base, basetest, y, splits, cfg, label, need_test=False):
    oof = np.zeros(len(train), np.float64)
    tests, rows = [], []
    has_quantile = any(v[0] == "quantile" for v in cfg.values())
    for fold, (tr, va) in enumerate(splits, 1):
        start = perf_counter()
        if has_quantile:
            xa, _, ea = build_fold(base, train, tr, cfg, quantile_fit=tr,
                                    test=test if need_test else None,
                                    basetest=basetest if need_test else None)
            xb, _, eb = build_fold(base, train, va, cfg, quantile_fit=tr)
            # build_fold returns test from its first output path; create it once
            xt = None
            if need_test:
                _, xt, _ = build_fold(base, train, tr, cfg, quantile_fit=tr,
                                      test=test, basetest=basetest)
            effective = {**ea, **eb}
        else:
            xa, _, effective = build_fold(base, train, tr, cfg)
            xb, _, _ = build_fold(base, train, va, cfg)
            xt = build_fold(base, train, tr, cfg, test=test, basetest=basetest)[1] if need_test else None
        encoder = OneHotEncoder(handle_unknown="ignore", dtype=np.float32, sparse_output=True)
        a = sparse.csr_matrix(encoder.fit_transform(xa), dtype=np.float32)
        b = sparse.csr_matrix(encoder.transform(xb), dtype=np.float32)
        e = sparse.csr_matrix(encoder.transform(xt), dtype=np.float32) if need_test else None
        model = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000, random_state=42)
        model.fit(a, y[tr])
        oof[va] = model.predict_proba(b)[:, 1]
        if need_test:
            tests.append(model.predict_proba(e)[:, 1].astype(np.float64))
        rows.append({
            "label": label, "fold": fold, "auc": roc_auc_score(y[va], oof[va]),
            "columns": a.shape[1], "nnz": a.nnz,
            "csr_mb": (a.data.nbytes + a.indices.nbytes + a.indptr.nbytes) / 1024**2,
            "peak_rss_gb": rss(), "seconds": perf_counter() - start,
            "effective_bins": str(effective),
        })
        del xa, xb, xt, encoder, model, a, b, e
        gc.collect()
    return oof, np.asarray(tests), rows


def fold_scores(y, p, splits):
    return np.asarray([roc_auc_score(y[v], p[v]) for _, v in splits])


def summarize(feature, spec, p, y, splits, baseline, train, test, stage):
    fs = fold_scores(y, p, splits)
    bfs = fold_scores(y, baseline, splits)
    tr = transform_nonquantile(values(train, feature), spec) if spec[0] != "quantile" else None
    te = transform_nonquantile(values(test, feature), spec) if spec[0] != "quantile" else None
    return {
        "stage": stage, "feature": feature, "method": spec[0], "parameter": spec[1],
        "auc": roc_auc_score(y, p), "delta": roc_auc_score(y, p) - roc_auc_score(y, baseline),
        "folds_improved": int(sum(fs > bfs)), "std": np.std(fs), "folds": str(fs.tolist()),
        "train_cardinality": tr.nunique(dropna=False) if tr is not None else int(spec[1]),
        "test_cardinality": te.nunique(dropna=False) if te is not None else np.nan,
    }


def nested_blend(y, base, logistic, splits, label):
    rb, rl = rank(base), rank(logistic)
    out = np.zeros(len(y)); rows = []
    for fold, (tr, va) in enumerate(splits, 1):
        opts = []
        for method in ["probability", "rank"]:
            for w in WEIGHTS:
                p = (1-w)*base+w*logistic if method == "probability" else (1-w)*rb+w*rl
                opts.append((roc_auc_score(y[tr], p[tr]), -w, method, w, p))
        z = max(opts, key=lambda q: (q[0], q[1]))
        out[va] = z[4][va]
        rows.append({"candidate": label, "fold": fold, "method": z[2],
                     "logistic_weight": z[3], "selection_auc": z[0]})
    fs = fold_scores(y, out, splits)
    rows.append({"candidate": label, "fold": 0, "auc": roc_auc_score(y, out),
                 "std": np.std(fs), "folds": str(fs.tolist())})
    return out, rows


def main():
    total_start = perf_counter()
    for d in [PRED, REPORTS, METRICS, SUB]:
        d.mkdir(parents=True, exist_ok=True)
    train, test = pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "test.csv")
    y = train.addicted_label.to_numpy()
    splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(train, y))
    base, basetest = base_rep(train), base_rep(test)
    p37 = load_oof(PRED / "oof_exp037_relational_logistic.csv", train)
    reproduced = roc_auc_score(y, p37)
    if abs(reproduced - EXP037_LOGISTIC) > 2e-6:
        raise RuntimeError(f"EXP-037 mismatch: {reproduced}")

    cache, progress = {}, []
    progress_path = REPORTS / "exp038_all_fold_progress.csv"

    def run(cfg, label, need_test=False):
        key = tuple(sorted((k, tuple(v)) for k, v in cfg.items()))
        if key not in cache or need_test:
            p, t, rows = fit_cv(train, test, base, basetest, y, splits, cfg, label, need_test)
            if not need_test:
                cache[key] = p
            progress.extend(rows)
            pd.DataFrame(progress).to_csv(progress_path, index=False)
            print(f"{label}: {roc_auc_score(y,p):.10f} elapsed={perf_counter()-total_start:.1f}", flush=True)
            return p, t
        return cache[key], np.empty((0, len(test)))

    # Screen-minus-social fixed/rounded variants. Baseline round1 is reused.
    screen_specs = [("round", 0.), ("round", 1.), ("round", 2.),
                    ("step", .25), ("step", .50), ("step", .75), ("step", 1.)]
    screen_rows = []
    candidate_predictions = {}
    for spec in screen_specs:
        cfg = dict(BASE_CFG); cfg["screen_minus_social"] = spec
        p = p37 if spec == BASE_CFG["screen_minus_social"] else run(cfg, f"screen_{spec[0]}_{spec[1]}")[0]
        row = summarize("screen_minus_social", spec, p, y, splits, p37, train, test, "screen")
        screen_rows.append(row); candidate_predictions[("screen_minus_social", spec)] = p
    pd.DataFrame(screen_rows).to_csv(REPORTS / "exp038_screen_minus_social_bins.csv", index=False)

    quant_rows = []
    for bins in [20, 40, 60, 80, 100]:
        spec = ("quantile", float(bins)); cfg = dict(BASE_CFG); cfg["screen_minus_social"] = spec
        p = run(cfg, f"screen_quantile_{bins}")[0]
        row = summarize("screen_minus_social", spec, p, y, splits, p37, train, test, "quantile")
        quant_rows.append(row); candidate_predictions[("screen_minus_social", spec)] = p
    pd.DataFrame(quant_rows).to_csv(REPORTS / "exp038_quantile_bins.csv", index=False)

    ratio_rows = []
    ratio_specs = [("round", 1.), ("round", 2.), ("round", 3.),
                   ("step", .01), ("step", .02), ("step", .025), ("step", .05), ("step", .10)]
    for feature in TOP:
        for spec in ratio_specs:
            cfg = dict(BASE_CFG); cfg[feature] = spec
            p = p37 if spec == BASE_CFG[feature] else run(cfg, f"{feature}_{spec[0]}_{spec[1]}")[0]
            row = summarize(feature, spec, p, y, splits, p37, train, test, "top_ratio")
            ratio_rows.append(row); candidate_predictions[(feature, spec)] = p
    top_improves = any(r["delta"] >= .00002 for r in ratio_rows)
    if top_improves:
        sec_specs = [("round", 1.), ("round", 2.), ("round", 3.),
                     ("step", .02), ("step", .05), ("step", .10)]
        for feature in SECONDARY:
            for spec in sec_specs:
                cfg = dict(BASE_CFG); cfg[feature] = spec
                p = p37 if spec == BASE_CFG[feature] else run(cfg, f"{feature}_{spec[0]}_{spec[1]}")[0]
                row = summarize(feature, spec, p, y, splits, p37, train, test, "secondary_ratio")
                ratio_rows.append(row); candidate_predictions[(feature, spec)] = p
    ratio_df = pd.DataFrame(ratio_rows)
    ratio_df.to_csv(REPORTS / "exp038_ratio_discretizations.csv", index=False)

    # Strict forward replacement selection.
    all_rows = screen_rows + quant_rows + ratio_rows
    eligible = [r for r in all_rows if r["delta"] >= .00002 and r["folds_improved"] >= 4
                and (r["feature"], (r["method"], r["parameter"])) !=
                    (r["feature"], BASE_CFG[r["feature"]])]
    cfg = dict(BASE_CFG); current_p = p37; current_auc = reproduced
    forward = [{"step": 0, "feature": "baseline", "auc": current_auc,
                "delta_incremental": 0., "folds_improved": 5, "accepted": True,
                "configuration": str(cfg)}]
    remaining = {(r["feature"], (r["method"], r["parameter"])) for r in eligible}
    for step in range(1, 4):
        trials = []
        for feature, spec in remaining:
            trial_cfg = dict(cfg); trial_cfg[feature] = spec
            # First-step OOFs already exist; later steps must be recalculated.
            if step == 1:
                p = candidate_predictions[(feature, spec)]
            else:
                p = run(trial_cfg, f"forward{step}_{feature}_{spec[0]}_{spec[1]}")[0]
            fnew, foldold = fold_scores(y, p, splits), fold_scores(y, current_p, splits)
            trials.append((roc_auc_score(y, p), int(sum(fnew > foldold)), feature, spec, p, fnew))
        if not trials:
            break
        best = max(trials, key=lambda z: z[0])
        delta = best[0] - current_auc
        accepted = delta >= .00002 and best[1] >= 4
        forward.append({"step": step, "feature": best[2], "method": best[3][0],
                        "parameter": best[3][1], "auc": best[0],
                        "delta_incremental": delta, "folds_improved": best[1],
                        "accepted": accepted, "folds": str(best[5].tolist())})
        if not accepted:
            break
        cfg[best[2]] = best[3]; current_auc, current_p = best[0], best[4]
        remaining = {x for x in remaining if x[0] != best[2]}
    pd.DataFrame(forward).to_csv(REPORTS / "exp038_forward_selection.csv", index=False)

    final_p, test_folds = run(cfg, "FINAL", need_test=True)
    final_test = np.mean(test_folds, axis=0, dtype=np.float64)
    final_auc = roc_auc_score(y, final_p); final_fs = fold_scores(y, final_p, splits)
    pd.DataFrame({"id": train.id, "y_true": y, "oof_prediction": final_p}).to_csv(
        PRED / "oof_exp038_discretized_logistic.csv", index=False)
    pd.DataFrame({"id": test.id, "prediction": final_test}).to_csv(
        PRED / "test_exp038_discretized_logistic.csv", index=False)

    # Cardinality and train->test transfer for the final non-quantile features.
    cards = []
    for feature, spec in cfg.items():
        if spec[0] == "quantile":
            cards.append({"feature": feature, "method": spec[0], "parameter": spec[1],
                          "train_cardinality": spec[1], "test_cardinality": spec[1],
                          "unseen_test_rate": 0., "coverage_test": 1.})
            continue
        a = transform_nonquantile(values(train, feature), spec)
        b = transform_nonquantile(values(test, feature), spec)
        seen = set(a.astype(str)); unseen = ~b.astype(str).isin(seen)
        cards.append({"feature": feature, "method": spec[0], "parameter": spec[1],
                      "train_cardinality": a.nunique(), "test_cardinality": b.nunique(),
                      "unseen_test_rate": unseen.mean(), "coverage_test": 1-unseen.mean()})
    card_df = pd.DataFrame(cards)
    card_df.to_csv(REPORTS / "exp038_cardinality.csv", index=False)

    p16 = load_oof(PRED / "oof_exp016_xgboost_depth5_9000.csv", train)
    p22 = load_oof(PRED / "oof_exp022_catboost_thresholds_9000.csv", train)
    seeds = [load_oof(PRED / f"oof_exp027_seed{x}.csv", train) for x in [42, 2026, 777]]
    p27 = .75*(.4*seeds[0]+.3*seeds[1]+.3*seeds[2])+.25*p22
    seeds5 = seeds + [load_oof(PRED / f"oof_exp028_seed{x}.csv", train) for x in [31415, 1234]]
    p28 = .75*np.mean(seeds5, axis=0)+.25*p22

    screen, social = train.daily_screen_time_hours, train.social_media_hours
    known = screen.notna() & social.notna()
    regions = {
        "clear_positive": known & ((screen > 8) | (social > 4)),
        "clear_negative": known & (screen <= 6) & (social <= 4),
    }
    regions["ambiguous"] = known & ~regions["clear_positive"] & ~regions["clear_negative"]
    regional = []
    for name, mask in regions.items():
        for model, p in [("EXP-038", final_p), ("EXP-037", p37), ("EXP-027", p27)]:
            regional.append({"analysis": "region", "segment": name, "model": model,
                             "rows": int(mask.sum()), "auc": roc_auc_score(y[mask], p[mask])})
    for name, lo, hi in [("0.30-0.40", .3, .4), ("0.40-0.50", .4, .5),
                         ("0.50-0.60", .5, .6), ("0.60-0.70", .6, .7),
                         ("0.70-0.80", .7, .8)]:
        mask = (p16 >= lo) & (p16 < hi)
        for model, p in [("EXP-038", final_p), ("EXP-037", p37), ("EXP-027", p27)]:
            regional.append({"analysis": "score_band", "segment": name, "model": model,
                             "rows": int(mask.sum()), "auc": roc_auc_score(y[mask], p[mask])})
    regional_df = pd.DataFrame(regional)
    regional_df.to_csv(REPORTS / "exp038_regional.csv", index=False)

    fold_ids = np.empty(len(y), int)
    for f, (_, va) in enumerate(splits): fold_ids[va] = f
    bp, bn, gp, gn, _ = sample_pairs(train, p16, np.random.default_rng(42), fold_ids)
    pair_rows = []
    for model, p in [("EXP-038", final_p), ("EXP-037", p37)]:
        corrected = np.mean(p[bp] > p[bn]); broken = np.mean(p[gp] <= p[gn])
        pair_rows.append({"model": model, "corrected": corrected, "broken": broken,
                          "net_gain": corrected-broken})
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(REPORTS / "exp038_pairwise.csv", index=False)

    blend_rows, blend_oofs = [], {}
    if final_auc - reproduced >= .00002:
        for name, p in [("EXP-027", p27), ("EXP-028", p28)]:
            z, rows = nested_blend(y, p, final_p, splits, name)
            blend_oofs[name] = z; blend_rows.extend(rows)
    blend_df = pd.DataFrame(blend_rows)
    blend_df.to_csv(REPORTS / "exp038_blends.csv", index=False)

    generated, subpath = False, ""
    if "EXP-027" in blend_oofs:
        z = blend_oofs["EXP-027"]
        zfs = fold_scores(y, z, splits)
        ref = .625*rank(p27)+.375*rank(p37)
        rfs = fold_scores(y, ref, splits)
        zauc = roc_auc_score(y, z)
        if zauc >= EXP037_ENSEMBLE+.00003 and sum(zfs > rfs) >= 4 and np.min(zfs-rfs) >= -.00003:
            choices = blend_df[(blend_df.candidate == "EXP-027") & (blend_df.fold > 0)]
            base_test = pd.read_csv(SUB / "submission_exp027_seed_ensemble.csv")
            sample = pd.read_csv(DATA / "sample_submission.csv")
            test_parts = []
            for choice, lp in zip(choices.itertuples(), test_folds):
                if choice.method == "rank":
                    test_parts.append((1-choice.logistic_weight)*rank(base_test.addicted_label)+choice.logistic_weight*rank(lp))
                else:
                    test_parts.append((1-choice.logistic_weight)*base_test.addicted_label.to_numpy()+choice.logistic_weight*lp)
            submission = pd.DataFrame({"id": sample.id,
                                       "addicted_label": np.mean(test_parts, axis=0, dtype=np.float64)})
            subpath = SUB / "submission_exp038_discretization_ensemble.csv"
            submission.to_csv(subpath, index=False); generated = True
            logpath = METRICS / "experiment_log.csv"; log = pd.read_csv(logpath)
            if not log.experiment_id.astype(str).eq("EXP-038").any():
                log = pd.concat([log, pd.DataFrame([{
                    "experiment_id": "EXP-038", "datetime": pd.Timestamp.now().isoformat(timespec="seconds"),
                    "model": "DiscretizedRelationalLogistic_Ensemble", "features": str(cfg),
                    "cv_strategy": "Nested_OOF_ensemble_optimization", "cv_roc_auc": zauc,
                    "kaggle_score": "", "notes": "fold-safe blend EXP027 + EXP038",
                }])], ignore_index=True); log.to_csv(logpath, index=False)

    problems = []
    if (card_df.unseen_test_rate > .005).any():
        problems.append("one or more final discretizations exceed 0.5% unseen test")
    lines = [
        "EXP-038 discretization refinement", f"EXP037_reproduced={reproduced}",
        f"screen=\n{pd.DataFrame(screen_rows).to_string(index=False)}",
        f"quantiles=\n{pd.DataFrame(quant_rows).to_string(index=False)}",
        f"ratios=\n{ratio_df.to_string(index=False)}", f"forward=\n{pd.DataFrame(forward).to_string(index=False)}",
        f"final_cfg={cfg}", f"final_auc={final_auc}; delta={final_auc-reproduced}; folds={final_fs.tolist()}; std={np.std(final_fs)}",
        f"cardinality=\n{card_df.to_string(index=False)}", f"regional=\n{regional_df.to_string(index=False)}",
        f"pairwise=\n{pair_df.to_string(index=False)}", f"blends=\n{blend_df.to_string(index=False)}",
        f"submission={generated}; path={subpath}", f"total_seconds={perf_counter()-total_start:.2f}",
        "problems=" + ("; ".join(problems) if problems else "none"),
    ]
    (METRICS / "exp038_discretization_metrics.txt").write_text("\n".join(lines)+"\n", encoding="utf8")
    print(f"FINAL auc={final_auc:.10f} submission={generated} elapsed={perf_counter()-total_start:.1f}")


if __name__ == "__main__":
    main()
