"""EXP-029: diagnose ranking errors in EXP-016 without changing predictions."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
PRED = OUT / "predictions"
REPORTS = OUT / "reports"
METRICS = OUT / "metrics"

NUMERIC = ["age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
           "notifications_per_day", "app_opens_per_day", "sleep_hours", "work_study_hours",
           "weekend_screen_time"]
CATEGORICAL = ["gender", "stress_level", "academic_work_impact"]
PCTS = [1, 5, 10, 20]


def load_oof(path: Path, train: pd.DataFrame) -> np.ndarray:
    df = pd.read_csv(path)
    if not df.id.equals(train.id) or not df.y_true.equals(train.addicted_label):
        raise ValueError(f"OOF desalineada: {path.name}")
    if df.oof_prediction.isna().any():
        raise ValueError(f"NaN en {path.name}")
    return df.oof_prediction.to_numpy(np.float64)


def region_labels(df: pd.DataFrame) -> pd.Series:
    screen = df.daily_screen_time_hours
    social = df.social_media_hours
    valid = screen.notna() & social.notna()
    pos = valid & ((screen > 8) | (social > 4))
    neg = valid & (screen <= 6) & (social <= 4)
    out = pd.Series("unknown", index=df.index, dtype="object")
    out.loc[pos] = "clear_positive"
    out.loc[neg] = "clear_negative"
    out.loc[valid & ~pos & ~neg] = "ambiguous"
    return out


def hard_case_summaries(train, score, region):
    y = train.addicted_label.to_numpy()
    rows, masks = [], {}
    pos_scores = score[y == 1]
    neg_scores = score[y == 0]
    for pct in PCTS:
        hp_thr = np.quantile(pos_scores, pct / 100)
        hn_thr = np.quantile(neg_scores, 1 - pct / 100)
        for name, mask, threshold in [
            (f"hard_positive_{pct}", (y == 1) & (score <= hp_thr), hp_thr),
            (f"hard_negative_{pct}", (y == 0) & (score >= hn_thr), hn_thr),
        ]:
            masks[name] = mask
            sub = train.loc[mask]
            reg = region.loc[mask].value_counts(normalize=True)
            miss = sub[NUMERIC + CATEGORICAL].isna().mean().mean()
            row = {"group": name, "rows": int(mask.sum()), "score_threshold": threshold,
                   "score_mean": float(score[mask].mean()), "score_median": float(np.median(score[mask])),
                   "mean_missing_rate": float(miss)}
            for r in ["clear_positive", "clear_negative", "ambiguous", "unknown"]:
                row[f"region_{r}"] = float(reg.get(r, 0))
            for c in CATEGORICAL:
                dist = sub[c].fillna("__MISSING__").value_counts(normalize=True)
                row[f"{c}_mode"] = str(dist.index[0]) if len(dist) else ""
                row[f"{c}_mode_share"] = float(dist.iloc[0]) if len(dist) else np.nan
            rows.append(row)
    return pd.DataFrame(rows), masks


def feature_shifts(train, masks):
    rows = []
    for name in ["hard_positive_5", "hard_negative_5", "hard_positive_10", "hard_negative_10"]:
        mask = masks[name]
        same_class = train.addicted_label.eq(1 if "positive" in name else 0).to_numpy()
        rest = same_class & ~mask
        for c in NUMERIC:
            a = train.loc[mask, c].dropna().to_numpy()
            b = train.loc[rest, c].dropna().to_numpy()
            pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2) if len(a) > 1 and len(b) > 1 else np.nan
            smd = (np.mean(a) - np.mean(b)) / pooled if pooled and np.isfinite(pooled) else np.nan
            rows.append({"group": name, "feature": c, "type": "numeric", "group_mean": np.mean(a),
                         "rest_mean": np.mean(b), "group_median": np.median(a), "rest_median": np.median(b),
                         "standardized_mean_difference": smd,
                         "distance": float(ks_2samp(a, b).statistic), "distance_name": "KS"})
        for c in CATEGORICAL:
            a = train.loc[mask, c].fillna("__MISSING__").value_counts(normalize=True)
            b = train.loc[rest, c].fillna("__MISSING__").value_counts(normalize=True)
            levels = a.index.union(b.index)
            tv = .5 * np.abs(a.reindex(levels, fill_value=0) - b.reindex(levels, fill_value=0)).sum()
            rows.append({"group": name, "feature": c, "type": "categorical", "group_mean": np.nan,
                         "rest_mean": np.nan, "group_median": np.nan, "rest_median": np.nan,
                         "standardized_mean_difference": np.nan, "distance": float(tv), "distance_name": "TV"})
    out = pd.DataFrame(rows)
    out["abs_smd"] = out.standardized_mean_difference.abs()
    return out


def sample_pairs(train, score, rng, fold_ids):
    y = train.addicted_label.to_numpy()
    all_bp, all_bn, all_gp, all_gn, all_pf = [], [], [], [], []
    # Construct pairs strictly within each original fold. Holding out pair_fold therefore
    # holds out every underlying train row as well.
    for fold in range(5):
        pos_all = np.flatnonzero((y == 1) & (fold_ids == fold))
        neg_all = np.flatnonzero((y == 0) & (fold_ids == fold))
        pos_sample = rng.choice(pos_all, min(10_000, len(pos_all)), replace=False)
        neg_sample = rng.choice(neg_all, min(10_000, len(neg_all)), replace=False)
        neg_sorted = neg_sample[np.argsort(score[neg_sample])]
        neg_scores = score[neg_sorted]
        bp, bn, gp, gn = [], [], [], []
        for pi in pos_sample:
            cut = np.searchsorted(neg_scores, score[pi], side="left")
            if cut < len(neg_sorted):
                available = len(neg_sorted) - cut
                take = min(4, available)
                offsets = rng.integers(0, available, size=take)
                bp.extend([pi] * take); bn.extend(neg_sorted[cut + offsets].tolist())
            lower = np.searchsorted(neg_scores, score[pi], side="left")
            if lower > 0:
                take = min(4, lower)
                offsets = rng.integers(0, min(lower, 100), size=take)
                gp.extend([pi] * take); gn.extend(neg_sorted[lower - 1 - offsets].tolist())
        n = min(len(bp), len(gp), 40_000)
        all_bp.extend(bp[:n]); all_bn.extend(bn[:n]); all_gp.extend(gp[:n]); all_gn.extend(gn[:n]); all_pf.extend([fold] * n)
    return (np.asarray(all_bp), np.asarray(all_bn), np.asarray(all_gp), np.asarray(all_gn),
            np.asarray(all_pf, dtype=np.int8))


def pair_frame(train, score, bp, bn, gp, gn, pair_folds):
    pidx = np.concatenate([bp, gp])
    nidx = np.concatenate([bn, gn])
    target = np.concatenate([np.ones(len(bp), dtype=np.int8), np.zeros(len(gp), dtype=np.int8)])
    data = {"target_pair_error": target, "score_gap": score[pidx] - score[nidx],
            "positive_index": pidx, "negative_index": nidx,
            "pair_fold": np.concatenate([pair_folds, pair_folds])}
    for c in NUMERIC:
        a = train[c].to_numpy(np.float64)[pidx]
        b = train[c].to_numpy(np.float64)[nidx]
        data[f"diff_{c}"] = a - b
        data[f"absdiff_{c}"] = np.abs(a - b)
    for c in CATEGORICAL:
        a = train[c].fillna("__MISSING__").astype(str).to_numpy()[pidx]
        b = train[c].fillna("__MISSING__").astype(str).to_numpy()[nidx]
        data[f"same_{c}"] = (a == b).astype(np.int8)
    return pd.DataFrame(data)


def fit_pair_models(pairs):
    feature_cols = [c for c in pairs if c.startswith(("diff_", "absdiff_", "same_"))]
    X = pairs[feature_cols]
    y = pairs.target_pair_error.to_numpy()
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
                     ("model", LogisticRegression(C=1.0, solver="lbfgs", max_iter=3000, random_state=42))])
    scores = []
    for fold in range(5):
        tr = pairs.pair_fold.to_numpy() != fold; va = ~tr
        pipe.fit(X.loc[tr], y[tr]); scores.append(float(roc_auc_score(y[va], pipe.predict_proba(X.loc[va])[:, 1])))
    scores = np.asarray(scores)
    pipe.fit(X, y)
    coef = pipe.named_steps["model"].coef_[0]
    importance = pd.DataFrame({"feature": feature_cols, "logistic_coefficient": coef,
                               "abs_logistic_coefficient": np.abs(coef)}).sort_values("abs_logistic_coefficient", ascending=False)

    lgb_scores = None
    lgb_imp = pd.Series(dtype=float)
    if HAS_LGBM:
        lgb_scores = []
        imps = []
        Ximp = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=feature_cols)
        for fold in range(5):
            tr = pairs.pair_fold.to_numpy() != fold; va = ~tr
            model = LGBMClassifier(objective="binary", n_estimators=500, learning_rate=.05, num_leaves=15,
                                   max_depth=5, min_child_samples=50, reg_lambda=1.0,
                                   random_state=42, n_jobs=-1, verbosity=-1)
            model.fit(Ximp.loc[tr], y[tr], eval_set=[(Ximp.loc[va], y[va])], eval_metric="auc",
                      callbacks=[early_stopping(50, verbose=False), log_evaluation(0)])
            pv = model.predict_proba(Ximp.iloc[va], num_iteration=model.best_iteration_)[:, 1]
            lgb_scores.append(float(roc_auc_score(y[va], pv)))
            imps.append(model.booster_.feature_importance(importance_type="gain"))
        lgb_imp = pd.Series(np.mean(imps, axis=0), index=feature_cols).sort_values(ascending=False)
        importance["lightgbm_gain"] = importance.feature.map(lgb_imp)
    return scores, lgb_scores, importance


def score_bands(train, score, masks):
    edges = np.arange(0, 1.00001, .1)
    rows = []
    definitions = [(f"{edges[i]:.2f}-{edges[i+1]:.2f}", edges[i], edges[i+1]) for i in range(10)]
    definitions += [(f"{x:.2f}-{x+.05:.2f}", x, x + .05) for x in [.40, .45, .50, .55, .60]]
    y = train.addicted_label.to_numpy()
    for label, lo, hi in definitions:
        mask = (score >= lo) & ((score < hi) if hi < 1 else (score <= hi))
        yy = y[mask]
        auc = float(roc_auc_score(yy, score[mask])) if len(np.unique(yy)) == 2 else np.nan
        rows.append({"band": label, "lower": lo, "upper": hi, "rows": int(mask.sum()),
                     "target_rate": float(yy.mean()) if len(yy) else np.nan, "local_auc": auc,
                     "hard_positive_5_share": float((mask & masks["hard_positive_5"]).sum() / max(mask.sum(), 1)),
                     "hard_negative_5_share": float((mask & masks["hard_negative_5"]).sum() / max(mask.sum(), 1))})
    return pd.DataFrame(rows)


def main():
    start = perf_counter()
    REPORTS.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA / "train.csv")
    y = train.addicted_label.to_numpy()
    predictions = {
        "EXP-016": load_oof(PRED / "oof_exp016_xgboost_depth5_9000.csv", train),
        "EXP-022": load_oof(PRED / "oof_exp022_catboost_thresholds_9000.csv", train),
    }
    seeds = {s: load_oof(PRED / f"oof_exp027_seed{s}.csv", train) for s in [42, 2026, 777]}
    xgb3 = .4 * seeds[42] + .3 * seeds[2026] + .3 * seeds[777]
    predictions["EXP-027"] = .75 * xgb3 + .25 * predictions["EXP-022"]
    seeds[31415] = load_oof(PRED / "oof_exp028_seed31415.csv", train)
    seeds[1234] = load_oof(PRED / "oof_exp028_seed1234.csv", train)
    xgb5 = sum(seeds[s] for s in [42, 2026, 777, 31415, 1234]) / 5
    predictions["EXP-028"] = .75 * xgb5 + .25 * predictions["EXP-022"]
    aucs = {k: float(roc_auc_score(y, v)) for k, v in predictions.items()}
    if abs(aucs["EXP-016"] - .96570173) > 2e-6:
        raise ValueError(f"EXP-016 no reproduce: {aucs['EXP-016']}")
    score = predictions["EXP-016"]
    region = region_labels(train)

    hard_df, masks = hard_case_summaries(train, score, region)
    hard_df.to_csv(REPORTS / "exp029_hard_cases.csv", index=False)
    shifts = feature_shifts(train, masks)
    shifts.to_csv(REPORTS / "exp029_feature_shifts.csv", index=False)

    rng = np.random.default_rng(42)
    fold_ids = np.empty(len(train), dtype=np.int8)
    for fold, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(train, y)):
        fold_ids[va] = fold
    random_pair_pos, random_pair_neg = [], []
    for fold in range(5):
        fp = np.flatnonzero((y == 1) & (fold_ids == fold))
        fn = np.flatnonzero((y == 0) & (fold_ids == fold))
        random_pair_pos.append(rng.choice(fp, 20_000, replace=len(fp) < 20_000))
        random_pair_neg.append(rng.choice(fn, 20_000, replace=len(fn) < 20_000))
    random_pair_pos = np.concatenate(random_pair_pos)
    random_pair_neg = np.concatenate(random_pair_neg)
    random_pair_misordered_rate = float(np.mean(score[random_pair_pos] <= score[random_pair_neg]))
    bp, bn, gp, gn, pair_folds = sample_pairs(train, score, rng, fold_ids)
    pairs = pair_frame(train, score, bp, bn, gp, gn, pair_folds)
    logistic_scores, lgb_scores, pair_importance = fit_pair_models(pairs)
    pair_importance.to_csv(REPORTS / "exp029_pair_error_features.csv", index=False)

    # Secondary-model behavior on hard single-class groups and paired cases.
    comparison_rows = []
    for group in ["hard_positive_5", "hard_positive_10", "hard_negative_5", "hard_negative_10"]:
        mask = masks[group]
        for model, p in predictions.items():
            comparison_rows.append({"analysis": "hard_group_score", "group": group, "model": model,
                                    "rows": int(mask.sum()), "mean_prediction": float(p[mask].mean()),
                                    "median_prediction": float(np.median(p[mask])), "corrected_rate": np.nan,
                                    "broken_rate": np.nan, "net_pair_gain": np.nan})
    for model, p in predictions.items():
        corrected = float(np.mean(p[bp] > p[bn]))
        broken = float(np.mean(p[gp] <= p[gn]))
        comparison_rows.append({"analysis": "pairwise", "group": "misordered_vs_correct_close", "model": model,
                                "rows": len(bp), "mean_prediction": np.nan, "median_prediction": np.nan,
                                "corrected_rate": corrected, "broken_rate": broken,
                                "net_pair_gain": corrected - broken})
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(REPORTS / "exp029_pairwise_model_comparison.csv", index=False)

    bands = score_bands(train, score, masks)
    bands.to_csv(REPORTS / "exp029_score_bands.csv", index=False)

    # Hardness correlations: positive low percentile and negative inverted high percentile.
    hardness = np.empty(len(train), dtype=np.float64)
    pos = y == 1
    neg = ~pos
    hardness[pos] = 1 - pd.Series(score[pos]).rank(pct=True).to_numpy()
    hardness[neg] = pd.Series(score[neg]).rank(pct=True).to_numpy()
    hardness_corr = []
    for cls, mask in [(1, pos), (0, neg)]:
        for c in NUMERIC:
            valid = mask & train[c].notna().to_numpy()
            hardness_corr.append({"target_class": cls, "feature": c,
                                  "pearson": pd.Series(hardness[valid]).corr(train.loc[valid, c].reset_index(drop=True)),
                                  "spearman": pd.Series(hardness[valid]).corr(train.loc[valid, c].reset_index(drop=True), method="spearman")})
    hardness_corr = pd.DataFrame(hardness_corr)

    # Simple interpretable rules derived only from the strongest signed-difference coefficients.
    rules = []
    for feature in pair_importance.feature:
        if not feature.startswith("diff_"):
            continue
        c = feature.removeprefix("diff_")
        vals = pairs[feature]
        for direction, cond in [("positive_gt_negative", vals > 0), ("positive_le_negative", vals <= 0)]:
            coverage = float(cond.mean())
            precision = float(pairs.loc[cond, "target_pair_error"].mean()) if cond.any() else np.nan
            rules.append({"feature": c, "direction": direction, "coverage": coverage,
                          "pair_error_precision": precision, "lift_vs_base": precision - .5})
        if len({r["feature"] for r in rules}) >= 6:
            break
    rules_df = pd.DataFrame(rules).sort_values("pair_error_precision", ascending=False)

    top_shift_lines = []
    for group in ["hard_positive_5", "hard_negative_5", "hard_positive_10", "hard_negative_10"]:
        top = shifts[shifts.group.eq(group)].sort_values("distance", ascending=False).head(20)
        top_shift_lines.append(group + ":\n" + top[["feature", "type", "standardized_mean_difference", "distance", "distance_name"]].to_string(index=False))
    pair_cmp = comparison[comparison.analysis.eq("pairwise")]
    secondary_best = pair_cmp.loc[pair_cmp.model.ne("EXP-016")].sort_values("net_pair_gain", ascending=False).iloc[0]
    log_auc = float(np.mean(logistic_scores))
    lgb_auc = float(np.mean(lgb_scores)) if lgb_scores is not None else np.nan
    broad_signal = (log_auc >= .60 or (np.isfinite(lgb_auc) and lgb_auc >= .60))
    secondary_signal = secondary_best.net_pair_gain > 0.01
    promising = broad_signal and secondary_signal

    elapsed = perf_counter() - start
    lines = [
        "EXP-029 pairwise ranking diagnostic",
        f"model_aucs={aucs}",
        "hard_cases:\n" + hard_df.to_string(index=False),
        "feature_shifts_top20:\n" + "\n\n".join(top_shift_lines),
        f"misordered_pairs={len(bp)}; correct_close_pairs={len(gp)}",
        f"random_pairs={len(random_pair_pos)}; random_pair_misordered_rate={random_pair_misordered_rate:.8f}",
        f"pair_logistic_folds={logistic_scores.tolist()}; mean={log_auc:.8f}; std={np.std(logistic_scores):.8f}",
        f"pair_lightgbm_folds={None if lgb_scores is None else lgb_scores}; mean={lgb_auc}",
        "pair_feature_importance:\n" + pair_importance.head(20).to_string(index=False),
        "hardness_correlations:\n" + hardness_corr.sort_values("spearman", key=lambda s: s.abs(), ascending=False).to_string(index=False),
        "pairwise_model_comparison:\n" + comparison.to_string(index=False),
        "score_bands:\n" + bands.to_string(index=False),
        "simple_rules:\n" + rules_df.to_string(index=False),
        f"promising_signal={promising}; pair_structure={broad_signal}; secondary_net_signal={secondary_signal}",
        f"total_seconds={elapsed:.2f}",
        "problems=none",
    ]
    (METRICS / "exp029_pairwise_ranking.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"AUCs={aucs}")
    print(f"pairs={len(bp)}; logistic={log_auc:.8f}; lightgbm={lgb_auc}")
    print(f"best_secondary={secondary_best.to_dict()}; promising={promising}; seconds={elapsed:.2f}")


if __name__ == "__main__":
    main()
