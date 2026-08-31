"""EXP-030: small, local, fully cross-fitted EXP-016/EXP-022 rank corrector."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from diagnose_exp029_pairwise_ranking import region_labels, sample_pairs


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
PRED = OUT / "predictions"
REPORTS = OUT / "reports"
METRICS = OUT / "metrics"
SUB = OUT / "submissions"

EXP016_AUC = .9657017303568276
EXP027_AUC = .965919188052602
EXP028_AUC = .9659307051390922
ALPHAS = [.05, .10, .15, .20, .25, .30, .40, .50]
SCOPES = ["ALL", "MID", "WIDE"]
TRANSFORMS = ["A_probability", "B_positive_confidence"]
ORIGINAL = ["daily_screen_time_hours", "weekend_screen_time", "social_media_hours",
            "gaming_hours", "sleep_hours", "work_study_hours"]
MODEL_PARAMS = dict(objective="binary", n_estimators=1000, learning_rate=.03, num_leaves=15,
                    max_depth=4, min_child_samples=100, subsample=.9, colsample_bytree=.9,
                    reg_alpha=1.0, reg_lambda=5.0, random_state=42, n_jobs=-1, verbosity=-1)


def load_oof(path: Path, train: pd.DataFrame) -> np.ndarray:
    df = pd.read_csv(path)
    if not df.id.equals(train.id) or not df.y_true.equals(train.addicted_label) or df.oof_prediction.isna().any():
        raise ValueError(f"OOF invalida: {path.name}")
    return df.oof_prediction.to_numpy(np.float64)


def make_features(train: pd.DataFrame, p16: np.ndarray, p22: np.ndarray) -> pd.DataFrame:
    X = pd.DataFrame(index=train.index)
    X["pred016"] = p16
    X["pred022"] = p22
    X["diff_022_016"] = p22 - p16
    X["abs_diff_022_016"] = np.abs(p22 - p16)
    for c in ORIGINAL:
        X[c] = train[c]
    screen = train.daily_screen_time_hours
    social = train.social_media_hours
    valid = screen.notna() & social.notna()
    cp = valid & ((screen > 8) | (social > 4))
    cn = valid & (screen.le(6) & social.le(4))
    amb = valid & ~cp & ~cn
    X["clear_positive_zone"] = np.where(valid, cp.astype(float), np.nan)
    X["clear_negative_zone"] = np.where(valid, cn.astype(float), np.nan)
    X["ambiguous_zone"] = np.where(valid, amb.astype(float), np.nan)
    X["region_code"] = np.select([cn, amb, cp], [0., 1., 2.], default=np.nan)
    X["in_040_065"] = ((p16 >= .40) & (p16 <= .65)).astype(np.int8)
    X["dist_to_050"] = np.abs(p16 - .50)
    X["dist_to_065"] = np.abs(p16 - .65)
    X["dist_to_040"] = np.abs(p16 - .40)
    return X


def fit_corrector(X, target, train_idx, eval_idx):
    model = LGBMClassifier(**MODEL_PARAMS)
    model.fit(X.iloc[train_idx], target[train_idx], eval_set=[(X.iloc[eval_idx], target[eval_idx])],
              eval_metric="auc", callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
    return model


def apply_correction(p16, p22, help_prob, transform, scope, alpha):
    gate = help_prob if transform == "A_probability" else np.maximum(0., 2 * help_prob - 1)
    if scope == "ALL":
        local = np.ones(len(p16), dtype=bool)
    elif scope == "MID":
        local = (p16 >= .40) & (p16 <= .65)
    else:
        local = (p16 >= .30) & (p16 <= .75)
    strength = gate * local
    return p16 + alpha * strength * (p22 - p16)


def choose_config(y, p16, p22, phelp):
    rows = []
    for transform in TRANSFORMS:
        for scope in SCOPES:
            for alpha in ALPHAS:
                p = apply_correction(p16, p22, phelp, transform, scope, alpha)
                rows.append({"transformation": transform, "scope": scope, "alpha": alpha,
                             "auc": float(roc_auc_score(y, p))})
    # Deterministic tie-breaking: smaller alpha, local scope, confidence-only transform.
    order_scope = {"MID": 0, "WIDE": 1, "ALL": 2}
    rows.sort(key=lambda r: (-r["auc"], r["alpha"], order_scope[r["scope"]], r["transformation"]))
    return rows[0], rows


def folds_auc(y, p, splits):
    return [float(roc_auc_score(y[va], p[va])) for _, va in splits]


def update_log(auc, choices):
    path = METRICS / "experiment_log.csv"
    cols = ["experiment_id", "datetime", "model", "features", "cv_strategy", "cv_roc_auc", "kaggle_score", "notes"]
    log = pd.read_csv(path, dtype=str, keep_default_na=False)
    if log.columns.tolist() != cols:
        raise ValueError("experiment_log.csv inesperado")
    log = log.loc[~log.experiment_id.eq("EXP-030")]
    summary = ";".join(f"F{int(r.fold)}:{r.scope}/{r.transformation}/a{r.alpha}" for r in choices.itertuples())
    row = {"experiment_id": "EXP-030", "datetime": datetime.now().astimezone().isoformat(timespec="seconds"),
           "model": "Nested_Local_Rank_Corrector", "features": "EXP016_EXP022_local_context",
           "cv_strategy": "Nested_StratifiedKFold", "cv_roc_auc": f"{auc:.8f}", "kaggle_score": "",
           "notes": summary}
    pd.concat([log, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)


def main():
    start = perf_counter()
    for d in [PRED, REPORTS, METRICS, SUB]:
        d.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA / "train.csv")
    y = train.addicted_label.to_numpy()
    p16 = load_oof(PRED / "oof_exp016_xgboost_depth5_9000.csv", train)
    p22 = load_oof(PRED / "oof_exp022_catboost_thresholds_9000.csv", train)
    if abs(roc_auc_score(y, p16) - EXP016_AUC) > 2e-8:
        raise ValueError("EXP-016 no reproduce")
    X = make_features(train, p16, p22)
    target = np.where(y == 1, p22 > p16, p22 < p16).astype(np.int8)
    splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))

    corrected_oof = np.zeros(len(train), dtype=np.float64)
    help_oof = np.zeros(len(train), dtype=np.float64)
    choices = []
    final_models = []
    all_search = []
    for outer_fold, (outer_train, outer_valid) in enumerate(splits, 1):
        # Cross-fitted help probabilities only within outer_train for honest config selection.
        inner_help = np.zeros(len(outer_train), dtype=np.float64)
        inner_target = target[outer_train]
        inner_cv = StratifiedKFold(4, shuffle=True, random_state=123)
        for itr_rel, iva_rel in inner_cv.split(np.zeros(len(outer_train)), inner_target):
            model = fit_corrector(X, target, outer_train[itr_rel], outer_train[iva_rel])
            inner_help[iva_rel] = model.predict_proba(X.iloc[outer_train[iva_rel]], num_iteration=model.best_iteration_)[:, 1]
        choice, search = choose_config(y[outer_train], p16[outer_train], p22[outer_train], inner_help)
        for r in search:
            all_search.append({"outer_fold": outer_fold, **r})

        fit_rel, es_rel = train_test_split(np.arange(len(outer_train)), test_size=.15, random_state=100 + outer_fold,
                                          stratify=inner_target)
        final_model = fit_corrector(X, target, outer_train[fit_rel], outer_train[es_rel])
        final_models.append(final_model)
        hv = final_model.predict_proba(X.iloc[outer_valid], num_iteration=final_model.best_iteration_)[:, 1]
        help_oof[outer_valid] = hv
        pv = apply_correction(p16[outer_valid], p22[outer_valid], hv, choice["transformation"], choice["scope"], choice["alpha"])
        corrected_oof[outer_valid] = pv
        auc16 = float(roc_auc_score(y[outer_valid], p16[outer_valid]))
        auc30 = float(roc_auc_score(y[outer_valid], pv))
        help_auc = float(roc_auc_score(target[outer_valid], hv))
        choices.append({"fold": outer_fold, "scope": choice["scope"], "transformation": choice["transformation"],
                        "alpha": choice["alpha"], "selection_auc_outer_train": choice["auc"],
                        "corrector_auc": help_auc, "auc_exp016": auc16, "auc_exp030": auc30,
                        "delta": auc30 - auc16, "best_iteration": int(final_model.best_iteration_)})
        print(f"fold={outer_fold} help_auc={help_auc:.6f} {choice} valid_auc={auc30:.8f}", flush=True)

    choices_df = pd.DataFrame(choices)
    choices_df.to_csv(REPORTS / "exp030_fold_choices.csv", index=False)
    auc30 = float(roc_auc_score(y, corrected_oof))
    auc16 = float(roc_auc_score(y, p16))
    folds30 = folds_auc(y, corrected_oof, splits)
    folds16 = folds_auc(y, p16, splits)

    # Reconstruct persisted ensemble OOF controls without using them as corrector inputs.
    s42 = load_oof(PRED / "oof_exp027_seed42.csv", train)
    s2026 = load_oof(PRED / "oof_exp027_seed2026.csv", train)
    s777 = load_oof(PRED / "oof_exp027_seed777.csv", train)
    s31415 = load_oof(PRED / "oof_exp028_seed31415.csv", train)
    s1234 = load_oof(PRED / "oof_exp028_seed1234.csv", train)
    p27 = .75 * (.4 * s42 + .3 * s2026 + .3 * s777) + .25 * p22
    p28 = .75 * ((s42 + s2026 + s777 + s31415 + s1234) / 5) + .25 * p22
    global_blend = .75 * p16 + .25 * p22
    controls = {"EXP-016": p16, "EXP-022": p22, "global_75_25": global_blend,
                "EXP-027": p27, "EXP-028": p28, "EXP-030": corrected_oof}

    metrics_rows = []
    for name, p in controls.items():
        fs = folds_auc(y, p, splits)
        metrics_rows.append({"model": name, "auc": roc_auc_score(y, p), "fold1": fs[0], "fold2": fs[1],
                             "fold3": fs[2], "fold4": fs[3], "fold5": fs[4], "std": np.std(fs),
                             "log_loss": log_loss(y, np.clip(p, 1e-15, 1 - 1e-15)),
                             "brier": brier_score_loss(y, p)})
    metrics_df = pd.DataFrame(metrics_rows)

    # Fold-safe pair construction identical to final EXP-029 methodology.
    fold_ids = np.empty(len(train), dtype=np.int8)
    for fold, (_, va) in enumerate(splits):
        fold_ids[va] = fold
    rng = np.random.default_rng(42)
    for fold in range(5):  # consume the same random-pair draws used before hard-pair construction
        fp = np.flatnonzero((y == 1) & (fold_ids == fold)); fn = np.flatnonzero((y == 0) & (fold_ids == fold))
        rng.choice(fp, 20_000, replace=len(fp) < 20_000); rng.choice(fn, 20_000, replace=len(fn) < 20_000)
    bp, bn, gp, gn, _ = sample_pairs(train, p16, rng, fold_ids)
    pair_rows = []
    for name, p in controls.items():
        corrected = float(np.mean(p[bp] > p[bn]))
        broken = float(np.mean(p[gp] <= p[gn]))
        pair_rows.append({"model": name, "misordered_pairs": len(bp), "corrected_misordered_rate": corrected,
                          "broken_correct_rate": broken, "net_pair_gain": corrected - broken})
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(REPORTS / "exp030_pairwise_results.csv", index=False)

    band_defs = [("0.30-0.40", .30, .40), ("0.40-0.45", .40, .45), ("0.45-0.50", .45, .50),
                 ("0.50-0.55", .50, .55), ("0.55-0.60", .55, .60), ("0.60-0.65", .60, .65),
                 ("0.65-0.75", .65, .75)]
    band_rows = []
    for label, lo, hi in band_defs:
        mask = (p16 >= lo) & (p16 < hi)
        for name in ["EXP-016", "EXP-022", "EXP-030"]:
            yy = y[mask]; pp = controls[name][mask]
            band_rows.append({"analysis": "score_band", "segment": label, "model": name, "rows": mask.sum(),
                              "auc": roc_auc_score(yy, pp) if len(np.unique(yy)) == 2 else np.nan})
    regions = region_labels(train)
    for reg in ["clear_positive", "clear_negative", "ambiguous"]:
        mask = regions.eq(reg).to_numpy()
        for name in ["EXP-016", "EXP-022", "EXP-030"]:
            yy = y[mask]; pp = controls[name][mask]
            band_rows.append({"analysis": "region", "segment": reg, "model": name, "rows": mask.sum(),
                              "auc": roc_auc_score(yy, pp) if len(np.unique(yy)) == 2 else np.nan})
    band_df = pd.DataFrame(band_rows)
    band_df.to_csv(REPORTS / "exp030_band_results.csv", index=False)

    delta27 = auc30 - EXP027_AUC
    delta28 = auc30 - EXP028_AUC
    improved_vs27 = int(np.sum(np.asarray(folds30) > np.asarray(folds_auc(y, p27, splits))))
    success = (delta27 >= .00005 or (delta27 >= .00002 and improved_vs27 >= 4)) and delta28 > 0
    submission_generated = False
    submission_path = "none"
    if success:
        test = pd.read_csv(DATA / "test.csv")
        sample = pd.read_csv(DATA / "sample_submission.csv")
        t16 = pd.read_csv(PRED / "test_exp016_xgboost_depth5_9000.csv").prediction.to_numpy(np.float64)
        t22 = pd.read_csv(PRED / "test_exp022_catboost_thresholds_9000.csv").prediction.to_numpy(np.float64)
        Xt = make_features(test, t16, t22)
        test_sum = np.zeros(len(test), dtype=np.float64)
        for model, choice in zip(final_models, choices):
            ht = model.predict_proba(Xt, num_iteration=model.best_iteration_)[:, 1]
            test_sum += apply_correction(t16, t22, ht, choice["transformation"], choice["scope"], choice["alpha"]) / 5
        sub = pd.DataFrame({"id": sample.id, "addicted_label": test_sum})
        if len(sub) != 296302 or not sub.id.equals(sample.id) or sub.isna().any().any() or not sub.addicted_label.between(0, 1).all():
            raise ValueError("Submission invalida")
        path = SUB / "submission_exp030_rank_corrector.csv"
        sub.to_csv(path, index=False)
        submission_generated = True; submission_path = str(path)
        update_log(auc30, choices_df)

    pd.DataFrame({"id": train.id, "y_true": y, "prediction": corrected_oof,
                  "p_catboost_help": help_oof, "fold": fold_ids + 1}).to_csv(PRED / "oof_exp030_rank_corrector.csv", index=False)
    elapsed = perf_counter() - start
    lines = ["EXP-030 nested local rank corrector", f"model_params={MODEL_PARAMS}",
             f"corrector_auc_folds={choices_df.corrector_auc.tolist()}; mean={choices_df.corrector_auc.mean():.8f}; std={choices_df.corrector_auc.std(ddof=0):.8f}",
             "fold_choices:\n" + choices_df.to_string(index=False), "global_metrics:\n" + metrics_df.to_string(index=False),
             f"delta_vs_exp016={auc30-auc16:.12f}; delta_vs_exp027={delta27:.12f}; delta_vs_exp028={delta28:.12f}",
             "pairwise:\n" + pair_df.to_string(index=False), "bands_regions:\n" + band_df.to_string(index=False),
             f"success={success}; submission_generated={submission_generated}; path={submission_path}",
             f"total_seconds={elapsed:.2f}", "problems=none"]
    (METRICS / "exp030_rank_corrector_metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"corrector_auc={choices_df.corrector_auc.mean():.8f}; exp030={auc30:.12f}; d27={delta27:.12f}; d28={delta28:.12f}")
    print(f"submission={submission_generated}; path={submission_path}; seconds={elapsed:.2f}")


if __name__ == "__main__":
    main()
