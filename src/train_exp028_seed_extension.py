"""EXP-028: extend the exact EXP-016 XGBoost seed bag with seeds 31415 and 1234."""

from __future__ import annotations

from datetime import datetime
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from train_xgboost_exp008 import MODEL_PARAMS, ordinal_encode_categories
from train_xgboost_exp012_threshold_features import NEW_FEATURES, add_threshold_features


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
PRED = OUT / "predictions"
REPORTS = OUT / "reports"
METRICS = OUT / "metrics"
SUB = OUT / "submissions"

NEW_SEEDS = [31415, 1234]
ALL_SEEDS = [42, 2026, 777, 31415, 1234]
EARLY = 300
MODEL = dict(MODEL_PARAMS)
MODEL.update({"max_depth": 5, "n_estimators": 9000})
EXP016 = 0.9657017303568276
EXP027 = 0.965919188052602
EXP027_FOLDS = np.array([0.9652142162573694, 0.9658410142721825, 0.9661312017426842,
                         0.9667082920298451, 0.96569869782973])
BASE_WEIGHTS = {42: 0.40, 2026: 0.30, 777: 0.30}


def ranks(a: np.ndarray) -> np.ndarray:
    r = pd.Series(a).rank(method="average").to_numpy(np.float64)
    return (r - 1.0) / (len(r) - 1.0)


def fold_aucs(y: np.ndarray, p: np.ndarray, splits) -> list[float]:
    return [float(roc_auc_score(y[va], p[va])) for _, va in splits]


def blend(preds: dict[int, np.ndarray], weights: dict[int, float], method: str) -> np.ndarray:
    values = preds if method == "probability" else {s: ranks(v) for s, v in preds.items()}
    return sum(weights[s] * values[s] for s in weights)


def result(name, stage, method, weights, preds, y, splits, baseline_auc):
    p = blend(preds, weights, method)
    fs = fold_aucs(y, p, splits)
    return {"stage": stage, "name": name, "method": method,
            "weights": ";".join(f"{s}:{weights[s]:.6f}" for s in weights),
            "auc": float(roc_auc_score(y, p)), "delta": float(roc_auc_score(y, p) - baseline_auc),
            "fold1": fs[0], "fold2": fs[1], "fold3": fs[2], "fold4": fs[3], "fold5": fs[4],
            "std": float(np.std(fs)), "prediction": p, "weight_dict": weights}


def update_log(auc: float, method: str, weights: dict[int, float], cat_weight: float):
    path = METRICS / "experiment_log.csv"
    cols = ["experiment_id", "datetime", "model", "features", "cv_strategy", "cv_roc_auc", "kaggle_score", "notes"]
    log = pd.read_csv(path, dtype=str, keep_default_na=False)
    if log.columns.tolist() != cols:
        raise ValueError("experiment_log.csv tiene columnas inesperadas")
    log.loc[log.experiment_id.eq("EXP-027"), "kaggle_score"] = "0.96736"
    log = log.loc[~log.experiment_id.eq("EXP-028")]
    notes = f"{method}; seed_weights={weights}; catboost_weight={cat_weight}"
    row = {"experiment_id": "EXP-028", "datetime": datetime.now().astimezone().isoformat(timespec="seconds"),
           "model": "XGBoost_5Seed_Ensemble_CatBoost", "features": "exp012_threshold_region_features",
           "cv_strategy": "StratifiedKFold_5_seed_ensemble", "cv_roc_auc": f"{auc:.8f}",
           "kaggle_score": "", "notes": notes}
    pd.concat([log, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)


def main():
    start_all = perf_counter()
    for d in [PRED, REPORTS, METRICS, SUB]:
        d.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    features = [c for c in train.columns if c not in {"id", "addicted_label"}]
    raw_train = add_threshold_features(train[features])
    raw_test = add_threshold_features(test[features])
    if raw_train.columns.tolist() != raw_test.columns.tolist() or len(raw_train.columns) != len(features) + len(NEW_FEATURES):
        raise ValueError("El esquema no coincide con EXP-016")
    cats = [c for c in features if not pd.api.types.is_numeric_dtype(raw_train[c])]
    numeric = [c for c in raw_train if c not in cats]
    X, Xtest, mappings = ordinal_encode_categories(raw_train, raw_test, cats)
    if not X[numeric].equals(raw_train[numeric]) or not Xtest[numeric].equals(raw_test[numeric]):
        raise ValueError("Se alteraron features numericas")

    y_series = train["addicted_label"]
    y = y_series.to_numpy()
    splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
    fold_rows = []
    new_oof = {}
    new_test = {}
    best_iterations = {}

    for seed in NEW_SEEDS:
        oof_path = PRED / f"oof_exp028_seed{seed}.csv"
        test_path = PRED / f"test_exp028_seed{seed}.csv"
        prior_rows = pd.read_csv(REPORTS / "exp028_seed_fold_metrics.csv") if (REPORTS / "exp028_seed_fold_metrics.csv").exists() else pd.DataFrame()
        if oof_path.exists() and test_path.exists() and not prior_rows.empty and int((prior_rows.seed == seed).sum()) == 5:
            saved_oof = pd.read_csv(oof_path)
            saved_test = pd.read_csv(test_path)
            if saved_oof.id.equals(train.id) and saved_oof.y_true.equals(y_series) and saved_test.id.equals(test.id):
                new_oof[seed] = saved_oof.oof_prediction.to_numpy(np.float64)
                new_test[seed] = saved_test.prediction.to_numpy(np.float64)
                seed_rows = prior_rows.loc[prior_rows.seed.eq(seed)].copy()
                fold_rows.extend(seed_rows.to_dict("records"))
                best_iterations[seed] = seed_rows.sort_values("fold").best_iteration.astype(int).tolist()
                print(f"seed={seed}: reusing completed OOF/test artifacts; no retraining", flush=True)
                continue
        oof = np.zeros(len(train), dtype=np.float64)
        test_sum = np.zeros(len(test), dtype=np.float64)
        best_iterations[seed] = []
        for fold, (tr, va) in enumerate(splits, 1):
            t0 = perf_counter()
            params = dict(MODEL)
            params["random_state"] = seed
            model = XGBClassifier(**params, early_stopping_rounds=EARLY)
            model.fit(X.iloc[tr], y_series.iloc[tr], eval_set=[(X.iloc[va], y_series.iloc[va])], verbose=False)
            best = int(model.best_iteration)
            pv = model.predict_proba(X.iloc[va], iteration_range=(0, best + 1))[:, 1].astype(np.float64)
            pt = model.predict_proba(Xtest, iteration_range=(0, best + 1))[:, 1].astype(np.float64)
            oof[va] = pv
            test_sum += pt / 5.0
            score = float(roc_auc_score(y[va], pv))
            elapsed = perf_counter() - t0
            best_iterations[seed].append(best)
            fold_rows.append({"seed": seed, "fold": fold, "auc": score, "best_iteration": best,
                              "last_iteration": len(model.evals_result()["validation_0"]["auc"]) - 1,
                              "seconds": elapsed})
            pd.DataFrame(fold_rows).to_csv(REPORTS / "exp028_seed_fold_metrics.csv", index=False)
            print(f"seed={seed} fold={fold} auc={score:.8f} best={best} seconds={elapsed:.1f}", flush=True)
        new_oof[seed] = oof
        new_test[seed] = test_sum
        pd.DataFrame({"id": train.id, "y_true": y, "oof_prediction": oof}).to_csv(PRED / f"oof_exp028_seed{seed}.csv", index=False)
        pd.DataFrame({"id": test.id, "prediction": test_sum}).to_csv(PRED / f"test_exp028_seed{seed}.csv", index=False)

    oof_preds = {}
    for seed in [42, 2026, 777]:
        df = pd.read_csv(PRED / f"oof_exp027_seed{seed}.csv")
        if not df.id.equals(train.id) or not df.y_true.equals(y_series) or df.oof_prediction.isna().any():
            raise ValueError(f"OOF existente invalida para seed {seed}")
        oof_preds[seed] = df.oof_prediction.to_numpy(np.float64)
    oof_preds.update(new_oof)

    individual = {s: float(roc_auc_score(y, oof_preds[s])) for s in ALL_SEEDS}
    corr_rows = []
    for a, b in combinations(ALL_SEEDS, 2):
        corr_rows.append({"seed_a": a, "seed_b": b,
                          "pearson": pd.Series(oof_preds[a]).corr(pd.Series(oof_preds[b])),
                          "spearman": pd.Series(oof_preds[a]).corr(pd.Series(oof_preds[b]), method="spearman")})
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(REPORTS / "exp028_seed_correlations.csv", index=False)

    baseline_pred = blend(oof_preds, BASE_WEIGHTS, "probability")
    baseline_auc = float(roc_auc_score(y, baseline_pred))
    candidates = []
    four_defs = [
        ("A_equal", [.25, .25, .25, .25]),
        ("B_35_25_20_20", [.35, .25, .20, .20]),
        ("C_35_25_25_15", [.35, .25, .25, .15]),
        ("D_40_25_20_15", [.40, .25, .20, .15]),
        ("E_proportional", [.35, .275, .275, .10]),
    ]
    for new_seed in NEW_SEEDS:
        order = [42, 2026, 777, new_seed]
        for label, ws in four_defs:
            weights = dict(zip(order, ws))
            for method in ["probability", "rank"]:
                candidates.append(result(f"4seed_{new_seed}_{label}", "specified_4seed", method, weights, oof_preds, y, splits, baseline_auc))

    five_defs = [
        ("A_equal", [.20, .20, .20, .20, .20]),
        ("B", [.30, .20, .20, .15, .15]),
        ("C", [.35, .20, .20, .125, .125]),
        ("D", [.35, .225, .225, .10, .10]),
        ("E", [.40, .20, .20, .10, .10]),
        ("F", [.30, .25, .20, .15, .10]),
    ]
    for label, ws in five_defs:
        weights = dict(zip(ALL_SEEDS, ws))
        for method in ["probability", "rank"]:
            candidates.append(result(f"5seed_{label}", "specified_5seed", method, weights, oof_preds, y, splits, baseline_auc))

    subset_rows = []
    for subset in combinations(ALL_SEEDS, 4):
        weights = {s: .25 for s in subset}
        for method in ["probability", "rank"]:
            subset_rows.append(result("subset_" + "_".join(map(str, subset)), "equal_subset4", method,
                                      weights, oof_preds, y, splits, baseline_auc))
    for subset in [(42, 2026, 777), tuple(ALL_SEEDS)]:
        weights = {s: 1 / len(subset) for s in subset}
        for method in ["probability", "rank"]:
            subset_rows.append(result(f"equal_{len(subset)}", "equal_control", method, weights,
                                      oof_preds, y, splits, baseline_auc))

    marginal_rows = []
    for new_seed in NEW_SEEDS:
        for w in [.025, .05, .075, .10, .15, .20]:
            weights = {s: (1 - w) * bw for s, bw in BASE_WEIGHTS.items()}
            weights[new_seed] = w
            for method in ["probability", "rank"]:
                marginal_rows.append(result(f"marginal_{new_seed}_{w}", "marginal", method, weights,
                                            oof_preds, y, splits, baseline_auc))

    all_seed_results = candidates + subset_rows + marginal_rows
    best4 = max([r for r in all_seed_results if len(r["weight_dict"]) == 4], key=lambda r: r["auc"])
    best5 = max([r for r in all_seed_results if len(r["weight_dict"]) == 5], key=lambda r: r["auc"])
    best_xgb = max([best4, best5], key=lambda r: r["auc"])

    # EXP-027 did not persist individual test predictions for seeds 2026/777. Reconstruct
    # their equal-weight mean exactly from the persisted EXP-027 submission and CatBoost test.
    t42 = pd.read_csv(PRED / "test_exp016_xgboost_depth5_9000.csv").prediction.to_numpy(np.float64)
    t22 = pd.read_csv(PRED / "test_exp022_catboost_thresholds_9000.csv").prediction.to_numpy(np.float64)
    exp027_sub = pd.read_csv(SUB / "submission_exp027_seed_ensemble.csv").addicted_label.to_numpy(np.float64)
    old_bag_test = (exp027_sub - .25 * t22) / .75
    other_old_mean = (old_bag_test - .40 * t42) / .60

    def test_for_weights(weights):
        # Exactly identifiable only when seeds 2026 and 777 have equal weights.
        if abs(weights.get(2026, 0) - weights.get(777, 0)) > 1e-12:
            return None
        out = weights.get(42, 0) * t42 + (weights.get(2026, 0) + weights.get(777, 0)) * other_old_mean
        out += weights.get(31415, 0) * new_test[31415] + weights.get(1234, 0) * new_test[1234]
        return out

    cat_oof_df = pd.read_csv(PRED / "oof_exp022_catboost_thresholds_9000.csv")
    if not cat_oof_df.id.equals(train.id):
        raise ValueError("OOF EXP-022 desalineada")
    p22 = cat_oof_df.oof_prediction.to_numpy(np.float64)
    final_rows = []
    # Per specification, combine CatBoost only with the selected best XGB seed bag.
    # This also keeps memory bounded on the 691k-row OOF set.
    for xr in [best_xgb]:
        for cw in [.15, .20, .225, .25, .275, .30, .325, .35]:
            for method in ["probability", "rank"]:
                xp = xr["prediction"] if method == "probability" else ranks(xr["prediction"])
                cp = p22 if method == "probability" else ranks(p22)
                p = (1 - cw) * xp + cw * cp
                fs = fold_aucs(y, p, splits)
                final_rows.append({"xgb_name": xr["name"], "xgb_method": xr["method"],
                                   "final_method": method, "weights": xr["weights"], "cat_weight": cw,
                                   "auc": float(roc_auc_score(y, p)), "delta_vs_exp027": float(roc_auc_score(y, p) - EXP027),
                                   "fold1": fs[0], "fold2": fs[1], "fold3": fs[2], "fold4": fs[3], "fold5": fs[4],
                                   "std": float(np.std(fs)), "folds_improved": int(np.sum(np.asarray(fs) > EXP027_FOLDS)),
                                   "large_drop": bool(np.any(np.asarray(fs) - EXP027_FOLDS < -0.00003)),
                                   "test_available": test_for_weights(xr["weight_dict"]) is not None,
                                   "prediction": p, "weight_dict": xr["weight_dict"]})
    best_final_oof = max(final_rows, key=lambda r: r["auc"])
    eligible = [r for r in final_rows if r["delta_vs_exp027"] >= .00002 and r["folds_improved"] >= 4
                and not r["large_drop"] and r["test_available"] and r["xgb_method"] == "probability"]
    selected_final = max(eligible, key=lambda r: r["auc"]) if eligible else None

    submission_generated = False
    submission_path = "none"
    if selected_final is not None:
        weights = selected_final["weight_dict"]
        xt = test_for_weights(weights)
        if selected_final["xgb_method"] == "rank":
            # The XGB candidate's OOF ranking method must also be applied on test.
            xt = ranks(xt)
        ct = t22 if selected_final["final_method"] == "probability" else ranks(t22)
        if selected_final["final_method"] == "rank":
            xt = ranks(xt)
        final_test = (1 - selected_final["cat_weight"]) * xt + selected_final["cat_weight"] * ct
        sub = pd.DataFrame({"id": sample.id, "addicted_label": final_test})
        if len(sub) != len(test) or len(sub) != 296302 or sub.columns.tolist() != ["id", "addicted_label"]:
            raise ValueError("Shape/columnas de submission invalidas")
        if not sub.id.equals(sample.id) or sub.isna().any().any() or not sub.addicted_label.between(0, 1).all():
            raise ValueError("Contenido de submission invalido")
        sub_path = SUB / "submission_exp028_seed5_ensemble.csv"
        sub.to_csv(sub_path, index=False)
        submission_generated = True
        submission_path = str(sub_path)
        update_log(selected_final["auc"], selected_final["final_method"], weights, selected_final["cat_weight"])

    def clean(rows):
        return pd.DataFrame([{k: v for k, v in r.items() if k not in {"prediction", "weight_dict"}} for r in rows])

    clean(marginal_rows).to_csv(REPORTS / "exp028_seed_marginal_gain.csv", index=False)
    clean(subset_rows).to_csv(REPORTS / "exp028_seed_subset_results.csv", index=False)
    clean(final_rows).to_csv(REPORTS / "exp028_final_blends.csv", index=False)

    gain34 = best4["auc"] - baseline_auc
    gain45 = best5["auc"] - best4["auc"]
    ratio = gain45 / gain34 if gain34 > 0 else np.nan
    if gain45 < 0:
        diminishing = "negative: stop seed bagging"
    elif ratio >= .5:
        diminishing = "bagging still has room"
    elif ratio >= .2:
        diminishing = "diminishing but useful"
    else:
        diminishing = "nearly saturated"

    elapsed = perf_counter() - start_all
    lines = [
        "EXP-028 seed extension; exact EXP-016 pipeline",
        f"model={MODEL}; early_stopping={EARLY}; features={NEW_FEATURES}; mappings={mappings}",
        f"individual_auc={individual}",
        f"best_iterations={best_iterations}",
        "correlations:\n" + corr_df.to_string(index=False),
        f"baseline_3seed_auc={baseline_auc:.12f}",
        f"best4={{{', '.join(f'{k}={v}' for k,v in best4.items() if k not in {'prediction','weight_dict'})}}}",
        f"best5={{{', '.join(f'{k}={v}' for k,v in best5.items() if k not in {'prediction','weight_dict'})}}}",
        f"best_xgb={{{', '.join(f'{k}={v}' for k,v in best_xgb.items() if k not in {'prediction','weight_dict'})}}}",
        f"gain_3_to_4={gain34:.12f}; gain_4_to_5={gain45:.12f}; ratio={ratio}; interpretation={diminishing}",
        f"best_final_oof={{{', '.join(f'{k}={v}' for k,v in best_final_oof.items() if k not in {'prediction','weight_dict'})}}}",
        f"selected_final={None if selected_final is None else {k:v for k,v in selected_final.items() if k not in {'prediction','weight_dict'}}}",
        f"submission_generated={submission_generated}; path={submission_path}",
        "test_artifact_note=individual test predictions for seeds 2026/777 were not persisted by EXP-027; exact reconstruction is available when their weights are equal",
        f"total_seconds={elapsed:.2f}",
        "problems=none during training/validation; historical EXP-027 test-artifact limitation documented",
    ]
    (METRICS / "exp028_seed5_metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"individual={individual}")
    print(f"baseline={baseline_auc:.12f}; best4={best4['auc']:.12f}; best5={best5['auc']:.12f}")
    print(f"best_xgb={best_xgb['name']} {best_xgb['method']} {best_xgb['weights']} auc={best_xgb['auc']:.12f}")
    print(f"best_final={best_final_oof['auc']:.12f}; selected={None if selected_final is None else selected_final['auc']}")
    print(f"submission={submission_generated}; path={submission_path}; seconds={elapsed:.2f}")


if __name__ == "__main__":
    main()
