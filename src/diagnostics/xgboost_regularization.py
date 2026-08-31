"""Depth-5 XGBoost regularization diagnostic using EXP-016 features only."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.project_paths import PROJECT_ROOT
from src.train_xgboost_exp008 import MODEL_PARAMS, ordinal_encode_categories
from src.train_xgboost_exp012_threshold_features import NEW_FEATURES, add_threshold_features


ROOT = PROJECT_ROOT
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
REPORT_PATH = OUTPUTS / "reports" / "depth5_regularization_diagnostic.csv"
METRICS_PATH = OUTPUTS / "metrics" / "depth5_regularization_diagnostic.txt"
EXP016_OOF_PATH = OUTPUTS / "predictions" / "oof_exp016_xgboost_depth5_9000.csv"
ID = "id"
TARGET = "addicted_label"
EARLY_STOPPING = 300
BASE_PARAMS = dict(MODEL_PARAMS)
BASE_PARAMS.update({"max_depth": 5, "n_estimators": 9000})
REFERENCE_FOLDS = np.array([0.96502029, 0.96566446, 0.96588369, 0.96645502, 0.96548740])
CONFIGS = [
    ("R1", 3, 1), ("R2", 5, 1), ("R3", 10, 1),
    ("R4", 1, 3), ("R5", 1, 5), ("R6", 1, 10),
    ("R7", 3, 5), ("R8", 5, 5), ("R9", 5, 10), ("R10", 10, 5),
]
CSV_COLUMNS = [
    "config_id", "min_child_weight", "reg_lambda",
    "fold1_auc", "fold2_auc", "fold3_auc", "mean_auc", "std_auc",
    "delta_vs_baseline", "best_iter_fold1", "best_iter_fold2", "best_iter_fold3",
    "training_seconds",
]


def load_features() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    originals = [column for column in train.columns if column not in {ID, TARGET}]
    if test.columns.tolist() != [ID, *originals]:
        raise ValueError("Esquema train/test original no coincide.")
    raw = add_threshold_features(train[originals])
    raw_test = add_threshold_features(test[originals])
    if raw.columns.tolist() != raw_test.columns.tolist():
        raise ValueError("Esquema de features EXP-016 no coincide.")
    if len(raw.columns) != len(originals) + len(NEW_FEATURES):
        raise ValueError("Se agregaron o quitaron features respecto a EXP-016.")
    categoricals = [c for c in originals if not pd.api.types.is_numeric_dtype(raw[c])]
    numeric = [c for c in raw if c not in categoricals]
    X, X_test, _ = ordinal_encode_categories(raw, raw_test, categoricals)
    if not X[numeric].equals(raw[numeric]) or not X_test[numeric].equals(raw_test[numeric]):
        raise ValueError("El preprocessing modifico variables numericas.")
    return X, train[TARGET], train[ID]


def train_fold(X: pd.DataFrame, y: pd.Series,
               split: tuple[np.ndarray, np.ndarray], params: dict[str, object]) -> dict[str, object]:
    train_idx, valid_idx = split
    start = perf_counter()
    model = XGBClassifier(**params, early_stopping_rounds=EARLY_STOPPING)
    model.fit(
        X.iloc[train_idx], y.iloc[train_idx],
        eval_set=[(X.iloc[valid_idx], y.iloc[valid_idx])], verbose=False,
    )
    best = int(model.best_iteration)
    prediction = model.predict_proba(X.iloc[valid_idx], iteration_range=(0, best + 1))[:, 1]
    return {
        "auc": float(roc_auc_score(y.iloc[valid_idx], prediction)),
        "best_iteration": best, "seconds": perf_counter() - start,
    }


def classify(delta: float) -> str:
    if delta >= .00008:
        return "strong"
    if delta >= .00003:
        return "marginal"
    if delta >= 0:
        return "tie"
    return "discard"


def save_screening(rows: list[dict[str, object]]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(REPORT_PATH, index=False)


def main() -> None:
    total_start = perf_counter()
    X, y, ids = load_features()
    splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, y))
    exp016 = pd.read_csv(EXP016_OOF_PATH)
    if not exp016[ID].equals(ids) or not exp016["y_true"].equals(y):
        raise ValueError("OOF EXP-016 no coincide con train.")
    exact_baseline = np.array([
        float(roc_auc_score(y.iloc[valid], exp016["oof_prediction"].iloc[valid]))
        for _, valid in splits
    ])
    if not np.allclose(exact_baseline, REFERENCE_FOLDS, atol=5e-9):
        raise ValueError(f"Referencia EXP-016 inesperada: {exact_baseline}")
    baseline3 = float(exact_baseline[:3].mean())

    print(f"Baseline exact folds={exact_baseline.tolist()} mean3={baseline3:.10f}", flush=True)
    print(f"Features={X.shape[1]}; configurations={len(CONFIGS)}", flush=True)
    rows: list[dict[str, object]] = []
    config_fold_details: dict[str, list[dict[str, object]]] = {}

    for config_id, child_weight, reg_lambda in CONFIGS:
        params = dict(BASE_PARAMS)
        params.update({"min_child_weight": child_weight, "reg_lambda": reg_lambda})
        results: list[dict[str, object]] = []
        for fold_index in range(3):
            result = train_fold(X, y, splits[fold_index], params)
            results.append(result)
            print(
                f"{config_id} Fold {fold_index+1}: AUC={result['auc']:.8f} "
                f"best={result['best_iteration']} time={result['seconds']:.1f}s",
                flush=True,
            )
        scores = np.array([float(result["auc"]) for result in results])
        row = {
            "config_id": config_id, "min_child_weight": child_weight,
            "reg_lambda": reg_lambda,
            "fold1_auc": scores[0], "fold2_auc": scores[1], "fold3_auc": scores[2],
            "mean_auc": float(scores.mean()), "std_auc": float(scores.std()),
            "delta_vs_baseline": float(scores.mean() - baseline3),
            "best_iter_fold1": int(results[0]["best_iteration"]),
            "best_iter_fold2": int(results[1]["best_iteration"]),
            "best_iter_fold3": int(results[2]["best_iteration"]),
            "training_seconds": float(sum(float(result["seconds"]) for result in results)),
        }
        rows.append(row)
        config_fold_details[config_id] = results
        save_screening(rows)
        print(
            f"{config_id}: mean={row['mean_auc']:.8f} delta={row['delta_vs_baseline']:+.8f} "
            f"class={classify(float(row['delta_vs_baseline']))}", flush=True,
        )

    screening = pd.DataFrame(rows, columns=CSV_COLUMNS)
    screening["classification"] = screening["delta_vs_baseline"].map(classify)
    screening["improved_first3"] = (
        screening[["fold1_auc", "fold2_auc", "fold3_auc"]].to_numpy()
        > exact_baseline[:3]
    ).sum(axis=1)
    eligible = screening.loc[screening["classification"].isin(["strong", "marginal"])].copy()
    eligible = eligible.sort_values(
        ["improved_first3", "mean_auc", "std_auc"], ascending=[False, False, True]
    )
    selected_ids = eligible.head(2)["config_id"].tolist()

    five_fold_results: list[dict[str, object]] = []
    config_lookup = {config_id: (child, lam) for config_id, child, lam in CONFIGS}
    for config_id in selected_ids:
        child_weight, reg_lambda = config_lookup[config_id]
        params = dict(BASE_PARAMS)
        params.update({"min_child_weight": child_weight, "reg_lambda": reg_lambda})
        screen_row = screening.loc[screening["config_id"] == config_id].iloc[0]
        scores = [float(screen_row[f"fold{i}_auc"]) for i in range(1, 4)]
        best_iterations = [int(screen_row[f"best_iter_fold{i}"]) for i in range(1, 4)]
        seconds = float(screen_row["training_seconds"])
        for fold_index in (3, 4):
            result = train_fold(X, y, splits[fold_index], params)
            scores.append(float(result["auc"]))
            best_iterations.append(int(result["best_iteration"]))
            seconds += float(result["seconds"])
            print(
                f"FINALIST {config_id} Fold {fold_index+1}: AUC={result['auc']:.8f} "
                f"best={result['best_iteration']} time={result['seconds']:.1f}s", flush=True,
            )
        score_array = np.asarray(scores)
        five_fold_results.append({
            "config_id": config_id, "min_child_weight": child_weight,
            "reg_lambda": reg_lambda, "fold_aucs": scores,
            "mean_auc": float(score_array.mean()), "std_auc": float(score_array.std()),
            "delta_vs_exp016_mean": float(score_array.mean() - exact_baseline.mean()),
            "fold_deltas": (score_array - exact_baseline).tolist(),
            "improved_folds": int((score_array > exact_baseline).sum()),
            "best_iterations": best_iterations, "training_seconds": seconds,
        })

    strong = screening.loc[screening["classification"] == "strong", "config_id"].tolist()
    marginal = screening.loc[screening["classification"] == "marginal", "config_id"].tolist()
    if five_fold_results:
        ranked_finalists = sorted(
            five_fold_results,
            key=lambda item: (item["delta_vs_exp016_mean"], item["improved_folds"], -item["std_auc"]),
            reverse=True,
        )
        best = ranked_finalists[0]
        if best["delta_vs_exp016_mean"] >= .00003 and best["improved_folds"] >= 4:
            recommendation = (
                f"Use {best['config_id']} for EXP-019: min_child_weight={best['min_child_weight']}, "
                f"reg_lambda={best['reg_lambda']}."
            )
        elif best["delta_vs_exp016_mean"] > 0:
            recommendation = f"{best['config_id']} is only marginal/inconsistent; EXP-019 not justified yet."
        else:
            recommendation = "No regularization candidate justifies EXP-019."
    else:
        recommendation = "No screening candidate reached marginal threshold; EXP-019 not justified."

    total_seconds = perf_counter() - total_start
    best_iteration_shift = {
        result["config_id"]: {
            "mean_best_iteration": float(np.mean(result["best_iterations"])),
            "exp016_mean_best_iteration": 7791.8,
            "difference": float(np.mean(result["best_iterations"]) - 7791.8),
        }
        for result in five_fold_results
    }
    ordered = screening.sort_values("mean_auc", ascending=False)
    report = [
        "Depth-5 XGBoost regularization diagnostic (EXP-016 features only)",
        f"baseline_exact_folds: {exact_baseline.tolist()}",
        f"baseline_3fold_mean: {baseline3:.10f}",
        "screening_ordered:", ordered.to_string(index=False),
        f"strong_candidates: {strong}", f"marginal_candidates: {marginal}",
        f"selected_for_5fold: {selected_ids}",
        "five_fold_results:", json.dumps(five_fold_results, indent=2),
        f"best_iteration_shift: {json.dumps(best_iteration_shift, indent=2)}",
        f"recommendation: {recommendation}", f"total_seconds: {total_seconds:.2f}",
        "submission_generated: false", "experiment_log_modified: false", "problems: none",
    ]
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\nSCREENING ORDERED\n" + ordered.to_string(index=False), flush=True)
    print(f"Strong={strong}; marginal={marginal}; selected={selected_ids}", flush=True)
    print("Five-fold=" + json.dumps(five_fold_results, indent=2), flush=True)
    print(f"Best iteration shift={best_iteration_shift}", flush=True)
    print(f"Recommendation={recommendation}", flush=True)
    print(f"Total={total_seconds:.2f}s", flush=True)


if __name__ == "__main__":
    main()
