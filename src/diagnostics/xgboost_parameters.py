"""Diagnostico economico de hiperparametros para EXP-012 (sin submission)."""

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
REPORTS = ROOT / "outputs" / "reports"
METRICS = ROOT / "outputs" / "metrics"
CSV_PATH = REPORTS / "xgboost_hyperparam_diagnostic.csv"
TXT_PATH = METRICS / "xgboost_hyperparam_diagnostic.txt"
EXP012_OOF = ROOT / "outputs" / "predictions" / "oof_exp012_xgboost_thresholds.csv"

ID_COLUMN = "id"
TARGET = "addicted_label"
EARLY_STOPPING_ROUNDS = 200
RESULT_COLUMNS = [
    "config_id", "max_depth", "min_child_weight", "learning_rate",
    "n_estimators", "subsample", "colsample_bytree", "reg_alpha",
    "reg_lambda", "gamma", "fold1_auc", "delta_vs_baseline",
    "best_iteration", "training_seconds",
]
BASELINE_FOLDS = [0.96460143, 0.96521723, 0.96534392]


def configurations() -> list[tuple[str, dict[str, float | int]]]:
    return [
        ("BASELINE", {}),
        ("A1_depth5", {"max_depth": 5}),
        ("A2_depth6", {"max_depth": 6}),
        ("A3_depth8", {"max_depth": 8}),
        ("A4_depth9", {"max_depth": 9}),
        ("B1_child3", {"min_child_weight": 3}),
        ("B2_child5", {"min_child_weight": 5}),
        ("B3_child10", {"min_child_weight": 10}),
        ("C1_lambda2", {"reg_lambda": 2}),
        ("C2_lambda5", {"reg_lambda": 5}),
        ("C3_lambda10", {"reg_lambda": 10}),
        ("D1_alpha0.1", {"reg_alpha": 0.1}),
        ("D2_alpha0.5", {"reg_alpha": 0.5}),
        ("D3_alpha1.0", {"reg_alpha": 1.0}),
        ("E1_gamma0.05", {"gamma": 0.05}),
        ("E2_gamma0.1", {"gamma": 0.1}),
        ("E3_gamma0.25", {"gamma": 0.25}),
        ("F1_sub0.8_col0.9", {"subsample": 0.8, "colsample_bytree": 0.9}),
        ("F2_sub1.0_col0.9", {"subsample": 1.0, "colsample_bytree": 0.9}),
        ("F3_sub0.9_col0.8", {"subsample": 0.9, "colsample_bytree": 0.8}),
        ("F4_sub0.9_col1.0", {"subsample": 0.9, "colsample_bytree": 1.0}),
        ("F5_sub0.8_col0.8", {"subsample": 0.8, "colsample_bytree": 0.8}),
        ("G1_lr0.02", {"learning_rate": 0.02, "n_estimators": 8000}),
        ("G2_lr0.04", {"learning_rate": 0.04, "n_estimators": 5000}),
        ("G3_lr0.05", {"learning_rate": 0.05, "n_estimators": 4000}),
    ]


def full_params(changes: dict[str, float | int]) -> dict[str, object]:
    params = dict(MODEL_PARAMS)
    params.update(changes)
    return params


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    original = [c for c in train.columns if c not in {ID_COLUMN, TARGET}]
    if test.columns.tolist() != [ID_COLUMN, *original]:
        raise ValueError("El esquema original train/test no coincide.")
    raw_train = add_threshold_features(train[original])
    raw_test = add_threshold_features(test[original])
    if raw_train.columns.tolist() != raw_test.columns.tolist():
        raise ValueError("El esquema EXP-012 train/test no coincide.")
    if not all(c in raw_train for c in NEW_FEATURES):
        raise ValueError("Faltan threshold features de EXP-012.")
    categoricals = [c for c in original if not pd.api.types.is_numeric_dtype(raw_train[c])]
    encoded, encoded_test, _ = ordinal_encode_categories(raw_train, raw_test, categoricals)
    numeric = [c for c in raw_train if c not in categoricals]
    if not encoded[numeric].equals(raw_train[numeric]):
        raise ValueError("Las features numericas fueron modificadas.")
    if not encoded_test[numeric].equals(raw_test[numeric]):
        raise ValueError("Las features numericas de test fueron modificadas.")
    return encoded, train[TARGET], train[ID_COLUMN]


def train_fold(
    X: pd.DataFrame, y: pd.Series, split: tuple[np.ndarray, np.ndarray],
    params: dict[str, object],
) -> tuple[float, int, float, bool]:
    train_idx, valid_idx = split
    start = perf_counter()
    model = XGBClassifier(**params, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    model.fit(
        X.iloc[train_idx], y.iloc[train_idx],
        eval_set=[(X.iloc[valid_idx], y.iloc[valid_idx])], verbose=False,
    )
    best_iteration = int(model.best_iteration)
    pred = model.predict_proba(
        X.iloc[valid_idx], iteration_range=(0, best_iteration + 1)
    )[:, 1]
    auc = float(roc_auc_score(y.iloc[valid_idx], pred))
    elapsed = perf_counter() - start
    # Without retaining the booster, only mark this as an effective early stop
    # when the full patience window fits strictly before the estimator cap.
    stopped = (
        best_iteration + EARLY_STOPPING_ROUNDS + 1
        < int(params["n_estimators"])
    )
    return auc, best_iteration, elapsed, stopped


def result_row(config_id: str, params: dict[str, object], auc: float,
               baseline: float, best: int, seconds: float) -> dict[str, object]:
    return {
        "config_id": config_id,
        **{key: params[key] for key in RESULT_COLUMNS[1:10]},
        "fold1_auc": auc,
        "delta_vs_baseline": auc - baseline,
        "best_iteration": best,
        "training_seconds": seconds,
    }


def save_rows(rows: list[dict[str, object]]) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(CSV_PATH, index=False)


def select_candidates(screen: pd.DataFrame) -> list[str]:
    alternatives = screen.loc[screen["config_id"] != "BASELINE"].sort_values(
        ["fold1_auc", "best_iteration"], ascending=[False, True]
    )
    preferred = alternatives.loc[alternatives["delta_vs_baseline"] >= 0.00010]
    if preferred.empty:
        preferred = alternatives.loc[alternatives["delta_vs_baseline"] >= 0.00003]
    if preferred.empty:
        preferred = alternatives.head(1)
    return preferred.head(3)["config_id"].tolist()


def main() -> None:
    total_start = perf_counter()
    X, y, ids = load_data()
    splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, y))

    reference = pd.read_csv(EXP012_OOF)
    if not reference[ID_COLUMN].equals(ids) or not reference["y_true"].equals(y):
        raise ValueError("OOF de EXP-012 no coincide con train.")
    exact_baseline_folds = [
        float(roc_auc_score(y.iloc[v], reference["oof_prediction"].iloc[v]))
        for _, v in splits[:3]
    ]
    if not np.allclose(exact_baseline_folds, BASELINE_FOLDS, atol=5e-9):
        raise ValueError(f"Baseline OOF inesperado: {exact_baseline_folds}")

    configs = configurations()
    by_id = {name: changes for name, changes in configs}
    rows: list[dict[str, object]] = []
    baseline_auc: float | None = None
    print(f"Filas={len(X)} | features={X.shape[1]} | screening={len(configs)} configs", flush=True)

    for config_id, changes in configs:
        params = full_params(changes)
        auc, best, seconds, stopped = train_fold(X, y, splits[0], params)
        if config_id == "BASELINE":
            baseline_auc = auc
            if abs(auc - exact_baseline_folds[0]) > 1e-8:
                raise ValueError(
                    f"Baseline no reproducible: nuevo={auc:.10f}, EXP-012={exact_baseline_folds[0]:.10f}"
                )
        assert baseline_auc is not None
        rows.append(result_row(config_id, params, auc, baseline_auc, best, seconds))
        save_rows(rows)
        print(
            f"{config_id}: AUC={auc:.8f} delta={auc-baseline_auc:+.8f} "
            f"best={best} stopped={stopped} time={seconds:.1f}s",
            flush=True,
        )

    screen = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    candidates = select_candidates(screen)
    candidate_results: list[dict[str, object]] = []
    for config_id in candidates:
        params = full_params(by_id[config_id])
        fold1 = screen.loc[screen["config_id"] == config_id].iloc[0]
        scores = [float(fold1["fold1_auc"])]
        best_iterations = [int(fold1["best_iteration"])]
        times = [float(fold1["training_seconds"])]
        stopped_flags = [
            best_iterations[0] + EARLY_STOPPING_ROUNDS + 1
            < int(params["n_estimators"])
        ]
        for fold_index in (1, 2):
            auc, best, seconds, stopped = train_fold(X, y, splits[fold_index], params)
            scores.append(auc)
            best_iterations.append(best)
            times.append(seconds)
            stopped_flags.append(stopped)
            print(
                f"3-fold {config_id} Fold {fold_index+1}: AUC={auc:.8f} "
                f"best={best} time={seconds:.1f}s",
                flush=True,
            )
        candidate_results.append({
            "config_id": config_id,
            "modified_params": by_id[config_id],
            "fold_aucs": scores,
            "mean_auc": float(np.mean(scores)),
            "std_auc": float(np.std(scores)),
            "delta_vs_baseline_3fold": float(np.mean(scores) - np.mean(exact_baseline_folds)),
            "best_iterations": best_iterations,
            "training_seconds": times,
            "early_stopping_activated": stopped_flags,
        })

    top10 = screen.sort_values("fold1_auc", ascending=False).head(10)
    block_prefixes = {letter: f"{letter}" for letter in "ABCDEFG"}
    best_by_block: dict[str, dict[str, object]] = {}
    for block, prefix in block_prefixes.items():
        subset = screen.loc[screen["config_id"].str.startswith(prefix)]
        best_by_block[block] = subset.sort_values("fold1_auc", ascending=False).iloc[0].to_dict()

    ranked_candidates = sorted(candidate_results, key=lambda x: x["mean_auc"], reverse=True)
    strong = [c for c in ranked_candidates if c["delta_vs_baseline_3fold"] >= 0.00008]
    marginal = [
        c for c in ranked_candidates
        if 0.00003 <= c["delta_vs_baseline_3fold"] < 0.00008
    ]
    if strong:
        recommendation = [c["config_id"] for c in strong[:2]]
        recommendation_text = f"Candidato(s) fuerte(s) para EXP-015: {recommendation}"
    elif marginal:
        recommendation = [c["config_id"] for c in marginal[:1]]
        recommendation_text = f"Candidato marginal para EXP-015: {recommendation}"
    else:
        recommendation = []
        recommendation_text = "Ninguna configuracion justifica un EXP-015 de 5 folds."

    total_seconds = perf_counter() - total_start
    report = [
        "XGBoost hyperparameter diagnostic (EXP-012 features; Fold 1 screening)",
        f"baseline_reproduced_fold1: {baseline_auc:.10f}",
        f"baseline_reference_fold1: {exact_baseline_folds[0]:.10f}",
        f"baseline_3fold_scores: {exact_baseline_folds}",
        f"baseline_3fold_mean: {np.mean(exact_baseline_folds):.10f}",
        "top10_screening:", top10.to_string(index=False),
        f"best_by_block: {json.dumps(best_by_block, ensure_ascii=False, default=float)}",
        f"selected_candidates: {candidates}",
        "candidate_3fold_results:",
        json.dumps(candidate_results, ensure_ascii=False, indent=2),
        f"recommendation: {recommendation_text}",
        f"total_seconds: {total_seconds:.2f}",
        "problems: none",
    ]
    METRICS.mkdir(parents=True, exist_ok=True)
    TXT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\nTOP 10\n" + top10.to_string(index=False), flush=True)
    print(f"Candidates: {candidates}", flush=True)
    print(json.dumps(candidate_results, ensure_ascii=False, indent=2), flush=True)
    print(recommendation_text, flush=True)
    print(f"Tiempo total: {total_seconds:.2f}s", flush=True)


if __name__ == "__main__":
    main()
