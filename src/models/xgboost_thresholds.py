"""EXP-016: EXP-015 with only 9000 estimators and early stopping patience 300."""

from __future__ import annotations

from datetime import datetime
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.features.thresholds import THRESHOLD_FEATURES as NEW_FEATURES, add_threshold_features
from src.models.xgboost_baseline import (
    MODEL_PARAMS,
    ordinal_encode_categories,
    validate_submission,
)
from src.project_paths import (
    DATA_DIR as DATA,
    METRICS_DIR as METRICS,
    OUTPUTS_DIR as OUTPUTS,
    PREDICTIONS_DIR as PREDICTIONS,
    PROJECT_ROOT as ROOT,
    REPORTS_DIR as REPORTS,
    SUBMISSIONS_DIR as SUBMISSIONS,
)


ID = "id"
TARGET = "addicted_label"
EXPERIMENT = "EXP-016"
OOF_PATH = PREDICTIONS / "oof_exp016_xgboost_depth5_9000.csv"
TEST_PATH = PREDICTIONS / "test_exp016_xgboost_depth5_9000.csv"
SUBMISSION_PATH = SUBMISSIONS / "submission_exp016_xgboost_depth5_9000.csv"
METRICS_PATH = METRICS / "exp016_xgboost_depth5_9000_metrics.txt"
CURVE_PATH = REPORTS / "exp016_learning_curve.csv"
LOG_PATH = METRICS / "experiment_log.csv"

REFERENCE_PATHS = {
    "EXP-015": PREDICTIONS / "oof_exp015_xgboost_depth5.csv",
    "EXP-012": PREDICTIONS / "oof_exp012_xgboost_thresholds.csv",
    "EXP-003": PREDICTIONS / "oof_exp003_catboost.csv",
    "EXP-006": PREDICTIONS / "oof_exp006_lightgbm_features.csv",
}
EXP015_FOLDS = np.array([0.96493421, 0.96557286, 0.96578517, 0.96633031, 0.96540304])
MODEL = dict(MODEL_PARAMS)
MODEL.update({"max_depth": 5, "n_estimators": 9000})
EARLY_STOPPING = 300


def update_log(mean_auc: float) -> None:
    columns = [
        "experiment_id", "datetime", "model", "features", "cv_strategy",
        "cv_roc_auc", "kaggle_score", "notes",
    ]
    log = pd.read_csv(LOG_PATH, dtype=str, keep_default_na=False)
    if log.columns.tolist() != columns or (log["experiment_id"] == "EXP-015").sum() != 1:
        raise ValueError("Estado inesperado de experiment_log.csv.")
    log.loc[log["experiment_id"] == "EXP-015", "kaggle_score"] = "0.96710"
    log = log.loc[log["experiment_id"] != EXPERIMENT].copy()
    row = pd.DataFrame([{
        "experiment_id": EXPERIMENT,
        "datetime": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": "XGBoost",
        "features": "exp012_threshold_region_features",
        "cv_strategy": "StratifiedKFold_5",
        "cv_roc_auc": f"{mean_auc:.6f}",
        "kaggle_score": "",
        "notes": (
            "EXP-015 max_depth=5 with n_estimators increased from 6000 to 9000 "
            "and early stopping 300"
        ),
    }])
    pd.concat([log, row], ignore_index=True).to_csv(LOG_PATH, index=False)


def calculate_correlations(current: pd.DataFrame) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name, path in REFERENCE_PATHS.items():
        ref = pd.read_csv(path)
        if ref.columns.tolist() != [ID, "y_true", "oof_prediction"]:
            raise ValueError(f"Esquema OOF inesperado para {name}.")
        aligned = current.merge(
            ref.rename(columns={"y_true": "ref_y", "oof_prediction": "ref_prediction"}),
            on=ID, how="inner", validate="one_to_one", sort=False,
        )
        if len(aligned) != len(current) or not aligned["y_true"].equals(aligned["ref_y"]):
            raise ValueError(f"IDs o target no coinciden con {name}.")
        result[name] = {
            "pearson": float(aligned["oof_prediction"].corr(aligned["ref_prediction"], method="pearson")),
            "spearman": float(aligned["oof_prediction"].corr(aligned["ref_prediction"], method="spearman")),
        }
    return result


def main() -> None:
    total_start = perf_counter()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    originals = [c for c in train.columns if c not in {ID, TARGET}]
    if test.columns.tolist() != [ID, *originals]:
        raise ValueError("Train/test original no coincide.")
    raw = add_threshold_features(train[originals])
    raw_test = add_threshold_features(test[originals])
    if raw.columns.tolist() != raw_test.columns.tolist():
        raise ValueError("Train/test final no coincide.")
    if not all(c in raw and c in raw_test for c in NEW_FEATURES):
        raise ValueError("Faltan features exactas de EXP-012/015.")
    if len(raw.columns) != len(originals) + len(NEW_FEATURES):
        raise ValueError("Se agregaron o quitaron features.")
    categoricals = [c for c in originals if not pd.api.types.is_numeric_dtype(raw[c])]
    numeric = [c for c in raw if c not in categoricals]
    X, X_test, mappings = ordinal_encode_categories(raw, raw_test, categoricals)
    if not X[numeric].equals(raw[numeric]) or not X_test[numeric].equals(raw_test[numeric]):
        raise ValueError("El preprocessing modifico variables numericas.")
    y = train[TARGET]
    splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, y))

    print(f"{EXPERIMENT}: {len(X)} filas, {X.shape[1]} features", flush=True)
    print(f"Parametros: {MODEL}; early_stopping_rounds={EARLY_STOPPING}", flush=True)
    print(f"Mappings: {mappings}", flush=True)

    oof = np.zeros(len(train), dtype=np.float64)
    test_prediction = np.zeros(len(test), dtype=np.float64)
    scores: list[float] = []
    best_iterations: list[int] = []
    last_iterations: list[int] = []
    fold_seconds: list[float] = []
    gains: list[dict[str, float]] = []
    curve_rows: list[dict[str, float | int]] = []

    for fold, (train_idx, valid_idx) in enumerate(splits, 1):
        start = perf_counter()
        model = XGBClassifier(**MODEL, early_stopping_rounds=EARLY_STOPPING)
        model.fit(
            X.iloc[train_idx], y.iloc[train_idx],
            eval_set=[(X.iloc[valid_idx], y.iloc[valid_idx])], verbose=False,
        )
        auc_curve = model.evals_result()["validation_0"]["auc"]
        best = int(model.best_iteration)
        last = len(auc_curve) - 1
        iteration_range = (0, best + 1)
        valid_prediction = model.predict_proba(
            X.iloc[valid_idx], iteration_range=iteration_range
        )[:, 1].astype(np.float64)
        score = float(roc_auc_score(y.iloc[valid_idx], valid_prediction))
        oof[valid_idx] = valid_prediction
        test_prediction += model.predict_proba(
            X_test, iteration_range=iteration_range
        )[:, 1].astype(np.float64) / 5.0
        raw_gain = model.get_booster().get_score(importance_type="gain")
        total_gain = float(sum(raw_gain.values()))
        gains.append({c: raw_gain.get(c, 0.0) / total_gain for c in X.columns})

        checkpoints = set(range(499, len(auc_curve), 500))
        checkpoints.update({best, last})
        for zero_index in sorted(checkpoints):
            curve_rows.append({
                "fold": fold,
                "iteration": zero_index + 1,
                "validation_auc": float(auc_curve[zero_index]),
            })
        elapsed = perf_counter() - start
        scores.append(score)
        best_iterations.append(best)
        last_iterations.append(last)
        fold_seconds.append(elapsed)
        print(
            f"Fold {fold}: AUC={score:.8f} best={best} last={last} "
            f"distance_to_limit={8999-best} time={elapsed:.2f}s",
            flush=True,
        )

    total_seconds = perf_counter() - total_start
    mean_auc = float(np.mean(scores))
    std_auc = float(np.std(scores))
    overall_auc = float(roc_auc_score(y, oof))
    exp015 = pd.read_csv(REFERENCE_PATHS["EXP-015"])
    exp015_global = float(roc_auc_score(exp015["y_true"], exp015["oof_prediction"]))
    global_delta = overall_auc - exp015_global
    fold_deltas = np.asarray(scores) - EXP015_FOLDS
    exp015_scores_exact = [
        float(roc_auc_score(exp015["y_true"].iloc[v], exp015["oof_prediction"].iloc[v]))
        for _, v in splits
    ]
    std_change = std_auc - float(np.std(exp015_scores_exact))

    test_prediction = np.clip(test_prediction, 0.0, 1.0)
    oof_frame = pd.DataFrame({ID: train[ID], "y_true": y, "oof_prediction": oof})
    test_frame = pd.DataFrame({ID: sample[ID], "prediction": test_prediction})
    submission = test_frame.rename(columns={"prediction": TARGET})
    if oof_frame.isna().any().any() or not oof_frame["oof_prediction"].between(0, 1).all():
        raise ValueError("OOF invalida.")
    validate_submission(submission, test, sample)

    corr = calculate_correlations(oof_frame)
    top20 = pd.DataFrame(gains).mean().sort_values(ascending=False).head(20)
    distribution = {
        "lt_6000": sum(v < 6000 for v in best_iterations),
        "6000_to_7000": sum(6000 <= v <= 7000 for v in best_iterations),
        "7000_to_8000": sum(7000 < v <= 8000 for v in best_iterations),
        "8000_to_8500": sum(8000 < v <= 8500 for v in best_iterations),
        "gt_8500": sum(v > 8500 for v in best_iterations),
    }
    early_stopping_real = [last < 8999 for last in last_iterations]
    improved_folds = int(sum(delta > 0 for delta in fold_deltas))
    if global_delta >= 0.00010:
        decision = "Mejora >= +0.00010 OOF: recomendar subir a Kaggle."
    elif global_delta >= 0.00003 and improved_folds >= 4:
        decision = "Mejora marginal y al menos 4 folds: recomendar submission."
    elif global_delta >= 0:
        decision = "Empate practico."
    else:
        decision = "Empeora: 6000 arboles eran suficientes."
    if distribution["gt_8500"] >= 4:
        tree_recommendation = "Podria justificarse un diagnostico posterior con mas arboles."
    elif sum(v < 8000 for v in best_iterations) >= 4:
        tree_recommendation = "Convergencia antes de 8000: no conviene seguir aumentando."
    else:
        tree_recommendation = "Resultado mixto: no aumentar arboles sin otro diagnostico economico."

    for directory in (PREDICTIONS, SUBMISSIONS, METRICS, REPORTS):
        directory.mkdir(parents=True, exist_ok=True)
    oof_frame.to_csv(OOF_PATH, index=False)
    test_frame.to_csv(TEST_PATH, index=False)
    submission.to_csv(SUBMISSION_PATH, index=False)
    pd.DataFrame(curve_rows, columns=["fold", "iteration", "validation_auc"]).to_csv(
        CURVE_PATH, index=False
    )
    validate_submission(pd.read_csv(SUBMISSION_PATH), test, sample)

    params_text = ", ".join(f"{k}={v!r}" for k, v in MODEL.items())
    top20_text = "\n".join(
        f"{i}. {name}: {value:.8f}" for i, (name, value) in enumerate(top20.items(), 1)
    )
    lines = [
        f"experiment_id: {EXPERIMENT}",
        f"parameters: XGBClassifier({params_text}, early_stopping_rounds={EARLY_STOPPING})",
        f"features_reused_exactly_from_exp012_exp015: {NEW_FEATURES}",
        "cv_strategy: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        *(f"fold_{i}_roc_auc: {v:.8f}" for i, v in enumerate(scores, 1)),
        f"mean_roc_auc: {mean_auc:.8f}",
        f"std_roc_auc: {std_auc:.8f}",
        f"overall_oof_roc_auc: {overall_auc:.8f}",
        f"exp015_overall_oof_auc: {exp015_global:.8f}",
        f"global_delta_vs_exp015: {global_delta:+.8f}",
        f"fold_deltas_vs_exp015: {fold_deltas.tolist()}",
        f"std_change_vs_exp015: {std_change:+.8f}",
        *(f"fold_{i}_best_iteration_zero_based: {v}" for i, v in enumerate(best_iterations, 1)),
        *(f"fold_{i}_last_iteration_zero_based: {v}" for i, v in enumerate(last_iterations, 1)),
        *(f"fold_{i}_distance_best_to_9000_limit: {8999-v}" for i, v in enumerate(best_iterations, 1)),
        *(f"fold_{i}_early_stopping_activated: {str(v).lower()}" for i, v in enumerate(early_stopping_real, 1)),
        f"best_iteration_distribution: {distribution}",
        *(f"fold_{i}_seconds: {v:.2f}" for i, v in enumerate(fold_seconds, 1)),
        f"total_seconds: {total_seconds:.2f}",
        *(f"pearson_vs_{k.lower().replace('-', '')}: {v['pearson']:.8f}" for k, v in corr.items()),
        *(f"spearman_vs_{k.lower().replace('-', '')}: {v['spearman']:.8f}" for k, v in corr.items()),
        "top20_mean_normalized_gain:", top20_text,
        f"decision: {decision}",
        f"tree_recommendation: {tree_recommendation}",
        f"learning_curve_path: {CURVE_PATH}",
        "submission_validations: OK",
    ]
    METRICS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    update_log(mean_auc)

    print(f"Media={mean_auc:.8f} std={std_auc:.8f} overall={overall_auc:.8f}", flush=True)
    print(f"Deltas={fold_deltas.tolist()} global={global_delta:+.8f}", flush=True)
    print(f"Best={best_iterations} last={last_iterations} distribution={distribution}", flush=True)
    print(f"Early stopping real={early_stopping_real}", flush=True)
    print(f"Correlations={corr}", flush=True)
    print("Top 20:\n" + top20_text, flush=True)
    print(f"Decision={decision}", flush=True)
    print(f"Trees={tree_recommendation}", flush=True)
    print(f"Submission={SUBMISSION_PATH}", flush=True)
    print("Validaciones=OK", flush=True)


if __name__ == "__main__":
    main()
