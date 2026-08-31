"""EXP-012: XGBoost with explicit screen/social threshold and region features."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

_PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_FOR_IMPORTS))

from src.features.thresholds import (
    THRESHOLD_FEATURES,
    add_threshold_features,
    indicator,
    safe_divide,
)
from src.models.xgboost_baseline import (
    MODEL_PARAMS,
    ordinal_encode_categories,
    validate_submission,
)
from src.project_paths import (
    DATA_DIR,
    METRICS_DIR,
    OUTPUTS_DIR,
    PREDICTIONS_DIR,
    PROJECT_ROOT,
    SUBMISSIONS_DIR,
)


ID_COLUMN = "id"
TARGET = "addicted_label"
EXPERIMENT_ID = "EXP-012"

OOF_PATH = PREDICTIONS_DIR / "oof_exp012_xgboost_thresholds.csv"
TEST_PREDICTIONS_PATH = PREDICTIONS_DIR / "test_exp012_xgboost_thresholds.csv"
SUBMISSION_PATH = SUBMISSIONS_DIR / "submission_exp012_xgboost_thresholds.csv"
METRICS_PATH = METRICS_DIR / "exp012_xgboost_thresholds_metrics.txt"
LOG_PATH = METRICS_DIR / "experiment_log.csv"

REFERENCE_OOF_PATHS = {
    "EXP-008": PREDICTIONS_DIR / "oof_exp008_xgboost.csv",
    "EXP-010": PREDICTIONS_DIR / "oof_exp010_xgboost_features.csv",
    "EXP-003": PREDICTIONS_DIR / "oof_exp003_catboost.csv",
    "EXP-006": PREDICTIONS_DIR / "oof_exp006_lightgbm_features.csv",
}
REFERENCE_METRICS_PATHS = {
    "EXP-008": METRICS_DIR / "exp008_xgboost_metrics.txt",
    "EXP-010": METRICS_DIR / "exp010_xgboost_features_metrics.txt",
}

NEW_FEATURES = THRESHOLD_FEATURES


def parse_metrics(path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            parsed[key] = value
    return parsed


def load_reference(
    current: pd.DataFrame, experiment_id: str
) -> tuple[pd.DataFrame, float]:
    reference = pd.read_csv(REFERENCE_OOF_PATHS[experiment_id])
    if reference.columns.tolist() != [ID_COLUMN, "y_true", "oof_prediction"]:
        raise ValueError(f"Esquema OOF inesperado para {experiment_id}.")
    if reference.isna().any().any() or reference[ID_COLUMN].duplicated().any():
        raise ValueError(f"OOF inválidas para {experiment_id}.")
    aligned = current.merge(
        reference.rename(
            columns={"y_true": "reference_y_true", "oof_prediction": "reference_prediction"}
        ),
        on=ID_COLUMN, how="inner", validate="one_to_one", sort=False,
    )
    if len(aligned) != len(current) or not aligned["y_true"].equals(aligned["reference_y_true"]):
        raise ValueError(f"IDs o y_true no coinciden con {experiment_id}.")
    auc = float(roc_auc_score(aligned["y_true"], aligned["reference_prediction"]))
    return aligned, auc


def update_log(mean_auc: float) -> None:
    columns = [
        "experiment_id", "datetime", "model", "features", "cv_strategy",
        "cv_roc_auc", "kaggle_score", "notes",
    ]
    log = pd.read_csv(LOG_PATH, dtype=str, keep_default_na=False)
    if log.columns.tolist() != columns or (log["experiment_id"] == "EXP-011").sum() != 1:
        raise ValueError("Estado inesperado de experiment_log.csv.")
    log.loc[log["experiment_id"] == "EXP-011", "kaggle_score"] = "0.96611"
    log = log.loc[log["experiment_id"] != EXPERIMENT_ID].copy()
    row = pd.DataFrame([{
        "experiment_id": EXPERIMENT_ID,
        "datetime": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": "XGBoost",
        "features": "original_plus_threshold_region_features",
        "cv_strategy": "StratifiedKFold_5",
        "cv_roc_auc": f"{mean_auc:.6f}",
        "kaggle_score": "",
        "notes": (
            "threshold/region features based on daily_screen_time_hours and social_media_hours"
        ),
    }])
    pd.concat([log, row], ignore_index=True).to_csv(LOG_PATH, index=False)


def main() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    original_features = [column for column in train.columns if column not in {ID_COLUMN, TARGET}]
    if test.columns.tolist() != [ID_COLUMN, *original_features]:
        raise ValueError("Las features originales de test no coinciden con train.")

    X_raw = add_threshold_features(train[original_features])
    X_test_raw = add_threshold_features(test[original_features])
    y = train[TARGET]
    if not all(feature in X_raw and feature in X_test_raw for feature in NEW_FEATURES):
        raise ValueError("No se generaron todas las features de umbral en train/test.")
    if X_raw.columns.tolist() != X_test_raw.columns.tolist():
        raise ValueError("Train y test no tienen el mismo esquema final.")
    if np.isinf(X_raw[NEW_FEATURES].to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("Hay infinitos en las features nuevas de train.")
    if np.isinf(X_test_raw[NEW_FEATURES].to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("Hay infinitos en las features nuevas de test.")

    categorical_columns = [
        column for column in original_features
        if not pd.api.types.is_numeric_dtype(X_raw[column])
    ]
    numeric_columns = [column for column in X_raw.columns if column not in categorical_columns]
    X, X_test, category_mappings = ordinal_encode_categories(
        X_raw, X_test_raw, categorical_columns
    )
    if not X[numeric_columns].equals(X_raw[numeric_columns]):
        raise ValueError("Las variables numéricas de train fueron modificadas.")
    if not X_test[numeric_columns].equals(X_test_raw[numeric_columns]):
        raise ValueError("Las variables numéricas de test fueron modificadas.")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(cv.split(np.zeros(len(y)), y))
    reference_data: dict[str, dict[str, object]] = {}
    for experiment_id in ("EXP-008", "EXP-010"):
        reference = pd.read_csv(REFERENCE_OOF_PATHS[experiment_id])
        if not reference[ID_COLUMN].equals(train[ID_COLUMN]) or not reference["y_true"].equals(y):
            raise ValueError(f"OOF {experiment_id} no coincide con train.")
        fold_scores = [
            float(roc_auc_score(y.iloc[valid], reference["oof_prediction"].iloc[valid]))
            for _, valid in splits
        ]
        metrics = parse_metrics(REFERENCE_METRICS_PATHS[experiment_id])
        reference_data[experiment_id] = {
            "global_auc": float(roc_auc_score(y, reference["oof_prediction"])),
            "fold_scores": fold_scores,
            "std": float(np.std(fold_scores)),
            "best_iterations": [
                int(metrics[f"fold_{fold}_best_iteration_zero_based"]) for fold in range(1, 6)
            ],
        }

    print(f"Experimento: {EXPERIMENT_ID}", flush=True)
    print(f"Nuevas features ({len(NEW_FEATURES)}): {NEW_FEATURES}", flush=True)
    print(f"Mappings categóricos: {category_mappings}", flush=True)

    oof_predictions = np.zeros(len(train), dtype=np.float64)
    test_predictions = np.zeros(len(test), dtype=np.float64)
    fold_scores: list[float] = []
    best_iterations: list[int] = []
    fold_times: list[float] = []
    fold_importances: list[dict[str, float]] = []
    total_start = perf_counter()

    for fold, (train_indices, valid_indices) in enumerate(splits, start=1):
        fold_start = perf_counter()
        model = XGBClassifier(**MODEL_PARAMS, early_stopping_rounds=200)
        model.fit(
            X.iloc[train_indices], y.iloc[train_indices],
            eval_set=[(X.iloc[valid_indices], y.iloc[valid_indices])], verbose=False,
        )
        best_iteration = int(model.best_iteration)
        iteration_range = (0, best_iteration + 1)
        valid_predictions = model.predict_proba(
            X.iloc[valid_indices], iteration_range=iteration_range
        )[:, 1]
        oof_predictions[valid_indices] = valid_predictions
        fold_test = model.predict_proba(X_test, iteration_range=iteration_range)[:, 1].astype(np.float64)
        test_predictions += fold_test / 5.0
        score = float(roc_auc_score(y.iloc[valid_indices], valid_predictions))

        raw_gain = model.get_booster().get_score(importance_type="gain")
        total_gain = sum(raw_gain.values())
        normalized_gain = {
            feature: (raw_gain.get(feature, 0.0) / total_gain if total_gain else 0.0)
            for feature in X.columns
        }
        fold_importances.append(normalized_gain)
        fold_seconds = perf_counter() - fold_start
        fold_scores.append(score)
        best_iterations.append(best_iteration)
        fold_times.append(fold_seconds)
        print(
            f"Fold {fold} ROC AUC: {score:.6f} | best_iteration: {best_iteration} "
            f"| tiempo: {fold_seconds:.2f} s",
            flush=True,
        )

    total_seconds = perf_counter() - total_start
    test_predictions = np.clip(test_predictions, 0.0, 1.0)
    mean_auc = float(np.mean(fold_scores))
    std_auc = float(np.std(fold_scores))
    overall_oof_auc = float(roc_auc_score(y, oof_predictions))

    importance_frame = pd.DataFrame(fold_importances)
    mean_importance = importance_frame.mean(axis=0).sort_values(ascending=False)
    top20 = mean_importance.head(20)

    oof_frame = pd.DataFrame(
        {ID_COLUMN: train[ID_COLUMN], "y_true": y, "oof_prediction": oof_predictions}
    )
    test_frame = pd.DataFrame(
        {ID_COLUMN: sample[ID_COLUMN].copy(), "prediction": test_predictions}
    )
    if oof_frame.isna().any().any() or not oof_frame["oof_prediction"].between(0, 1).all():
        raise ValueError("Las OOF de EXP-012 no son válidas.")
    if test_frame.isna().any().any() or not test_frame["prediction"].between(0, 1).all():
        raise ValueError("Las predicciones test de EXP-012 no son válidas.")
    oof_frame.to_csv(OOF_PATH, index=False)
    test_frame.to_csv(TEST_PREDICTIONS_PATH, index=False)
    submission = test_frame.rename(columns={"prediction": TARGET})
    validate_submission(submission, test, sample)
    submission.to_csv(SUBMISSION_PATH, index=False)
    validate_submission(pd.read_csv(SUBMISSION_PATH), test, sample)

    correlations: dict[str, dict[str, float]] = {}
    for experiment_id in REFERENCE_OOF_PATHS:
        aligned, _ = load_reference(oof_frame, experiment_id)
        correlations[experiment_id] = {
            "pearson": float(aligned["oof_prediction"].corr(aligned["reference_prediction"], method="pearson")),
            "spearman": float(aligned["oof_prediction"].corr(aligned["reference_prediction"], method="spearman")),
        }

    comparisons: dict[str, dict[str, object]] = {}
    for experiment_id, data in reference_data.items():
        comparisons[experiment_id] = {
            "global_difference": overall_oof_auc - float(data["global_auc"]),
            "fold_differences": [new - old for new, old in zip(fold_scores, data["fold_scores"])],
            "std_change": std_auc - float(data["std"]),
            "best_iteration_changes": [
                new - old for new, old in zip(best_iterations, data["best_iterations"])
            ],
        }
    difference_vs_exp008 = float(comparisons["EXP-008"]["global_difference"])
    if difference_vs_exp008 >= 0.00010:
        decision = "Mejora >= +0.00010: recomendar subir a Kaggle."
    elif difference_vs_exp008 >= 0.00003:
        decision = "Mejora entre +0.00003 y +0.00010: marginal pero interesante."
    elif difference_vs_exp008 >= 0:
        decision = "Mejora < +0.00003: empate práctico."
    else:
        decision = "Empeora: las features de regiones no ayudan al modelo."

    parameter_text = ", ".join(f"{key}={value!r}" for key, value in MODEL_PARAMS.items())
    top20_text = "\n".join(
        f"{rank}. {feature}: {gain:.8f}"
        for rank, (feature, gain) in enumerate(top20.items(), start=1)
    )
    metrics_lines = [
        f"experiment_id: {EXPERIMENT_ID}",
        f"parameters: XGBClassifier({parameter_text}, early_stopping_rounds=200)",
        f"new_features: {NEW_FEATURES}",
        "cv_strategy: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        *(f"fold_{fold}_roc_auc: {score:.8f}" for fold, score in enumerate(fold_scores, 1)),
        f"mean_roc_auc: {mean_auc:.8f}",
        f"std_roc_auc: {std_auc:.8f}",
        f"overall_oof_roc_auc: {overall_oof_auc:.8f}",
        *(f"fold_{fold}_best_iteration_zero_based: {value}" for fold, value in enumerate(best_iterations, 1)),
        *(f"fold_{fold}_seconds: {value:.2f}" for fold, value in enumerate(fold_times, 1)),
        f"total_seconds: {total_seconds:.2f}",
        *(f"{experiment_id.lower().replace('-', '')}_global_difference: {data['global_difference']:+.8f}" for experiment_id, data in comparisons.items()),
        *(f"{experiment_id.lower().replace('-', '')}_fold_differences: {data['fold_differences']}" for experiment_id, data in comparisons.items()),
        *(f"{experiment_id.lower().replace('-', '')}_std_change: {data['std_change']:+.8f}" for experiment_id, data in comparisons.items()),
        *(f"{experiment_id.lower().replace('-', '')}_best_iteration_changes: {data['best_iteration_changes']}" for experiment_id, data in comparisons.items()),
        "top20_mean_normalized_gain:", top20_text,
        *(f"pearson_vs_{name.lower().replace('-', '')}: {values['pearson']:.8f}" for name, values in correlations.items()),
        *(f"spearman_vs_{name.lower().replace('-', '')}: {values['spearman']:.8f}" for name, values in correlations.items()),
        f"decision: {decision}",
    ]
    METRICS_PATH.write_text("\n".join(metrics_lines) + "\n", encoding="utf-8")
    update_log(mean_auc)

    print(f"Media ROC AUC: {mean_auc:.8f}", flush=True)
    print(f"Std ROC AUC: {std_auc:.8f}", flush=True)
    print(f"Best iterations: {best_iterations}", flush=True)
    print(f"Tiempo total: {total_seconds:.2f} s", flush=True)
    print(f"Comparaciones: {comparisons}", flush=True)
    print("Top 20 mean normalized gain:\n" + top20_text, flush=True)
    print(f"Correlaciones: {correlations}", flush=True)
    print(f"Decisión: {decision}", flush=True)
    print(f"Submission: {SUBMISSION_PATH}", flush=True)
    print("Validaciones del submission: OK", flush=True)


if __name__ == "__main__":
    main()
