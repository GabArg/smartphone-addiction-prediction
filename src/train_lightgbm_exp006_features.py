"""EXP-006: EXP-004 LightGBM plus eight specified behavioral features."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from train_lightgbm_exp004 import (
    MODEL_PARAMS,
    prepare_aligned_categories,
    validate_submission,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"

ID_COLUMN = "id"
TARGET = "addicted_label"
EXPERIMENT_ID = "EXP-006"
NEW_FEATURES = [
    "total_leisure_hours",
    "screen_to_sleep_ratio",
    "social_share",
    "gaming_share",
    "weekend_screen_ratio",
    "notifications_per_screen_hour",
    "app_opens_per_screen_hour",
    "screen_minus_work",
]

OOF_PATH = PREDICTIONS_DIR / "oof_exp006_lightgbm_features.csv"
TEST_PREDICTIONS_PATH = PREDICTIONS_DIR / "test_exp006_lightgbm_features.csv"
SUBMISSION_PATH = SUBMISSIONS_DIR / "submission_exp006_lightgbm_features.csv"
METRICS_PATH = METRICS_DIR / "exp006_lightgbm_features_metrics.txt"
EXP004_METRICS_PATH = METRICS_DIR / "exp004_lightgbm_metrics.txt"
LOG_PATH = METRICS_DIR / "experiment_log.csv"


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=numerator.index, dtype=np.float64)
    valid = denominator.notna() & denominator.ne(0)
    result.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return result.replace([np.inf, -np.inf], np.nan)


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    engineered = frame.copy()
    engineered["total_leisure_hours"] = (
        engineered["social_media_hours"] + engineered["gaming_hours"]
    )
    engineered["screen_to_sleep_ratio"] = safe_divide(
        engineered["daily_screen_time_hours"], engineered["sleep_hours"]
    )
    engineered["social_share"] = safe_divide(
        engineered["social_media_hours"], engineered["daily_screen_time_hours"]
    )
    engineered["gaming_share"] = safe_divide(
        engineered["gaming_hours"], engineered["daily_screen_time_hours"]
    )
    engineered["weekend_screen_ratio"] = safe_divide(
        engineered["weekend_screen_time"], engineered["daily_screen_time_hours"]
    )
    engineered["notifications_per_screen_hour"] = safe_divide(
        engineered["notifications_per_day"], engineered["daily_screen_time_hours"]
    )
    engineered["app_opens_per_screen_hour"] = safe_divide(
        engineered["app_opens_per_day"], engineered["daily_screen_time_hours"]
    )
    engineered["screen_minus_work"] = (
        engineered["daily_screen_time_hours"] - engineered["work_study_hours"]
    )
    return engineered


def parse_metrics(path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            parsed[key] = value
    return parsed


def update_experiment_log(mean_auc: float) -> None:
    columns = [
        "experiment_id",
        "datetime",
        "model",
        "features",
        "cv_strategy",
        "cv_roc_auc",
        "kaggle_score",
        "notes",
    ]
    log = pd.read_csv(LOG_PATH, dtype=str, keep_default_na=False)
    if log.columns.tolist() != columns:
        raise ValueError(f"Encabezado inesperado en {LOG_PATH}: {log.columns.tolist()}")
    for experiment_id in ("EXP-004", "EXP-005"):
        if (log["experiment_id"] == experiment_id).sum() != 1:
            raise ValueError(f"Se esperaba exactamente una fila {experiment_id} en el log.")
    log.loc[log["experiment_id"] == "EXP-004", "kaggle_score"] = "0.96515"
    log.loc[log["experiment_id"] == "EXP-005", "kaggle_score"] = "0.96536"
    log = log.loc[log["experiment_id"] != EXPERIMENT_ID].copy()
    row = pd.DataFrame(
        [
            {
                "experiment_id": EXPERIMENT_ID,
                "datetime": datetime.now().astimezone().isoformat(timespec="seconds"),
                "model": "LightGBM",
                "features": "original_plus_engineered_v1",
                "cv_strategy": "StratifiedKFold_5",
                "cv_roc_auc": f"{mean_auc:.6f}",
                "kaggle_score": "",
                "notes": (
                    "LightGBM with 8 engineered behavioral ratio/difference features"
                ),
            }
        ]
    )
    pd.concat([log, row], ignore_index=True).to_csv(LOG_PATH, index=False)


def main() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    if not {ID_COLUMN, TARGET}.issubset(train.columns):
        raise ValueError("Train no contiene id y addicted_label.")

    original_features = [c for c in train.columns if c not in {ID_COLUMN, TARGET}]
    if test.columns.tolist() != [ID_COLUMN, *original_features]:
        raise ValueError("Las features originales de test no coinciden con train.")
    X_train_raw = engineer_features(train[original_features])
    X_test_raw = engineer_features(test[original_features])
    y = train[TARGET]

    if not all(column in X_train_raw and column in X_test_raw for column in NEW_FEATURES):
        raise ValueError("No se generaron todas las nuevas features en train y test.")
    if X_train_raw.columns.tolist() != X_test_raw.columns.tolist():
        raise ValueError("Train y test no tienen exactamente las mismas features finales.")
    if np.isinf(X_train_raw[NEW_FEATURES].to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("Hay infinitos en las nuevas features de train.")
    if np.isinf(X_test_raw[NEW_FEATURES].to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("Hay infinitos en las nuevas features de test.")

    categorical_columns = [
        c for c in original_features if not pd.api.types.is_numeric_dtype(X_train_raw[c])
    ]
    numeric_columns = [c for c in X_train_raw.columns if c not in categorical_columns]
    X, X_test = prepare_aligned_categories(
        X_train_raw, X_test_raw, categorical_columns
    )
    if not X[numeric_columns].equals(X_train_raw[numeric_columns]):
        raise ValueError("Las variables numéricas de train fueron modificadas.")
    if not X_test[numeric_columns].equals(X_test_raw[numeric_columns]):
        raise ValueError("Las variables numéricas de test fueron modificadas.")

    exp004_metrics = parse_metrics(EXP004_METRICS_PATH)
    exp004_fold_scores = [
        float(exp004_metrics[f"fold_{fold}_roc_auc"]) for fold in range(1, 6)
    ]
    exp004_mean = float(exp004_metrics["mean_roc_auc"])

    print(f"Experimento: {EXPERIMENT_ID}", flush=True)
    print(f"Features originales: {original_features}", flush=True)
    print(f"Nuevas features: {NEW_FEATURES}", flush=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_predictions = np.zeros(len(train), dtype=np.float64)
    test_predictions = np.zeros(len(test), dtype=np.float64)
    fold_scores: list[float] = []
    best_iterations: list[int] = []
    fold_times: list[float] = []
    total_start = perf_counter()

    for fold, (train_indices, valid_indices) in enumerate(cv.split(X, y), start=1):
        fold_start = perf_counter()
        model = LGBMClassifier(**MODEL_PARAMS)
        model.fit(
            X.iloc[train_indices],
            y.iloc[train_indices],
            eval_set=[(X.iloc[valid_indices], y.iloc[valid_indices])],
            eval_metric="auc",
            categorical_feature=categorical_columns,
            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)],
        )
        best_iteration = int(model.best_iteration_)
        valid_predictions = model.predict_proba(
            X.iloc[valid_indices], num_iteration=best_iteration
        )[:, 1]
        oof_predictions[valid_indices] = valid_predictions
        test_predictions += model.predict_proba(
            X_test, num_iteration=best_iteration
        )[:, 1] / 5
        score = float(roc_auc_score(y.iloc[valid_indices], valid_predictions))
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
    mean_auc = float(np.mean(fold_scores))
    std_auc = float(np.std(fold_scores))
    overall_oof_auc = float(roc_auc_score(y, oof_predictions))
    fold_differences = [new - old for new, old in zip(fold_scores, exp004_fold_scores)]
    difference_vs_exp004 = mean_auc - exp004_mean

    oof_frame = pd.DataFrame(
        {ID_COLUMN: train[ID_COLUMN], "y_true": y, "oof_prediction": oof_predictions}
    )
    test_frame = pd.DataFrame(
        {ID_COLUMN: sample[ID_COLUMN].copy(), "prediction": test_predictions}
    )
    if oof_frame.isna().any().any() or not oof_frame["oof_prediction"].between(0, 1).all():
        raise ValueError("Las OOF de EXP-006 no son válidas.")
    if test_frame.isna().any().any() or not test_frame["prediction"].between(0, 1).all():
        raise ValueError("Las predicciones test de EXP-006 no son válidas.")
    oof_frame.to_csv(OOF_PATH, index=False)
    test_frame.to_csv(TEST_PREDICTIONS_PATH, index=False)

    submission = test_frame.rename(columns={"prediction": TARGET})
    validate_submission(submission, test, sample)
    submission.to_csv(SUBMISSION_PATH, index=False)
    validate_submission(pd.read_csv(SUBMISSION_PATH), test, sample)

    if difference_vs_exp004 >= 0.00010:
        decision = "Mejora >= +0.00010: conservar y recomendar subir a Kaggle."
    elif difference_vs_exp004 > 0:
        decision = "Mejora positiva pero < +0.00010: mejora marginal."
    else:
        decision = "No mejora: este bloque de features no aporta en CV."

    parameter_text = ", ".join(f"{key}={value!r}" for key, value in MODEL_PARAMS.items())
    metrics_lines = [
        f"experiment_id: {EXPERIMENT_ID}",
        f"original_features: {original_features}",
        f"engineered_features: {NEW_FEATURES}",
        f"parameters: LGBMClassifier({parameter_text})",
        "cv_strategy: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "early_stopping_rounds: 200",
        *(f"fold_{i}_roc_auc: {score:.6f}" for i, score in enumerate(fold_scores, 1)),
        f"mean_roc_auc: {mean_auc:.6f}",
        f"std_roc_auc: {std_auc:.6f}",
        f"overall_oof_roc_auc: {overall_oof_auc:.6f}",
        *(f"fold_{i}_best_iteration: {value}" for i, value in enumerate(best_iterations, 1)),
        f"mean_best_iteration: {np.mean(best_iterations):.2f}",
        *(f"fold_{i}_seconds: {value:.2f}" for i, value in enumerate(fold_times, 1)),
        f"total_seconds: {total_seconds:.2f}",
        f"exp004_mean_roc_auc: {exp004_mean:.6f}",
        *(f"fold_{i}_difference_vs_exp004: {value:+.6f}" for i, value in enumerate(fold_differences, 1)),
        f"difference_vs_exp004: {difference_vs_exp004:+.6f}",
        f"decision: {decision}",
    ]
    METRICS_PATH.write_text("\n".join(metrics_lines) + "\n", encoding="utf-8")
    update_experiment_log(mean_auc)

    print(f"Media ROC AUC: {mean_auc:.6f}", flush=True)
    print(f"Desviación estándar: {std_auc:.6f}", flush=True)
    print(f"Best iterations: {best_iterations}", flush=True)
    print(f"Tiempo total: {total_seconds:.2f} s", flush=True)
    print(f"Diferencia vs EXP-004: {difference_vs_exp004:+.6f}", flush=True)
    print(f"Decisión: {decision}", flush=True)
    print(f"Submission: {SUBMISSION_PATH}", flush=True)
    print("Validaciones del submission: OK", flush=True)


if __name__ == "__main__":
    main()
