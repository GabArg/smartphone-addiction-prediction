"""EXP-004: LightGBM baseline with native missing and categorical handling."""

from __future__ import annotations

from datetime import datetime
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

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
EXPERIMENT_ID = "EXP-004"
N_SPLITS = 5
RANDOM_STATE = 42

SUBMISSION_PATH = SUBMISSIONS_DIR / "submission_exp004_lightgbm.csv"
METRICS_PATH = METRICS_DIR / "exp004_lightgbm_metrics.txt"
OOF_PATH = PREDICTIONS_DIR / "oof_exp004_lightgbm.csv"
TEST_PREDICTIONS_PATH = PREDICTIONS_DIR / "test_exp004_lightgbm.csv"
EXP003_SOURCE_OOF_PATH = METRICS_DIR / "exp003_catboost_oof.csv"
EXP003_NORMALIZED_OOF_PATH = PREDICTIONS_DIR / "oof_exp003_catboost.csv"
LOG_PATH = METRICS_DIR / "experiment_log.csv"

MODEL_PARAMS = {
    "objective": "binary",
    "n_estimators": 5000,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def prepare_aligned_categories(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    categorical_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared_train = train_features.copy()
    prepared_test = test_features.copy()
    for column in categorical_columns:
        train_values = prepared_train[column].fillna("__MISSING__").astype(str)
        test_values = prepared_test[column].fillna("__MISSING__").astype(str)
        categories = pd.Index(pd.concat([train_values, test_values], ignore_index=True).unique())
        dtype = pd.CategoricalDtype(categories=categories, ordered=False)
        prepared_train[column] = train_values.astype(dtype)
        prepared_test[column] = test_values.astype(dtype)
    return prepared_train, prepared_test


def validate_submission(
    submission: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame
) -> None:
    if submission.columns.tolist() != [ID_COLUMN, TARGET]:
        raise ValueError(f"Columnas inválidas: {submission.columns.tolist()}")
    if len(submission) != 296_302 or len(submission) != len(test):
        raise ValueError(f"Cantidad de filas inválida: {len(submission)}")
    if submission.isna().any().any():
        raise ValueError("El submission contiene NaN.")
    if not submission[TARGET].between(0, 1, inclusive="both").all():
        raise ValueError("El submission contiene probabilidades fuera de [0, 1].")
    if not submission[ID_COLUMN].equals(sample[ID_COLUMN]):
        raise ValueError("Los IDs no coinciden exactamente con sample_submission.")
    if not test[ID_COLUMN].equals(sample[ID_COLUMN]):
        raise ValueError("Los IDs de test no coinciden con sample_submission.")


def normalize_exp003_oof() -> bool:
    if not EXP003_SOURCE_OOF_PATH.is_file():
        return False
    source = pd.read_csv(EXP003_SOURCE_OOF_PATH)
    expected = [ID_COLUMN, TARGET, "oof_probability"]
    if source.columns.tolist() != expected:
        raise ValueError(
            f"Esquema inesperado en OOF de EXP-003: {source.columns.tolist()}"
        )
    normalized = source.rename(
        columns={TARGET: "y_true", "oof_probability": "oof_prediction"}
    )
    if normalized.isna().any().any() or not normalized["oof_prediction"].between(0, 1).all():
        raise ValueError("Las OOF existentes de EXP-003 no son válidas.")
    normalized.to_csv(EXP003_NORMALIZED_OOF_PATH, index=False)
    return True


def read_exp003_auc() -> float:
    log = pd.read_csv(LOG_PATH, dtype={"experiment_id": str})
    match = log.loc[log["experiment_id"] == "EXP-003", "cv_roc_auc"]
    if len(match) != 1:
        raise ValueError("Se esperaba exactamente una fila EXP-003 en el log.")
    return float(match.iloc[0])


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
    log = log.loc[log["experiment_id"] != EXPERIMENT_ID].copy()
    row = pd.DataFrame(
        [
            {
                "experiment_id": EXPERIMENT_ID,
                "datetime": datetime.now().astimezone().isoformat(timespec="seconds"),
                "model": "LightGBM",
                "features": "original_features",
                "cv_strategy": "StratifiedKFold_5",
                "cv_roc_auc": f"{mean_auc:.6f}",
                "kaggle_score": "",
                "notes": (
                    "LightGBM baseline, native missing handling, categorical features, "
                    "5-fold ensemble"
                ),
            }
        ]
    )
    pd.concat([log, row], ignore_index=True).to_csv(LOG_PATH, index=False)


def main() -> None:
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    if not {ID_COLUMN, TARGET}.issubset(train.columns):
        raise ValueError("Train no contiene id y addicted_label.")
    if sample.columns.tolist() != [ID_COLUMN, TARGET]:
        raise ValueError(f"Esquema inesperado en sample_submission: {sample.columns.tolist()}")

    feature_columns = [c for c in train.columns if c not in {ID_COLUMN, TARGET}]
    if test.columns.tolist() != [ID_COLUMN, *feature_columns]:
        raise ValueError("Las features de test no coinciden exactamente con train.")
    X_raw = train[feature_columns]
    X_test_raw = test[feature_columns]
    y = train[TARGET]
    numeric_columns = X_raw.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [c for c in feature_columns if c not in numeric_columns]
    X, X_test = prepare_aligned_categories(X_raw, X_test_raw, categorical_columns)

    if not X[numeric_columns].equals(X_raw[numeric_columns]):
        raise ValueError("Las variables numéricas de train fueron modificadas.")
    if not X_test[numeric_columns].equals(X_test_raw[numeric_columns]):
        raise ValueError("Las variables numéricas de test fueron modificadas.")
    for column in categorical_columns:
        if not X[column].cat.categories.equals(X_test[column].cat.categories):
            raise ValueError(f"Categorías desalineadas en {column}.")
        if X[column].isna().any() or X_test[column].isna().any():
            raise ValueError(f"Quedaron missing categóricos en {column}.")

    print(f"Experimento: {EXPERIMENT_ID}", flush=True)
    print(f"Features numéricas: {numeric_columns}", flush=True)
    print(f"Features categóricas: {categorical_columns}", flush=True)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
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
        )[:, 1] / N_SPLITS
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
    mean_best_iteration = float(np.mean(best_iterations))
    exp003_auc = read_exp003_auc()
    difference_vs_exp003 = mean_auc - exp003_auc

    oof_frame = pd.DataFrame(
        {ID_COLUMN: train[ID_COLUMN], "y_true": y, "oof_prediction": oof_predictions}
    )
    test_frame = pd.DataFrame(
        {ID_COLUMN: sample[ID_COLUMN].copy(), "prediction": test_predictions}
    )
    if oof_frame.isna().any().any() or not oof_frame["oof_prediction"].between(0, 1).all():
        raise ValueError("Las OOF de EXP-004 no son válidas.")
    if test_frame.isna().any().any() or not test_frame["prediction"].between(0, 1).all():
        raise ValueError("Las predicciones de test de EXP-004 no son válidas.")
    oof_frame.to_csv(OOF_PATH, index=False)
    test_frame.to_csv(TEST_PREDICTIONS_PATH, index=False)

    submission = test_frame.rename(columns={"prediction": TARGET})
    validate_submission(submission, test, sample)
    submission.to_csv(SUBMISSION_PATH, index=False)
    validate_submission(pd.read_csv(SUBMISSION_PATH), test, sample)

    exp003_oof_available = normalize_exp003_oof()
    parameter_text = ", ".join(f"{key}={value!r}" for key, value in MODEL_PARAMS.items())
    metrics_lines = [
        f"experiment_id: {EXPERIMENT_ID}",
        f"parameters: LGBMClassifier({parameter_text})",
        "cv_strategy: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "early_stopping_rounds: 200",
        *(f"fold_{i}_roc_auc: {score:.6f}" for i, score in enumerate(fold_scores, 1)),
        f"mean_roc_auc: {mean_auc:.6f}",
        f"std_roc_auc: {std_auc:.6f}",
        f"overall_oof_roc_auc: {overall_oof_auc:.6f}",
        *(f"fold_{i}_best_iteration: {value}" for i, value in enumerate(best_iterations, 1)),
        f"mean_best_iteration: {mean_best_iteration:.2f}",
        *(f"fold_{i}_seconds: {value:.2f}" for i, value in enumerate(fold_times, 1)),
        f"total_seconds: {total_seconds:.2f}",
        f"exp003_mean_roc_auc: {exp003_auc:.6f}",
        f"difference_vs_exp003: {difference_vs_exp003:+.6f}",
        f"exp003_oof_available: {exp003_oof_available}",
    ]
    METRICS_PATH.write_text("\n".join(metrics_lines) + "\n", encoding="utf-8")
    update_experiment_log(mean_auc)

    print(f"Media ROC AUC: {mean_auc:.6f}", flush=True)
    print(f"Desviación estándar: {std_auc:.6f}", flush=True)
    print(f"Best iterations: {best_iterations}", flush=True)
    print(f"Promedio best_iteration: {mean_best_iteration:.2f}", flush=True)
    print(f"Tiempo total: {total_seconds:.2f} s", flush=True)
    print(f"Diferencia vs EXP-003: {difference_vs_exp003:+.6f}", flush=True)
    print(f"OOF EXP-003 disponibles: {exp003_oof_available}", flush=True)
    print(f"Submission: {SUBMISSION_PATH}", flush=True)
    print("Validaciones del submission: OK", flush=True)


if __name__ == "__main__":
    main()
