"""EXP-002: CatBoost baseline with native categorical handling for Kaggle S6E8."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
SUBMISSIONS_DIR = PROJECT_ROOT / "outputs" / "submissions"

ID_COLUMN = "id"
TARGET = "addicted_label"
EXPERIMENT_ID = "EXP-002"
N_SPLITS = 5
RANDOM_SEED = 42

SUBMISSION_PATH = SUBMISSIONS_DIR / "submission_exp002_catboost.csv"
METRICS_PATH = METRICS_DIR / "exp002_catboost_metrics.txt"
OOF_PATH = METRICS_DIR / "exp002_catboost_oof.csv"
LOG_PATH = METRICS_DIR / "experiment_log.csv"

MODEL_PARAMS = {
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 7,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "random_seed": RANDOM_SEED,
    "verbose": False,
    "allow_writing_files": False,
}


def prepare_categorical_missing(
    frame: pd.DataFrame, categorical_columns: list[str]
) -> pd.DataFrame:
    prepared = frame.copy()
    for column in categorical_columns:
        prepared[column] = prepared[column].fillna("__MISSING__").astype(str)
    return prepared


def validate_submission(
    submission: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame
) -> None:
    if submission.columns.tolist() != [ID_COLUMN, TARGET]:
        raise ValueError(f"Columnas inválidas: {submission.columns.tolist()}")
    if len(submission) != 296_302 or len(submission) != len(test):
        raise ValueError(
            f"Cantidad de filas inválida: submission={len(submission)}, test={len(test)}"
        )
    if submission.isna().any().any():
        raise ValueError("El submission contiene NaN.")
    if not submission[TARGET].between(0, 1, inclusive="both").all():
        raise ValueError("El submission contiene probabilidades fuera de [0, 1].")
    if not submission[ID_COLUMN].equals(sample[ID_COLUMN]):
        raise ValueError("Los IDs no coinciden exactamente con sample_submission.")
    if not test[ID_COLUMN].equals(sample[ID_COLUMN]):
        raise ValueError("El orden de IDs de test no coincide con sample_submission.")


def read_exp001_auc() -> float:
    if not LOG_PATH.is_file():
        raise FileNotFoundError(f"No existe el registro de experimentos: {LOG_PATH}")
    log = pd.read_csv(LOG_PATH, dtype={"experiment_id": str})
    match = log.loc[log["experiment_id"] == "EXP-001", "cv_roc_auc"]
    if len(match) != 1:
        raise ValueError("Se esperaba exactamente una entrada EXP-001 en experiment_log.csv.")
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
                "model": "CatBoostClassifier",
                "features": "original_features",
                "cv_strategy": "StratifiedKFold_5",
                "cv_roc_auc": f"{mean_auc:.6f}",
                "kaggle_score": "",
                "notes": (
                    "CatBoost baseline; native numerical missing handling; categorical "
                    "missing as __MISSING__; fold ensemble"
                ),
            }
        ]
    )
    pd.concat([log, row], ignore_index=True).to_csv(LOG_PATH, index=False)


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

    if not {ID_COLUMN, TARGET}.issubset(train.columns):
        raise ValueError("Train no contiene las columnas id y addicted_label requeridas.")
    if sample.columns.tolist() != [ID_COLUMN, TARGET]:
        raise ValueError(f"Esquema inesperado en sample_submission: {sample.columns.tolist()}")

    feature_columns = [c for c in train.columns if c not in {ID_COLUMN, TARGET}]
    if test.columns.tolist() != [ID_COLUMN, *feature_columns]:
        raise ValueError("Las features de test no coinciden exactamente y en orden con train.")

    X_raw = train[feature_columns]
    y = train[TARGET]
    X_test_raw = test[feature_columns]
    numeric_columns = X_raw.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [c for c in feature_columns if c not in numeric_columns]

    X = prepare_categorical_missing(X_raw, categorical_columns)
    X_test = prepare_categorical_missing(X_test_raw, categorical_columns)

    # Verify that categorical missing values were replaced and numeric values were untouched.
    if X[categorical_columns].isna().any().any() or X_test[categorical_columns].isna().any().any():
        raise ValueError("Quedaron NaN en columnas categóricas.")
    if not X[numeric_columns].equals(X_raw[numeric_columns]):
        raise ValueError("Las columnas numéricas de train fueron modificadas.")
    if not X_test[numeric_columns].equals(X_test_raw[numeric_columns]):
        raise ValueError("Las columnas numéricas de test fueron modificadas.")

    print(f"Experimento: {EXPERIMENT_ID}")
    print(f"Filas train/test: {len(train):,} / {len(test):,}")
    print(f"Features numéricas: {numeric_columns}")
    print(f"Features categóricas: {categorical_columns}")

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    oof_predictions = np.zeros(len(train), dtype=np.float64)
    test_predictions = np.zeros(len(test), dtype=np.float64)
    fold_scores: list[float] = []
    best_iterations: list[int] = []

    total_start = perf_counter()
    for fold, (train_indices, valid_indices) in enumerate(cv.split(X, y), start=1):
        fold_start = perf_counter()
        model = CatBoostClassifier(**MODEL_PARAMS)
        model.fit(
            X.iloc[train_indices],
            y.iloc[train_indices],
            cat_features=categorical_columns,
            eval_set=(X.iloc[valid_indices], y.iloc[valid_indices]),
            early_stopping_rounds=100,
            verbose=False,
        )
        valid_predictions = model.predict_proba(X.iloc[valid_indices])[:, 1]
        fold_test_predictions = model.predict_proba(X_test)[:, 1]
        oof_predictions[valid_indices] = valid_predictions
        test_predictions += fold_test_predictions / N_SPLITS

        score = float(roc_auc_score(y.iloc[valid_indices], valid_predictions))
        best_iteration = int(model.get_best_iteration())
        fold_scores.append(score)
        best_iterations.append(best_iteration)
        fold_seconds = perf_counter() - fold_start
        print(
            f"Fold {fold} ROC AUC: {score:.6f} | "
            f"best_iteration: {best_iteration} | tiempo: {fold_seconds:.2f} s"
        )

    total_seconds = perf_counter() - total_start
    mean_auc = float(np.mean(fold_scores))
    std_auc = float(np.std(fold_scores))
    overall_oof_auc = float(roc_auc_score(y, oof_predictions))
    mean_best_iteration = float(np.mean(best_iterations))
    exp001_auc = read_exp001_auc()
    auc_difference = mean_auc - exp001_auc

    oof_frame = pd.DataFrame(
        {ID_COLUMN: train[ID_COLUMN], TARGET: y, "oof_probability": oof_predictions}
    )
    if oof_frame["oof_probability"].isna().any():
        raise ValueError("Las predicciones OOF contienen NaN.")
    oof_frame.to_csv(OOF_PATH, index=False)

    submission = pd.DataFrame(
        {ID_COLUMN: sample[ID_COLUMN].copy(), TARGET: test_predictions}
    )
    validate_submission(submission, test, sample)
    submission.to_csv(SUBMISSION_PATH, index=False)
    saved_submission = pd.read_csv(SUBMISSION_PATH)
    validate_submission(saved_submission, test, sample)

    parameter_text = ", ".join(f"{key}={value!r}" for key, value in MODEL_PARAMS.items())
    metrics_lines = [
        f"experiment_id: {EXPERIMENT_ID}",
        f"parameters: CatBoostClassifier({parameter_text})",
        "cv_strategy: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "early_stopping_rounds: 100",
        *(f"fold_{i}_roc_auc: {score:.6f}" for i, score in enumerate(fold_scores, 1)),
        f"mean_roc_auc: {mean_auc:.6f}",
        f"std_roc_auc: {std_auc:.6f}",
        f"overall_oof_roc_auc: {overall_oof_auc:.6f}",
        *(f"fold_{i}_best_iteration: {iteration}" for i, iteration in enumerate(best_iterations, 1)),
        f"mean_best_iteration: {mean_best_iteration:.2f}",
        f"total_seconds: {total_seconds:.2f}",
        f"exp001_mean_roc_auc: {exp001_auc:.6f}",
        f"absolute_cv_auc_difference_exp002_minus_exp001: {auc_difference:+.6f}",
    ]
    METRICS_PATH.write_text("\n".join(metrics_lines) + "\n", encoding="utf-8")
    update_experiment_log(mean_auc)

    print(f"Media ROC AUC: {mean_auc:.6f}")
    print(f"Desviación estándar: {std_auc:.6f}")
    print(f"Best iterations: {best_iterations}")
    print(f"Promedio best_iteration: {mean_best_iteration:.2f}")
    print(f"Tiempo total: {total_seconds:.2f} s")
    print(f"Diferencia vs EXP-001: {auc_difference:+.6f}")
    print(f"Submission: {SUBMISSION_PATH}")
    print("Validaciones del submission: OK")


if __name__ == "__main__":
    main()
