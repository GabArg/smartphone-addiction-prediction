"""EXP-003: isolate the effect of increasing CatBoost iterations to 4000."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.project_paths import DATA_DIR, METRICS_DIR, PROJECT_ROOT, SUBMISSIONS_DIR

ID_COLUMN = "id"
TARGET = "addicted_label"
EXPERIMENT_ID = "EXP-003"
N_SPLITS = 5
RANDOM_SEED = 42

SUBMISSION_PATH = SUBMISSIONS_DIR / "submission_exp003_catboost_4000.csv"
METRICS_PATH = METRICS_DIR / "exp003_catboost_metrics.txt"
OOF_PATH = METRICS_DIR / "exp003_catboost_oof.csv"
EXP002_METRICS_PATH = METRICS_DIR / "exp002_catboost_metrics.txt"
LOG_PATH = METRICS_DIR / "experiment_log.csv"

MODEL_PARAMS = {
    "iterations": 4000,
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


def parse_metrics(path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            parsed[key] = value
    return parsed


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
    if (log["experiment_id"] == "EXP-002").sum() != 1:
        raise ValueError("Se esperaba exactamente una fila EXP-002 en el log.")
    log.loc[log["experiment_id"] == "EXP-002", "kaggle_score"] = "0.95996"
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
                    "CatBoost 4000 iterations; native numerical missing handling; "
                    "categorical missing as __MISSING__; fold ensemble"
                ),
            }
        ]
    )
    pd.concat([log, row], ignore_index=True).to_csv(LOG_PATH, index=False)


def main() -> None:
    if not EXP002_METRICS_PATH.is_file():
        raise FileNotFoundError(f"Faltan las métricas de EXP-002: {EXP002_METRICS_PATH}")
    exp002_metrics = parse_metrics(EXP002_METRICS_PATH)
    exp002_scores = [float(exp002_metrics[f"fold_{i}_roc_auc"]) for i in range(1, 6)]
    exp002_best_iterations = [
        int(exp002_metrics[f"fold_{i}_best_iteration"]) for i in range(1, 6)
    ]
    exp002_mean = float(exp002_metrics["mean_roc_auc"])

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
    X = prepare_categorical_missing(X_raw, categorical_columns)
    X_test = prepare_categorical_missing(X_test_raw, categorical_columns)

    if X[categorical_columns].isna().any().any() or X_test[categorical_columns].isna().any().any():
        raise ValueError("Quedaron NaN categóricos.")
    if not X[numeric_columns].equals(X_raw[numeric_columns]):
        raise ValueError("Las variables numéricas de train fueron modificadas.")
    if not X_test[numeric_columns].equals(X_test_raw[numeric_columns]):
        raise ValueError("Las variables numéricas de test fueron modificadas.")

    print(f"Experimento: {EXPERIMENT_ID}", flush=True)
    print(f"Features numéricas: {numeric_columns}", flush=True)
    print(f"Features categóricas: {categorical_columns}", flush=True)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    oof_predictions = np.zeros(len(train), dtype=np.float64)
    test_predictions = np.zeros(len(test), dtype=np.float64)
    fold_scores: list[float] = []
    best_iterations: list[int] = []
    fold_times: list[float] = []
    total_start = perf_counter()

    for fold, (train_indices, valid_indices) in enumerate(cv.split(X, y), start=1):
        fold_start = perf_counter()
        model = CatBoostClassifier(**MODEL_PARAMS)
        model.fit(
            X.iloc[train_indices],
            y.iloc[train_indices],
            cat_features=categorical_columns,
            eval_set=(X.iloc[valid_indices], y.iloc[valid_indices]),
            early_stopping_rounds=200,
            verbose=False,
        )
        valid_predictions = model.predict_proba(X.iloc[valid_indices])[:, 1]
        oof_predictions[valid_indices] = valid_predictions
        test_predictions += model.predict_proba(X_test)[:, 1] / N_SPLITS
        score = float(roc_auc_score(y.iloc[valid_indices], valid_predictions))
        best_iteration = int(model.get_best_iteration())
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
    fold_differences = [new - old for new, old in zip(fold_scores, exp002_scores)]
    mean_difference = mean_auc - exp002_mean

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
    validate_submission(pd.read_csv(SUBMISSION_PATH), test, sample)

    parameter_text = ", ".join(f"{key}={value!r}" for key, value in MODEL_PARAMS.items())
    improvement_label = "mejora" if mean_difference > 0 else "empeora o no mejora"
    metrics_lines = [
        f"experiment_id: {EXPERIMENT_ID}",
        f"parameters: CatBoostClassifier({parameter_text})",
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
        f"exp002_mean_roc_auc: {exp002_mean:.6f}",
        *(f"fold_{i}_difference_vs_exp002: {value:+.6f}" for i, value in enumerate(fold_differences, 1)),
        f"mean_difference_vs_exp002: {mean_difference:+.6f}",
        f"exp002_best_iterations: {exp002_best_iterations}",
        f"conclusion: EXP-003 {improvement_label} EXP-002 en CV",
    ]
    METRICS_PATH.write_text("\n".join(metrics_lines) + "\n", encoding="utf-8")
    update_experiment_log(mean_auc)

    print(f"Media ROC AUC: {mean_auc:.6f}", flush=True)
    print(f"Desviación estándar: {std_auc:.6f}", flush=True)
    print(f"Best iterations: {best_iterations}", flush=True)
    print(f"Promedio best_iteration: {mean_best_iteration:.2f}", flush=True)
    print(f"Tiempo total: {total_seconds:.2f} s", flush=True)
    print(f"Diferencia vs EXP-002: {mean_difference:+.6f}", flush=True)
    print(f"Submission: {SUBMISSION_PATH}", flush=True)
    print("Validaciones del submission: OK", flush=True)


if __name__ == "__main__":
    main()
