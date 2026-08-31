"""EXP-001: reproducible Logistic Regression baseline for Kaggle S6E8."""

from __future__ import annotations

from datetime import datetime
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.project_paths import DATA_DIR, METRICS_DIR, PROJECT_ROOT, SUBMISSIONS_DIR


TARGET = "addicted_label"
ID_COLUMN = "id"
EXPERIMENT_ID = "EXP-001"
RANDOM_STATE = 42
N_SPLITS = 5

SUBMISSION_PATH = SUBMISSIONS_DIR / "submission_exp001_logistic.csv"
METRICS_PATH = METRICS_DIR / "exp001_logistic_metrics.txt"
OOF_PATH = METRICS_DIR / "exp001_logistic_oof.csv"
LOG_PATH = METRICS_DIR / "experiment_log.csv"


def build_pipeline(numeric_columns: list[str], categorical_columns: list[str]) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )
    model = LogisticRegression(max_iter=2000, solver="liblinear")
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


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
    if LOG_PATH.exists():
        log = pd.read_csv(LOG_PATH, dtype=str, keep_default_na=False)
        if log.columns.tolist() != columns:
            raise ValueError(f"Encabezado inesperado en {LOG_PATH}: {log.columns.tolist()}")
        log = log.loc[log["experiment_id"] != EXPERIMENT_ID].copy()
    else:
        log = pd.DataFrame(columns=columns)

    row = pd.DataFrame(
        [
            {
                "experiment_id": EXPERIMENT_ID,
                "datetime": datetime.now().astimezone().isoformat(timespec="seconds"),
                "model": "LogisticRegression",
                "features": "original_features",
                "cv_strategy": "StratifiedKFold_5",
                "cv_roc_auc": f"{mean_auc:.6f}",
                "kaggle_score": "",
                "notes": "baseline con median imputation + scaling + one-hot encoding",
            }
        ]
    )
    pd.concat([log, row], ignore_index=True).to_csv(LOG_PATH, index=False)


def validate_submission(submission: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame) -> None:
    expected_columns = [ID_COLUMN, TARGET]
    if submission.columns.tolist() != expected_columns:
        raise ValueError(f"Columnas inválidas: {submission.columns.tolist()}")
    if len(submission) != len(test):
        raise ValueError("La cantidad de filas del submission no coincide con test.")
    if submission.isna().any().any():
        raise ValueError("El submission contiene NaN.")
    if not submission[TARGET].between(0, 1, inclusive="both").all():
        raise ValueError("Hay probabilidades fuera del intervalo [0, 1].")
    if not submission[ID_COLUMN].equals(sample[ID_COLUMN]):
        raise ValueError("Los IDs del submission no coinciden exactamente con sample_submission.")
    if not test[ID_COLUMN].equals(sample[ID_COLUMN]):
        raise ValueError("El orden de IDs de test no coincide con sample_submission.")


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

    required_train = {ID_COLUMN, TARGET}
    if not required_train.issubset(train.columns):
        raise ValueError(f"Faltan columnas requeridas en train: {required_train - set(train.columns)}")
    if ID_COLUMN not in test or sample.columns.tolist() != [ID_COLUMN, TARGET]:
        raise ValueError("Esquema inesperado en test o sample_submission.")

    feature_columns = [c for c in train.columns if c not in {ID_COLUMN, TARGET}]
    if test.columns.tolist() != [ID_COLUMN, *feature_columns]:
        raise ValueError("Las features de test no coinciden exactamente y en orden con train.")

    X = train[feature_columns]
    y = train[TARGET]
    X_test = test[feature_columns]
    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [c for c in feature_columns if c not in numeric_columns]

    print(f"Experimento: {EXPERIMENT_ID}")
    print(f"Filas train/test: {len(train):,} / {len(test):,}")
    print(f"Features numéricas: {numeric_columns}")
    print(f"Features categóricas: {categorical_columns}")

    pipeline = build_pipeline(numeric_columns, categorical_columns)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_predictions = np.zeros(len(train), dtype=np.float64)
    fold_scores: list[float] = []

    total_start = perf_counter()
    cv_start = perf_counter()
    for fold, (train_indices, valid_indices) in enumerate(cv.split(X, y), start=1):
        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X.iloc[train_indices], y.iloc[train_indices])
        predictions = fold_pipeline.predict_proba(X.iloc[valid_indices])[:, 1]
        oof_predictions[valid_indices] = predictions
        score = roc_auc_score(y.iloc[valid_indices], predictions)
        fold_scores.append(score)
        print(f"Fold {fold} ROC AUC: {score:.6f}")
    cv_seconds = perf_counter() - cv_start

    mean_auc = float(np.mean(fold_scores))
    std_auc = float(np.std(fold_scores))
    overall_oof_auc = float(roc_auc_score(y, oof_predictions))
    print(f"Media ROC AUC: {mean_auc:.6f}")
    print(f"Desviación estándar: {std_auc:.6f}")

    oof_frame = pd.DataFrame(
        {ID_COLUMN: train[ID_COLUMN], TARGET: y, "oof_probability": oof_predictions}
    )
    if oof_frame["oof_probability"].isna().any():
        raise ValueError("Las predicciones OOF contienen NaN.")
    oof_frame.to_csv(OOF_PATH, index=False)

    final_fit_start = perf_counter()
    pipeline.fit(X, y)
    final_fit_seconds = perf_counter() - final_fit_start
    test_probabilities = pipeline.predict_proba(X_test)[:, 1]
    feature_count = len(pipeline.named_steps["preprocessor"].get_feature_names_out())

    submission = pd.DataFrame(
        {ID_COLUMN: sample[ID_COLUMN].copy(), TARGET: test_probabilities}
    )
    validate_submission(submission, test, sample)
    submission.to_csv(SUBMISSION_PATH, index=False)

    # Read it back so validation also covers the serialized artifact.
    saved_submission = pd.read_csv(SUBMISSION_PATH)
    validate_submission(saved_submission, test, sample)

    total_seconds = perf_counter() - total_start
    metrics_lines = [
        f"experiment_id: {EXPERIMENT_ID}",
        "model: LogisticRegression(max_iter=2000, solver='liblinear')",
        "cv_strategy: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        *(f"fold_{i}_roc_auc: {score:.6f}" for i, score in enumerate(fold_scores, 1)),
        f"mean_roc_auc: {mean_auc:.6f}",
        f"std_roc_auc: {std_auc:.6f}",
        f"overall_oof_roc_auc: {overall_oof_auc:.6f}",
        f"preprocessed_feature_count: {feature_count}",
        f"cv_training_seconds: {cv_seconds:.2f}",
        f"final_fit_seconds: {final_fit_seconds:.2f}",
        f"total_training_and_prediction_seconds: {total_seconds:.2f}",
    ]
    METRICS_PATH.write_text("\n".join(metrics_lines) + "\n", encoding="utf-8")
    update_experiment_log(mean_auc)

    print(f"Features después del preprocessing: {feature_count}")
    print(f"Tiempo CV: {cv_seconds:.2f} s")
    print(f"Tiempo ajuste final: {final_fit_seconds:.2f} s")
    print(f"Tiempo total: {total_seconds:.2f} s")
    print(f"Submission: {SUBMISSION_PATH}")
    print("Validaciones del submission: OK")


if __name__ == "__main__":
    main()
