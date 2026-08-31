"""EXP-008: XGBoost baseline with stable ordinal categorical encoding."""

from __future__ import annotations

from datetime import datetime
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

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
EXPERIMENT_ID = "EXP-008"

OOF_PATH = PREDICTIONS_DIR / "oof_exp008_xgboost.csv"
TEST_PREDICTIONS_PATH = PREDICTIONS_DIR / "test_exp008_xgboost.csv"
SUBMISSION_PATH = SUBMISSIONS_DIR / "submission_exp008_xgboost.csv"
METRICS_PATH = METRICS_DIR / "exp008_xgboost_metrics.txt"
LOG_PATH = METRICS_DIR / "experiment_log.csv"

REFERENCE_OOF_PATHS = {
    "EXP-003": PREDICTIONS_DIR / "oof_exp003_catboost.csv",
    "EXP-004": PREDICTIONS_DIR / "oof_exp004_lightgbm.csv",
    "EXP-006": PREDICTIONS_DIR / "oof_exp006_lightgbm_features.csv",
}

MODEL_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_estimators": 6000,
    "learning_rate": 0.03,
    "max_depth": 7,
    "min_child_weight": 1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "gamma": 0.0,
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}


def ordinal_encode_categories(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    categorical_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, int]]]:
    encoded_train = train_features.copy()
    encoded_test = test_features.copy()
    mappings: dict[str, dict[str, int]] = {}
    for column in categorical_columns:
        train_values = encoded_train[column].fillna("__MISSING__").astype(str)
        test_values = encoded_test[column].fillna("__MISSING__").astype(str)
        categories = sorted(set(train_values.unique()) | set(test_values.unique()))
        mapping = {category: code for code, category in enumerate(categories)}
        mappings[column] = mapping
        encoded_train[column] = train_values.map(mapping).astype(np.int32)
        encoded_test[column] = test_values.map(mapping).astype(np.int32)
    return encoded_train, encoded_test, mappings


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


def diversity_correlations(xgb_oof: pd.DataFrame) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for experiment_id, path in REFERENCE_OOF_PATHS.items():
        reference = pd.read_csv(path)
        expected = [ID_COLUMN, "y_true", "oof_prediction"]
        if reference.columns.tolist() != expected:
            raise ValueError(f"Esquema OOF inesperado para {experiment_id}.")
        if reference.isna().any().any() or reference[ID_COLUMN].duplicated().any():
            raise ValueError(f"OOF inválidas para {experiment_id}.")
        aligned = xgb_oof.merge(
            reference.rename(
                columns={"y_true": "reference_y_true", "oof_prediction": "reference_prediction"}
            ),
            on=ID_COLUMN,
            how="inner",
            validate="one_to_one",
            sort=False,
        )
        if len(aligned) != len(xgb_oof):
            raise ValueError(f"IDs OOF no coinciden con {experiment_id}.")
        if not aligned["y_true"].equals(aligned["reference_y_true"]):
            raise ValueError(f"y_true no coincide con {experiment_id}.")
        results[experiment_id] = {
            "pearson": float(aligned["oof_prediction"].corr(aligned["reference_prediction"], method="pearson")),
            "spearman": float(aligned["oof_prediction"].corr(aligned["reference_prediction"], method="spearman")),
        }
    return results


def update_experiment_log(mean_auc: float) -> None:
    columns = [
        "experiment_id", "datetime", "model", "features", "cv_strategy",
        "cv_roc_auc", "kaggle_score", "notes",
    ]
    log = pd.read_csv(LOG_PATH, dtype=str, keep_default_na=False)
    if log.columns.tolist() != columns:
        raise ValueError(f"Encabezado inesperado en {LOG_PATH}: {log.columns.tolist()}")
    if (log["experiment_id"] == "EXP-007").sum() != 1:
        raise ValueError("Se esperaba exactamente una fila EXP-007.")
    log.loc[log["experiment_id"] == "EXP-007", "kaggle_score"] = "0.96555"
    log = log.loc[log["experiment_id"] != EXPERIMENT_ID].copy()
    row = pd.DataFrame([{
        "experiment_id": EXPERIMENT_ID,
        "datetime": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": "XGBoost",
        "features": "original_features",
        "cv_strategy": "StratifiedKFold_5",
        "cv_roc_auc": f"{mean_auc:.6f}",
        "kaggle_score": "",
        "notes": (
            "XGBoost baseline with ordinal categorical encoding, native numerical NaN, "
            "fold ensemble"
        ),
    }])
    pd.concat([log, row], ignore_index=True).to_csv(LOG_PATH, index=False)


def main() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    if not {ID_COLUMN, TARGET}.issubset(train.columns):
        raise ValueError("Train no contiene id y addicted_label.")

    feature_columns = [c for c in train.columns if c not in {ID_COLUMN, TARGET}]
    if test.columns.tolist() != [ID_COLUMN, *feature_columns]:
        raise ValueError("Las features de test no coinciden exactamente con train.")
    X_raw = train[feature_columns]
    X_test_raw = test[feature_columns]
    y = train[TARGET]
    numeric_columns = X_raw.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [c for c in feature_columns if c not in numeric_columns]
    X, X_test, category_mappings = ordinal_encode_categories(
        X_raw, X_test_raw, categorical_columns
    )
    if not X[numeric_columns].equals(X_raw[numeric_columns]):
        raise ValueError("Las variables numéricas de train fueron modificadas.")
    if not X_test[numeric_columns].equals(X_test_raw[numeric_columns]):
        raise ValueError("Las variables numéricas de test fueron modificadas.")
    if X[categorical_columns].isna().any().any() or X_test[categorical_columns].isna().any().any():
        raise ValueError("La codificación categórica produjo NaN.")

    print(f"Experimento: {EXPERIMENT_ID}", flush=True)
    print(f"Features numéricas: {numeric_columns}", flush=True)
    print(f"Features categóricas: {categorical_columns}", flush=True)
    print(f"Mappings categóricos: {category_mappings}", flush=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_predictions = np.zeros(len(train), dtype=np.float64)
    test_predictions = np.zeros(len(test), dtype=np.float64)
    fold_scores: list[float] = []
    best_iterations: list[int] = []
    fold_times: list[float] = []
    total_start = perf_counter()

    for fold, (train_indices, valid_indices) in enumerate(cv.split(X, y), start=1):
        fold_start = perf_counter()
        model = XGBClassifier(**MODEL_PARAMS, early_stopping_rounds=200)
        model.fit(
            X.iloc[train_indices],
            y.iloc[train_indices],
            eval_set=[(X.iloc[valid_indices], y.iloc[valid_indices])],
            verbose=False,
        )
        # XGBoost exposes best_iteration as a zero-based tree index.
        best_iteration = int(model.best_iteration)
        iteration_range = (0, best_iteration + 1)
        valid_predictions = model.predict_proba(
            X.iloc[valid_indices], iteration_range=iteration_range
        )[:, 1]
        oof_predictions[valid_indices] = valid_predictions
        fold_test_predictions = model.predict_proba(
            X_test, iteration_range=iteration_range
        )[:, 1].astype(np.float64)
        test_predictions += fold_test_predictions / 5.0
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

    # Guard against sub-ULP float32 accumulation outside the mathematical [0, 1] range.
    test_predictions = np.clip(test_predictions, 0.0, 1.0)
    oof_frame = pd.DataFrame(
        {ID_COLUMN: train[ID_COLUMN], "y_true": y, "oof_prediction": oof_predictions}
    )
    test_frame = pd.DataFrame(
        {ID_COLUMN: sample[ID_COLUMN].copy(), "prediction": test_predictions}
    )
    if oof_frame.isna().any().any() or not oof_frame["oof_prediction"].between(0, 1).all():
        raise ValueError("Las OOF de EXP-008 no son válidas.")
    if test_frame.isna().any().any() or not test_frame["prediction"].between(0, 1).all():
        raise ValueError("Las predicciones test de EXP-008 no son válidas.")
    oof_frame.to_csv(OOF_PATH, index=False)
    test_frame.to_csv(TEST_PREDICTIONS_PATH, index=False)

    submission = test_frame.rename(columns={"prediction": TARGET})
    validate_submission(submission, test, sample)
    submission.to_csv(SUBMISSION_PATH, index=False)
    validate_submission(pd.read_csv(SUBMISSION_PATH), test, sample)

    correlations = diversity_correlations(oof_frame)
    reference_aucs = {"EXP-003": 0.963592, "EXP-004": 0.963536, "EXP-006": 0.963664}
    differences = {name: mean_auc - auc for name, auc in reference_aucs.items()}
    min_pearson = min(values["pearson"] for values in correlations.values())
    if min_pearson < 0.99:
        diversity_assessment = (
            "Correlación menor a 0.99: diversidad potencialmente útil para un ensemble posterior."
        )
    else:
        diversity_assessment = (
            "Correlaciones cercanas o superiores a 0.99: diversidad limitada; validar por OOF "
            "antes de incluir en un ensemble."
        )

    parameter_text = ", ".join(f"{key}={value!r}" for key, value in MODEL_PARAMS.items())
    mapping_text = "; ".join(f"{column}={mapping}" for column, mapping in category_mappings.items())
    metrics_lines = [
        f"experiment_id: {EXPERIMENT_ID}",
        f"parameters: XGBClassifier({parameter_text}, early_stopping_rounds=200)",
        "categorical_treatment: stable ordinal codes fitted on combined train/test categories; no target used",
        f"category_mappings: {mapping_text}",
        "numeric_treatment: NaN preserved; no imputation; no scaling",
        "cv_strategy: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        *(f"fold_{i}_roc_auc: {score:.6f}" for i, score in enumerate(fold_scores, 1)),
        f"mean_roc_auc: {mean_auc:.6f}",
        f"std_roc_auc: {std_auc:.6f}",
        f"overall_oof_roc_auc: {overall_oof_auc:.6f}",
        *(f"fold_{i}_best_iteration_zero_based: {value}" for i, value in enumerate(best_iterations, 1)),
        f"mean_best_iteration_zero_based: {np.mean(best_iterations):.2f}",
        *(f"fold_{i}_seconds: {value:.2f}" for i, value in enumerate(fold_times, 1)),
        f"total_seconds: {total_seconds:.2f}",
        *(f"difference_vs_{name.lower().replace('-', '')}: {value:+.6f}" for name, value in differences.items()),
        *(f"pearson_vs_{name.lower().replace('-', '')}: {value['pearson']:.8f}" for name, value in correlations.items()),
        *(f"spearman_vs_{name.lower().replace('-', '')}: {value['spearman']:.8f}" for name, value in correlations.items()),
        f"diversity_assessment: {diversity_assessment}",
    ]
    METRICS_PATH.write_text("\n".join(metrics_lines) + "\n", encoding="utf-8")
    update_experiment_log(mean_auc)

    print(f"Media ROC AUC: {mean_auc:.6f}", flush=True)
    print(f"Desviación estándar: {std_auc:.6f}", flush=True)
    print(f"Best iterations: {best_iterations}", flush=True)
    print(f"Tiempo total: {total_seconds:.2f} s", flush=True)
    print(f"Diferencias: {differences}", flush=True)
    print(f"Correlaciones: {correlations}", flush=True)
    print(f"Diversidad: {diversity_assessment}", flush=True)
    print(f"Submission: {SUBMISSION_PATH}", flush=True)
    print("Validaciones del submission: OK", flush=True)


if __name__ == "__main__":
    main()
