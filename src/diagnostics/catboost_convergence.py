"""Single-fold CatBoost convergence diagnostic up to 8000 trees."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold

from src.project_paths import PROJECT_ROOT


DATA_PATH = PROJECT_ROOT / "data" / "train.csv"
METRICS_PATH = PROJECT_ROOT / "outputs" / "metrics" / "catboost_convergence_diagnostic.txt"
CURVE_PATH = PROJECT_ROOT / "outputs" / "reports" / "catboost_convergence.csv"
LOG_PATH = PROJECT_ROOT / "outputs" / "metrics" / "experiment_log.csv"

ID_COLUMN = "id"
TARGET = "addicted_label"
CHECKPOINTS = list(range(500, 8001, 500))

MODEL_PARAMS = {
    "iterations": 8000,
    "learning_rate": 0.05,
    "depth": 7,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "random_seed": 42,
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


def update_exp003_kaggle_score() -> None:
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
    if (log["experiment_id"] == "EXP-003").sum() != 1:
        raise ValueError("Se esperaba exactamente una fila EXP-003 en experiment_log.csv.")
    original_ids = log["experiment_id"].tolist()
    log.loc[log["experiment_id"] == "EXP-003", "kaggle_score"] = "0.96497"
    if log["experiment_id"].tolist() != original_ids:
        raise RuntimeError("La actualización alteraría las filas del experiment log.")
    log.to_csv(LOG_PATH, index=False)


def recommendation(
    checkpoint_frame: pd.DataFrame, best_iteration: int, best_auc: float
) -> str:
    auc_4000 = float(checkpoint_frame.loc[checkpoint_frame["iterations"] == 4000, "validation_auc"].iloc[0])
    auc_8000 = float(checkpoint_frame.loc[checkpoint_frame["iterations"] == 8000, "validation_auc"].iloc[0])
    gain_4000_8000 = auc_8000 - auc_4000
    last_auc = auc_8000
    decline_from_best = best_auc - last_auc
    thousand_rows = checkpoint_frame[checkpoint_frame["iterations"] % 1000 == 0]
    recent_gains = thousand_rows["validation_auc"].diff().dropna().tail(4)

    if best_iteration < 7500 and decline_from_best > 0.0002:
        return (
            "C) La AUC alcanzó un máximo antes de 8000 y luego cayó de forma apreciable; "
            f"usar como referencia aproximadamente {best_iteration} iteraciones."
        )
    if gain_4000_8000 >= 0.001 and (recent_gains > 0).sum() >= 3:
        return (
            "A) La mejora entre 4000 y 8000 es clara y mayormente continua; recomendar "
            "EXP-004 con 8000 iteraciones y 5 folds."
        )
    return (
        "B) La mejora entre 4000 y 8000 es mínima o se está estabilizando; dejar de "
        "escalar árboles y pasar a otra estrategia."
    )


def main() -> None:
    train = pd.read_csv(DATA_PATH)
    if not {ID_COLUMN, TARGET}.issubset(train.columns):
        raise ValueError("Train no contiene las columnas id y addicted_label.")

    feature_columns = [c for c in train.columns if c not in {ID_COLUMN, TARGET}]
    X_raw = train[feature_columns]
    y = train[TARGET]
    numeric_columns = X_raw.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [c for c in feature_columns if c not in numeric_columns]
    X = prepare_categorical_missing(X_raw, categorical_columns)

    if X[categorical_columns].isna().any().any():
        raise ValueError("Quedaron NaN en las columnas categóricas.")
    if not X[numeric_columns].equals(X_raw[numeric_columns]):
        raise ValueError("Las columnas numéricas fueron modificadas.")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_indices, valid_indices = next(cv.split(X, y))
    model = CatBoostClassifier(**MODEL_PARAMS)

    print("Diagnóstico CatBoost: Fold 1, 8000 iteraciones, sin early stopping", flush=True)
    print(f"Features numéricas: {numeric_columns}", flush=True)
    print(f"Features categóricas: {categorical_columns}", flush=True)
    start = perf_counter()
    model.fit(
        X.iloc[train_indices],
        y.iloc[train_indices],
        cat_features=categorical_columns,
        eval_set=(X.iloc[valid_indices], y.iloc[valid_indices]),
        verbose=False,
    )
    total_seconds = perf_counter() - start

    evals = model.get_evals_result()
    if "validation" not in evals or "AUC" not in evals["validation"]:
        raise RuntimeError(f"No se encontró la curva validation/AUC: {evals.keys()}")
    auc_curve = np.asarray(evals["validation"]["AUC"], dtype=np.float64)
    if len(auc_curve) != MODEL_PARAMS["iterations"]:
        raise RuntimeError(f"Se esperaban 8000 puntos AUC y se obtuvieron {len(auc_curve)}.")

    curve_frame = pd.DataFrame(
        {
            "iterations": np.arange(1, len(auc_curve) + 1),
            "validation_auc": auc_curve,
        }
    )
    curve_frame["is_checkpoint"] = curve_frame["iterations"].isin(CHECKPOINTS)
    CURVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    curve_frame.to_csv(CURVE_PATH, index=False)

    # CatBoost's metric array is zero-indexed; tree count N is stored at index N - 1.
    checkpoint_frame = curve_frame.loc[
        curve_frame["iterations"].isin(CHECKPOINTS), ["iterations", "validation_auc"]
    ].copy()
    checkpoint_frame["gain_vs_previous_checkpoint"] = checkpoint_frame["validation_auc"].diff()
    best_zero_based_index = int(np.argmax(auc_curve))
    best_iteration = best_zero_based_index + 1
    best_auc = float(auc_curve[best_zero_based_index])
    auc_4000 = float(auc_curve[4000 - 1])
    auc_8000 = float(auc_curve[8000 - 1])
    gain_4000_8000 = auc_8000 - auc_4000

    thousand_checkpoints = list(range(1000, 8001, 1000))
    thousand_aucs = {iteration: float(auc_curve[iteration - 1]) for iteration in thousand_checkpoints}
    thousand_gains = {
        f"{previous}->{current}": thousand_aucs[current] - thousand_aucs[previous]
        for previous, current in zip(thousand_checkpoints, thousand_checkpoints[1:])
    }
    recommendation_text = recommendation(checkpoint_frame, best_iteration, best_auc)

    table_text = checkpoint_frame.to_string(
        index=False,
        formatters={
            "validation_auc": lambda value: f"{value:.6f}",
            "gain_vs_previous_checkpoint": lambda value: "" if pd.isna(value) else f"{value:+.6f}",
        },
    )
    parameter_text = ", ".join(f"{key}={value!r}" for key, value in MODEL_PARAMS.items())
    report_lines = [
        "CATBOOST CONVERGENCE DIAGNOSTIC - FOLD 1",
        f"parameters: CatBoostClassifier({parameter_text})",
        "cv: Fold 1 of StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "early_stopping: disabled",
        "",
        table_text,
        "",
        f"best_iteration_tree_count: {best_iteration}",
        f"best_validation_auc: {best_auc:.8f}",
        f"auc_at_4000: {auc_4000:.8f}",
        f"auc_at_8000: {auc_8000:.8f}",
        f"gain_4000_to_8000: {gain_4000_8000:+.8f}",
        "gains_per_1000_trees:",
        *(f"- {block}: {gain:+.8f}" for block, gain in thousand_gains.items()),
        f"total_seconds: {total_seconds:.2f}",
        f"recommendation: {recommendation_text}",
    ]
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    update_exp003_kaggle_score()

    print(table_text, flush=True)
    print(f"Mejor iteración: {best_iteration}", flush=True)
    print(f"Mejor AUC: {best_auc:.8f}", flush=True)
    print(f"Ganancia 4000 -> 8000: {gain_4000_8000:+.8f}", flush=True)
    print(f"Tiempo total: {total_seconds:.2f} s", flush=True)
    print(f"Recomendación: {recommendation_text}", flush=True)


if __name__ == "__main__":
    main()
