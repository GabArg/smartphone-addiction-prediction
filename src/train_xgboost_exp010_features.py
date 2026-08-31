"""EXP-010: EXP-008 XGBoost plus the exact eight EXP-006 engineered features."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from train_lightgbm_exp006_features import NEW_FEATURES, engineer_features
from train_xgboost_exp008 import MODEL_PARAMS, ordinal_encode_categories, validate_submission


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"
METRICS_DIR = OUTPUTS_DIR / "metrics"

ID_COLUMN = "id"
TARGET = "addicted_label"
EXPERIMENT_ID = "EXP-010"

OOF_PATH = PREDICTIONS_DIR / "oof_exp010_xgboost_features.csv"
TEST_PREDICTIONS_PATH = PREDICTIONS_DIR / "test_exp010_xgboost_features.csv"
SUBMISSION_PATH = SUBMISSIONS_DIR / "submission_exp010_xgboost_features.csv"
METRICS_PATH = METRICS_DIR / "exp010_xgboost_features_metrics.txt"
LOG_PATH = METRICS_DIR / "experiment_log.csv"

REFERENCE_OOF_PATHS = {
    "EXP-008": PREDICTIONS_DIR / "oof_exp008_xgboost.csv",
    "EXP-003": PREDICTIONS_DIR / "oof_exp003_catboost.csv",
    "EXP-006": PREDICTIONS_DIR / "oof_exp006_lightgbm_features.csv",
}
EXP008_METRICS_PATH = METRICS_DIR / "exp008_xgboost_metrics.txt"


def parse_metrics(path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            parsed[key] = value
    return parsed


def load_reference_oof(
    current: pd.DataFrame, path: Path, experiment_id: str
) -> pd.DataFrame:
    reference = pd.read_csv(path)
    expected = [ID_COLUMN, "y_true", "oof_prediction"]
    if reference.columns.tolist() != expected:
        raise ValueError(f"Esquema OOF inesperado para {experiment_id}.")
    if reference.isna().any().any() or reference[ID_COLUMN].duplicated().any():
        raise ValueError(f"OOF inválidas para {experiment_id}.")
    aligned = current.merge(
        reference.rename(
            columns={"y_true": "reference_y_true", "oof_prediction": "reference_prediction"}
        ),
        on=ID_COLUMN,
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    if len(aligned) != len(current):
        raise ValueError(f"IDs OOF no coinciden con {experiment_id}.")
    if not aligned["y_true"].equals(aligned["reference_y_true"]):
        raise ValueError(f"y_true no coincide con {experiment_id}.")
    return aligned


def update_experiment_log(mean_auc: float) -> None:
    columns = [
        "experiment_id", "datetime", "model", "features", "cv_strategy",
        "cv_roc_auc", "kaggle_score", "notes",
    ]
    log = pd.read_csv(LOG_PATH, dtype=str, keep_default_na=False)
    if log.columns.tolist() != columns:
        raise ValueError(f"Encabezado inesperado en {LOG_PATH}: {log.columns.tolist()}")
    if (log["experiment_id"] == "EXP-009").sum() != 1:
        raise ValueError("Se esperaba exactamente una fila EXP-009.")
    log.loc[log["experiment_id"] == "EXP-009", "kaggle_score"] = "0.96601"
    log = log.loc[log["experiment_id"] != EXPERIMENT_ID].copy()
    row = pd.DataFrame([{
        "experiment_id": EXPERIMENT_ID,
        "datetime": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": "XGBoost",
        "features": "original_plus_engineered_v1",
        "cv_strategy": "StratifiedKFold_5",
        "cv_roc_auc": f"{mean_auc:.6f}",
        "kaggle_score": "",
        "notes": (
            "XGBoost baseline from EXP-008 plus same 8 engineered features used in EXP-006"
        ),
    }])
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
    X_raw = engineer_features(train[original_features])
    X_test_raw = engineer_features(test[original_features])
    y = train[TARGET]

    if not all(column in X_raw and column in X_test_raw for column in NEW_FEATURES):
        raise ValueError("No se generaron las ocho features en train y test.")
    if X_raw.columns.tolist() != X_test_raw.columns.tolist():
        raise ValueError("Train y test no tienen las mismas features finales.")
    if np.isinf(X_raw[NEW_FEATURES].to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("Hay infinitos en nuevas features de train.")
    if np.isinf(X_test_raw[NEW_FEATURES].to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("Hay infinitos en nuevas features de test.")

    categorical_columns = [
        c for c in original_features if not pd.api.types.is_numeric_dtype(X_raw[c])
    ]
    numeric_columns = [c for c in X_raw.columns if c not in categorical_columns]
    X, X_test, category_mappings = ordinal_encode_categories(
        X_raw, X_test_raw, categorical_columns
    )
    if not X[numeric_columns].equals(X_raw[numeric_columns]):
        raise ValueError("Las variables numéricas de train fueron modificadas.")
    if not X_test[numeric_columns].equals(X_test_raw[numeric_columns]):
        raise ValueError("Las variables numéricas de test fueron modificadas.")

    exp008_metrics = parse_metrics(EXP008_METRICS_PATH)
    exp008_best_iterations = [
        int(exp008_metrics[f"fold_{fold}_best_iteration_zero_based"]) for fold in range(1, 6)
    ]
    exp008_oof = pd.read_csv(REFERENCE_OOF_PATHS["EXP-008"])
    if exp008_oof.columns.tolist() != [ID_COLUMN, "y_true", "oof_prediction"]:
        raise ValueError("Esquema inesperado en OOF EXP-008.")
    if not exp008_oof[ID_COLUMN].equals(train[ID_COLUMN]) or not exp008_oof["y_true"].equals(y):
        raise ValueError("OOF EXP-008 no coincide en ID/target con train.")
    exp008_global_auc = float(roc_auc_score(y, exp008_oof["oof_prediction"]))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(cv.split(np.zeros(len(y)), y))
    exp008_fold_scores = [
        float(roc_auc_score(y.iloc[valid], exp008_oof["oof_prediction"].iloc[valid]))
        for _, valid in splits
    ]
    exp008_std = float(np.std(exp008_fold_scores))

    print(f"Experimento: {EXPERIMENT_ID}", flush=True)
    print(f"Nuevas features: {NEW_FEATURES}", flush=True)
    print(f"Mappings categóricos: {category_mappings}", flush=True)

    oof_predictions = np.zeros(len(train), dtype=np.float64)
    test_predictions = np.zeros(len(test), dtype=np.float64)
    fold_scores: list[float] = []
    best_iterations: list[int] = []
    fold_times: list[float] = []
    total_start = perf_counter()

    for fold, (train_indices, valid_indices) in enumerate(splits, start=1):
        fold_start = perf_counter()
        model = XGBClassifier(**MODEL_PARAMS, early_stopping_rounds=200)
        model.fit(
            X.iloc[train_indices],
            y.iloc[train_indices],
            eval_set=[(X.iloc[valid_indices], y.iloc[valid_indices])],
            verbose=False,
        )
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
    test_predictions = np.clip(test_predictions, 0.0, 1.0)
    mean_auc = float(np.mean(fold_scores))
    std_auc = float(np.std(fold_scores))
    overall_oof_auc = float(roc_auc_score(y, oof_predictions))
    global_difference = overall_oof_auc - exp008_global_auc
    fold_differences = [new - old for new, old in zip(fold_scores, exp008_fold_scores)]
    std_change = std_auc - exp008_std
    best_iteration_changes = [new - old for new, old in zip(best_iterations, exp008_best_iterations)]

    oof_frame = pd.DataFrame(
        {ID_COLUMN: train[ID_COLUMN], "y_true": y, "oof_prediction": oof_predictions}
    )
    test_frame = pd.DataFrame(
        {ID_COLUMN: sample[ID_COLUMN].copy(), "prediction": test_predictions}
    )
    if oof_frame.isna().any().any() or not oof_frame["oof_prediction"].between(0, 1).all():
        raise ValueError("Las OOF de EXP-010 no son válidas.")
    if test_frame.isna().any().any() or not test_frame["prediction"].between(0, 1).all():
        raise ValueError("Las predicciones test de EXP-010 no son válidas.")
    oof_frame.to_csv(OOF_PATH, index=False)
    test_frame.to_csv(TEST_PREDICTIONS_PATH, index=False)

    submission = test_frame.rename(columns={"prediction": TARGET})
    validate_submission(submission, test, sample)
    submission.to_csv(SUBMISSION_PATH, index=False)
    validate_submission(pd.read_csv(SUBMISSION_PATH), test, sample)

    correlations: dict[str, dict[str, float]] = {}
    for experiment_id, path in REFERENCE_OOF_PATHS.items():
        aligned = load_reference_oof(oof_frame, path, experiment_id)
        correlations[experiment_id] = {
            "pearson": float(aligned["oof_prediction"].corr(aligned["reference_prediction"], method="pearson")),
            "spearman": float(aligned["oof_prediction"].corr(aligned["reference_prediction"], method="spearman")),
        }

    if global_difference >= 0.00010:
        decision = "Mejora >= +0.00010 OOF: recomendar subir a Kaggle."
    elif global_difference >= 0.00003:
        decision = "Mejora entre +0.00003 y +0.00010: marginal pero potencialmente útil."
    elif global_difference >= 0:
        decision = "Mejora < +0.00003: prácticamente empate."
    else:
        decision = "Empeora: estas features no ayudan a XGBoost aunque ayudaron a LightGBM."

    parameter_text = ", ".join(f"{key}={value!r}" for key, value in MODEL_PARAMS.items())
    metrics_lines = [
        f"experiment_id: {EXPERIMENT_ID}",
        f"parameters: XGBClassifier({parameter_text}, early_stopping_rounds=200)",
        f"engineered_features: {NEW_FEATURES}",
        "categorical_treatment: identical stable ordinal encoding from EXP-008",
        "cv_strategy: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        *(f"fold_{i}_roc_auc: {score:.8f}" for i, score in enumerate(fold_scores, 1)),
        f"mean_roc_auc: {mean_auc:.8f}",
        f"std_roc_auc: {std_auc:.8f}",
        f"overall_oof_roc_auc: {overall_oof_auc:.8f}",
        *(f"fold_{i}_best_iteration_zero_based: {value}" for i, value in enumerate(best_iterations, 1)),
        *(f"fold_{i}_seconds: {value:.2f}" for i, value in enumerate(fold_times, 1)),
        f"total_seconds: {total_seconds:.2f}",
        f"exp008_overall_oof_auc: {exp008_global_auc:.8f}",
        f"global_difference_vs_exp008: {global_difference:+.8f}",
        *(f"fold_{i}_difference_vs_exp008: {value:+.8f}" for i, value in enumerate(fold_differences, 1)),
        f"exp008_std: {exp008_std:.8f}",
        f"std_change_vs_exp008: {std_change:+.8f}",
        *(f"fold_{i}_best_iteration_change_vs_exp008: {value:+d}" for i, value in enumerate(best_iteration_changes, 1)),
        *(f"pearson_vs_{name.lower().replace('-', '')}: {values['pearson']:.8f}" for name, values in correlations.items()),
        *(f"spearman_vs_{name.lower().replace('-', '')}: {values['spearman']:.8f}" for name, values in correlations.items()),
        f"decision: {decision}",
    ]
    METRICS_PATH.write_text("\n".join(metrics_lines) + "\n", encoding="utf-8")
    update_experiment_log(mean_auc)

    print(f"Media ROC AUC: {mean_auc:.8f}", flush=True)
    print(f"Std ROC AUC: {std_auc:.8f}", flush=True)
    print(f"Best iterations: {best_iterations}", flush=True)
    print(f"Tiempo total: {total_seconds:.2f} s", flush=True)
    print(f"Diferencia global vs EXP-008: {global_difference:+.8f}", flush=True)
    print(f"Diferencias fold: {fold_differences}", flush=True)
    print(f"Correlaciones: {correlations}", flush=True)
    print(f"Decisión: {decision}", flush=True)
    print(f"Submission: {SUBMISSION_PATH}", flush=True)
    print("Validaciones del submission: OK", flush=True)


if __name__ == "__main__":
    main()
