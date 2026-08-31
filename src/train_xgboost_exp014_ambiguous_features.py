"""EXP-014: EXP-012 plus targeted ambiguous-zone context features."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from train_xgboost_exp008 import MODEL_PARAMS, ordinal_encode_categories, validate_submission
from train_xgboost_exp012_threshold_features import (
    NEW_FEATURES as THRESHOLD_FEATURES,
    add_threshold_features,
    safe_divide,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"
METRICS_DIR = OUTPUTS_DIR / "metrics"

ID_COLUMN = "id"
TARGET = "addicted_label"
EXPERIMENT_ID = "EXP-014"

OOF_PATH = PREDICTIONS_DIR / "oof_exp014_xgboost_ambiguous.csv"
TEST_PREDICTIONS_PATH = PREDICTIONS_DIR / "test_exp014_xgboost_ambiguous.csv"
SUBMISSION_PATH = SUBMISSIONS_DIR / "submission_exp014_xgboost_ambiguous.csv"
METRICS_PATH = METRICS_DIR / "exp014_xgboost_ambiguous_metrics.txt"
LOG_PATH = METRICS_DIR / "experiment_log.csv"
EXP012_METRICS_PATH = METRICS_DIR / "exp012_xgboost_thresholds_metrics.txt"

REFERENCE_OOF_PATHS = {
    "EXP-012": PREDICTIONS_DIR / "oof_exp012_xgboost_thresholds.csv",
    "EXP-003": PREDICTIONS_DIR / "oof_exp003_catboost.csv",
    "EXP-006": PREDICTIONS_DIR / "oof_exp006_lightgbm_features.csv",
    "EXP-008": PREDICTIONS_DIR / "oof_exp008_xgboost.csv",
    "EXP-010": PREDICTIONS_DIR / "oof_exp010_xgboost_features.csv",
}

AMBIGUOUS_FEATURES = [
    "screen_band_position",
    "screen_dist_to_7", "screen_abs_dist_to_7",
    "social_margin_from_4",
    "weekend_minus_daily", "weekend_over_daily", "daily_over_weekend",
    "screen_plus_social_minus_sleep", "screen_sleep_gap", "total_screen_over_sleep",
    "notifications_per_app_open", "app_opens_per_notification", "social_per_app_open",
    "screen_plus_social_minus_work", "work_to_screen_ratio",
    "ambig_daily_screen", "ambig_social_media", "ambig_weekend_screen",
    "ambig_sleep", "ambig_notifications", "ambig_app_opens", "ambig_work_study",
    "ambig_screen_x_social", "ambig_weekend_minus_daily", "ambig_screen_sleep_gap",
]


def add_ambiguous_features(frame: pd.DataFrame) -> pd.DataFrame:
    engineered = frame.copy()
    screen = engineered["daily_screen_time_hours"]
    social = engineered["social_media_hours"]
    weekend = engineered["weekend_screen_time"]
    sleep = engineered["sleep_hours"]
    notifications = engineered["notifications_per_day"]
    app_opens = engineered["app_opens_per_day"]
    work = engineered["work_study_hours"]

    engineered["screen_band_position"] = (screen - 6) / 2
    engineered["screen_dist_to_7"] = screen - 7
    engineered["screen_abs_dist_to_7"] = (screen - 7).abs()
    engineered["social_margin_from_4"] = social - 4
    engineered["weekend_minus_daily"] = weekend - screen
    engineered["weekend_over_daily"] = safe_divide(weekend, screen)
    engineered["daily_over_weekend"] = safe_divide(screen, weekend)
    engineered["screen_plus_social_minus_sleep"] = screen + social - sleep
    engineered["screen_sleep_gap"] = screen - sleep
    engineered["total_screen_over_sleep"] = safe_divide(screen + social, sleep)
    engineered["notifications_per_app_open"] = safe_divide(notifications, app_opens)
    engineered["app_opens_per_notification"] = safe_divide(app_opens, notifications)
    engineered["social_per_app_open"] = safe_divide(social, app_opens)
    engineered["screen_plus_social_minus_work"] = screen + social - work
    engineered["work_to_screen_ratio"] = safe_divide(work, screen)

    ambiguous_mask = engineered["ambiguous_zone"].eq(1)
    engineered["ambig_daily_screen"] = screen.where(ambiguous_mask)
    engineered["ambig_social_media"] = social.where(ambiguous_mask)
    engineered["ambig_weekend_screen"] = weekend.where(ambiguous_mask)
    engineered["ambig_sleep"] = sleep.where(ambiguous_mask)
    engineered["ambig_notifications"] = notifications.where(ambiguous_mask)
    engineered["ambig_app_opens"] = app_opens.where(ambiguous_mask)
    engineered["ambig_work_study"] = work.where(ambiguous_mask)
    engineered["ambig_screen_x_social"] = (screen * social).where(ambiguous_mask)
    engineered["ambig_weekend_minus_daily"] = (weekend - screen).where(ambiguous_mask)
    engineered["ambig_screen_sleep_gap"] = (screen - sleep).where(ambiguous_mask)
    return engineered


def parse_metrics(path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            parsed[key] = value
    return parsed


def align_reference(current: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
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
    return aligned


def region_auc(
    name: str,
    mask: pd.Series,
    y_true: pd.Series,
    exp012_prediction: pd.Series,
    exp014_prediction: pd.Series,
) -> dict[str, object]:
    region_y = y_true.loc[mask]
    result: dict[str, object] = {
        "region": name,
        "count": int(mask.sum()),
        "proportion": float(mask.mean()),
        "class_count": int(region_y.nunique()),
    }
    if region_y.nunique() < 2:
        result.update(
            {"exp012_auc": np.nan, "exp014_auc": np.nan, "difference": np.nan,
             "note": "AUC no calculable: la región contiene una sola clase."}
        )
    else:
        exp012_auc = float(roc_auc_score(region_y, exp012_prediction.loc[mask]))
        exp014_auc = float(roc_auc_score(region_y, exp014_prediction.loc[mask]))
        result.update(
            {"exp012_auc": exp012_auc, "exp014_auc": exp014_auc,
             "difference": exp014_auc - exp012_auc, "note": "OK"}
        )
    return result


def update_log(mean_auc: float) -> None:
    columns = [
        "experiment_id", "datetime", "model", "features", "cv_strategy",
        "cv_roc_auc", "kaggle_score", "notes",
    ]
    log = pd.read_csv(LOG_PATH, dtype=str, keep_default_na=False)
    if log.columns.tolist() != columns or (log["experiment_id"] == "EXP-013").sum() != 1:
        raise ValueError("Estado inesperado de experiment_log.csv.")
    log.loc[log["experiment_id"] == "EXP-013", "kaggle_score"] = "0.96688"
    log = log.loc[log["experiment_id"] != EXPERIMENT_ID].copy()
    row = pd.DataFrame([{
        "experiment_id": EXPERIMENT_ID,
        "datetime": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": "XGBoost",
        "features": "exp012_thresholds_plus_ambiguous_context",
        "cv_strategy": "StratifiedKFold_5",
        "cv_roc_auc": f"{mean_auc:.6f}",
        "kaggle_score": "",
        "notes": "EXP-012 threshold features plus targeted ambiguous-zone context features",
    }])
    pd.concat([log, row], ignore_index=True).to_csv(LOG_PATH, index=False)


def main() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    original_features = [column for column in train.columns if column not in {ID_COLUMN, TARGET}]
    if test.columns.tolist() != [ID_COLUMN, *original_features]:
        raise ValueError("Las features originales de test no coinciden con train.")

    exp012_train = add_threshold_features(train[original_features])
    exp012_test = add_threshold_features(test[original_features])
    X_raw = add_ambiguous_features(exp012_train)
    X_test_raw = add_ambiguous_features(exp012_test)
    y = train[TARGET]

    if X_raw.columns.tolist() != X_test_raw.columns.tolist():
        raise ValueError("Train y test no tienen el mismo esquema final.")
    if not all(feature in X_raw and feature in X_test_raw for feature in AMBIGUOUS_FEATURES):
        raise ValueError("No se generaron todas las nuevas features ambiguas.")
    if not X_raw[exp012_train.columns].equals(exp012_train):
        raise ValueError("Se modificaron features existentes de EXP-012 en train.")
    if not X_test_raw[exp012_test.columns].equals(exp012_test):
        raise ValueError("Se modificaron features existentes de EXP-012 en test.")
    if np.isinf(X_raw[AMBIGUOUS_FEATURES].to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("Hay infinitos en nuevas features de train.")
    if np.isinf(X_test_raw[AMBIGUOUS_FEATURES].to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("Hay infinitos en nuevas features de test.")

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

    exp012_oof = pd.read_csv(REFERENCE_OOF_PATHS["EXP-012"])
    if not exp012_oof[ID_COLUMN].equals(train[ID_COLUMN]) or not exp012_oof["y_true"].equals(y):
        raise ValueError("OOF EXP-012 no coincide con train.")
    exp012_global_auc = float(roc_auc_score(y, exp012_oof["oof_prediction"]))
    exp012_metrics = parse_metrics(EXP012_METRICS_PATH)
    exp012_best_iterations = [
        int(exp012_metrics[f"fold_{fold}_best_iteration_zero_based"]) for fold in range(1, 6)
    ]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(cv.split(np.zeros(len(y)), y))
    exp012_fold_scores = [
        float(roc_auc_score(y.iloc[valid], exp012_oof["oof_prediction"].iloc[valid]))
        for _, valid in splits
    ]
    exp012_std = float(np.std(exp012_fold_scores))

    print(f"Experimento: {EXPERIMENT_ID}", flush=True)
    print(f"Features ambiguas nuevas ({len(AMBIGUOUS_FEATURES)}): {AMBIGUOUS_FEATURES}", flush=True)
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
        valid_prediction = model.predict_proba(
            X.iloc[valid_indices], iteration_range=iteration_range
        )[:, 1]
        oof_predictions[valid_indices] = valid_prediction
        fold_test = model.predict_proba(X_test, iteration_range=iteration_range)[:, 1].astype(np.float64)
        test_predictions += fold_test / 5.0
        score = float(roc_auc_score(y.iloc[valid_indices], valid_prediction))

        raw_gain = model.get_booster().get_score(importance_type="gain")
        total_gain = sum(raw_gain.values())
        fold_importances.append({
            feature: (raw_gain.get(feature, 0.0) / total_gain if total_gain else 0.0)
            for feature in X.columns
        })
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
    global_difference = overall_oof_auc - exp012_global_auc
    fold_differences = [new - old for new, old in zip(fold_scores, exp012_fold_scores)]
    std_change = std_auc - exp012_std
    best_iteration_changes = [new - old for new, old in zip(best_iterations, exp012_best_iterations)]

    oof_frame = pd.DataFrame(
        {ID_COLUMN: train[ID_COLUMN], "y_true": y, "oof_prediction": oof_predictions}
    )
    test_frame = pd.DataFrame(
        {ID_COLUMN: sample[ID_COLUMN].copy(), "prediction": test_predictions}
    )
    if oof_frame.isna().any().any() or not oof_frame["oof_prediction"].between(0, 1).all():
        raise ValueError("Las OOF de EXP-014 no son válidas.")
    if test_frame.isna().any().any() or not test_frame["prediction"].between(0, 1).all():
        raise ValueError("Las predicciones test de EXP-014 no son válidas.")
    oof_frame.to_csv(OOF_PATH, index=False)
    test_frame.to_csv(TEST_PREDICTIONS_PATH, index=False)
    submission = test_frame.rename(columns={"prediction": TARGET})
    validate_submission(submission, test, sample)
    submission.to_csv(SUBMISSION_PATH, index=False)
    validate_submission(pd.read_csv(SUBMISSION_PATH), test, sample)

    region_rows = [
        region_auc(
            "clear_positive_zone", exp012_train["clear_positive_zone"].eq(1), y,
            exp012_oof["oof_prediction"], oof_frame["oof_prediction"],
        ),
        region_auc(
            "clear_negative_zone", exp012_train["clear_negative_zone"].eq(1), y,
            exp012_oof["oof_prediction"], oof_frame["oof_prediction"],
        ),
        region_auc(
            "ambiguous_zone", exp012_train["ambiguous_zone"].eq(1), y,
            exp012_oof["oof_prediction"], oof_frame["oof_prediction"],
        ),
    ]
    regions = pd.DataFrame(region_rows)
    ambiguous_difference = float(
        regions.loc[regions["region"] == "ambiguous_zone", "difference"].iloc[0]
    )

    importance = pd.DataFrame(fold_importances).mean(axis=0).sort_values(ascending=False)
    top30 = importance.head(30)
    feature_groups = {
        "original_features": original_features,
        "exp012_threshold_features": THRESHOLD_FEATURES,
        "new_ambiguous_features": AMBIGUOUS_FEATURES,
    }
    group_importance = {
        group: float(importance.reindex(features, fill_value=0.0).sum())
        for group, features in feature_groups.items()
    }

    correlations: dict[str, dict[str, float]] = {}
    for experiment_id in REFERENCE_OOF_PATHS:
        aligned = align_reference(oof_frame, experiment_id)
        correlations[experiment_id] = {
            "pearson": float(aligned["oof_prediction"].corr(aligned["reference_prediction"], method="pearson")),
            "spearman": float(aligned["oof_prediction"].corr(aligned["reference_prediction"], method="spearman")),
        }

    improved_folds = sum(value > 0 for value in fold_differences)
    if global_difference >= 0.00010:
        decision = "Mejora >= +0.00010: recomendar subir a Kaggle."
    elif global_difference >= 0.00003:
        if improved_folds >= 4 and ambiguous_difference > 0:
            decision = (
                "Mejora marginal interesante; recomendar submission porque mejora >=4 folds "
                "y ambiguous_zone."
            )
        else:
            decision = (
                "Mejora marginal, pero no cumple simultáneamente >=4 folds y mejora ambigua; "
                "no recomendar submission."
            )
    elif global_difference >= 0:
        decision = "Mejora < +0.00003: empate práctico."
    else:
        decision = "Empeora: este bloque de contexto ambiguo no aporta."

    parameter_text = ", ".join(f"{key}={value!r}" for key, value in MODEL_PARAMS.items())
    top30_text = "\n".join(
        f"{rank}. {feature}: {gain:.8f}"
        for rank, (feature, gain) in enumerate(top30.items(), start=1)
    )
    metrics_lines = [
        f"experiment_id: {EXPERIMENT_ID}",
        f"parameters: XGBClassifier({parameter_text}, early_stopping_rounds=200)",
        f"threshold_features_reused_unchanged: {THRESHOLD_FEATURES}",
        f"new_ambiguous_features: {AMBIGUOUS_FEATURES}",
        "cv_strategy: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        *(f"fold_{fold}_roc_auc: {score:.8f}" for fold, score in enumerate(fold_scores, 1)),
        f"mean_roc_auc: {mean_auc:.8f}",
        f"std_roc_auc: {std_auc:.8f}",
        f"overall_oof_roc_auc: {overall_oof_auc:.8f}",
        *(f"fold_{fold}_best_iteration_zero_based: {value}" for fold, value in enumerate(best_iterations, 1)),
        *(f"fold_{fold}_seconds: {value:.2f}" for fold, value in enumerate(fold_times, 1)),
        f"total_seconds: {total_seconds:.2f}",
        f"exp012_overall_oof_auc: {exp012_global_auc:.8f}",
        f"global_difference_vs_exp012: {global_difference:+.8f}",
        f"fold_differences_vs_exp012: {fold_differences}",
        f"std_change_vs_exp012: {std_change:+.8f}",
        f"best_iteration_changes_vs_exp012: {best_iteration_changes}",
        "regional_auc:",
        regions.to_string(index=False, float_format=lambda value: f"{value:.8f}"),
        "top30_mean_normalized_gain:", top30_text,
        f"group_importance: {group_importance}",
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
    print(f"Diferencia vs EXP-012: {global_difference:+.8f}", flush=True)
    print("Regiones:\n" + regions.to_string(index=False), flush=True)
    print("Top 30 gain:\n" + top30_text, flush=True)
    print(f"Importancia por grupos: {group_importance}", flush=True)
    print(f"Correlaciones: {correlations}", flush=True)
    print(f"Decisión: {decision}", flush=True)
    print(f"Submission: {SUBMISSION_PATH}", flush=True)
    print("Validaciones del submission: OK", flush=True)


if __name__ == "__main__":
    main()
