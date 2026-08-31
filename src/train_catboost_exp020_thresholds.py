"""EXP-020: CatBoost using exact EXP-012/016 threshold-region features."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from train_catboost_exp003 import prepare_categorical_missing, validate_submission
from train_xgboost_exp012_threshold_features import NEW_FEATURES, add_threshold_features


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
PREDICTIONS = OUTPUTS / "predictions"
SUBMISSIONS = OUTPUTS / "submissions"
METRICS = OUTPUTS / "metrics"
ID = "id"
TARGET = "addicted_label"
EXPERIMENT = "EXP-020"

OOF_PATH = PREDICTIONS / "oof_exp020_catboost_thresholds.csv"
TEST_PATH = PREDICTIONS / "test_exp020_catboost_thresholds.csv"
SUBMISSION_PATH = SUBMISSIONS / "submission_exp020_catboost_thresholds.csv"
METRICS_PATH = METRICS / "exp020_catboost_thresholds_metrics.txt"
LOG_PATH = METRICS / "experiment_log.csv"
REFERENCE_PATHS = {
    "EXP-016": PREDICTIONS / "oof_exp016_xgboost_depth5_9000.csv",
    "EXP-015": PREDICTIONS / "oof_exp015_xgboost_depth5.csv",
    "EXP-012": PREDICTIONS / "oof_exp012_xgboost_thresholds.csv",
    "EXP-003": PREDICTIONS / "oof_exp003_catboost.csv",
    "EXP-006": PREDICTIONS / "oof_exp006_lightgbm_features.csv",
    "EXP-019": PREDICTIONS / "oof_exp019_lightgbm_thresholds.csv",
}
MODEL_PARAMS = {
    "iterations": 6500, "learning_rate": 0.05, "depth": 7,
    "loss_function": "Logloss", "eval_metric": "AUC",
    "random_seed": 42, "verbose": False, "allow_writing_files": False,
}
EARLY_STOPPING = 250
EXP016_FOLDS = np.array([0.9650202929179754, 0.9656644620983974, 0.9658836874972772,
                         0.966455024819542, 0.9654874019318932])


def load_reference(current: pd.DataFrame, name: str) -> pd.DataFrame:
    reference = pd.read_csv(REFERENCE_PATHS[name])
    if reference.columns.tolist() != [ID, "y_true", "oof_prediction"]:
        raise ValueError(f"Esquema OOF inesperado para {name}.")
    if reference.isna().any().any() or reference[ID].duplicated().any():
        raise ValueError(f"OOF invalida para {name}.")
    aligned = current.merge(
        reference.rename(columns={"y_true": "reference_y", "oof_prediction": "reference_prediction"}),
        on=ID, how="inner", validate="one_to_one", sort=False,
    )
    if len(aligned) != len(current) or not aligned["y_true"].equals(aligned["reference_y"]):
        raise ValueError(f"IDs o y_true no coinciden con {name}.")
    return aligned


def update_log(mean_auc: float) -> None:
    columns = [
        "experiment_id", "datetime", "model", "features", "cv_strategy",
        "cv_roc_auc", "kaggle_score", "notes",
    ]
    log = pd.read_csv(LOG_PATH, dtype=str, keep_default_na=False)
    if log.columns.tolist() != columns:
        raise ValueError("Encabezado inesperado en experiment_log.csv.")
    log = log.loc[log["experiment_id"] != EXPERIMENT].copy()
    row = pd.DataFrame([{
        "experiment_id": EXPERIMENT, "datetime": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": "CatBoostClassifier", "features": "exp012_threshold_region_features",
        "cv_strategy": "StratifiedKFold_5", "cv_roc_auc": f"{mean_auc:.6f}",
        "kaggle_score": "",
        "notes": "CatBoost using exact threshold/region features from EXP-012/016",
    }])
    pd.concat([log, row], ignore_index=True).to_csv(LOG_PATH, index=False)


def main() -> None:
    total_start = perf_counter()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    originals = [column for column in train.columns if column not in {ID, TARGET}]
    if test.columns.tolist() != [ID, *originals]:
        raise ValueError("Esquema original train/test no coincide.")
    raw = add_threshold_features(train[originals])
    raw_test = add_threshold_features(test[originals])
    if raw.columns.tolist() != raw_test.columns.tolist():
        raise ValueError("Esquema final train/test no coincide.")
    if not all(feature in raw and feature in raw_test for feature in NEW_FEATURES):
        raise ValueError("Faltan threshold features de EXP-012/016.")
    if len(raw.columns) != len(originals) + len(NEW_FEATURES):
        raise ValueError("Se agregaron o quitaron features respecto a EXP-016.")
    categorical_columns = [
        column for column in originals if not pd.api.types.is_numeric_dtype(raw[column])
    ]
    numeric_columns = [column for column in raw if column not in categorical_columns]
    X = prepare_categorical_missing(raw, categorical_columns)
    X_test = prepare_categorical_missing(raw_test, categorical_columns)
    if X[categorical_columns].isna().any().any() or X_test[categorical_columns].isna().any().any():
        raise ValueError("Missing categorico residual.")
    if not X[numeric_columns].equals(raw[numeric_columns]):
        raise ValueError("Las variables numericas de train fueron modificadas.")
    if not X_test[numeric_columns].equals(raw_test[numeric_columns]):
        raise ValueError("Las variables numericas de test fueron modificadas.")
    y = train[TARGET]
    splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, y))

    print(f"{EXPERIMENT}: rows={len(train)}, features={X.shape[1]}", flush=True)
    print(f"Threshold features ({len(NEW_FEATURES)}): {NEW_FEATURES}", flush=True)
    print(f"Categoricals={categorical_columns}; params={MODEL_PARAMS}; early={EARLY_STOPPING}", flush=True)
    oof = np.zeros(len(train), dtype=np.float64)
    test_prediction = np.zeros(len(test), dtype=np.float64)
    scores: list[float] = []
    best_iterations: list[int] = []
    fold_seconds: list[float] = []
    importances: list[dict[str, float]] = []

    for fold, (train_idx, valid_idx) in enumerate(splits, 1):
        fold_start = perf_counter()
        model = CatBoostClassifier(**MODEL_PARAMS)
        model.fit(
            X.iloc[train_idx], y.iloc[train_idx], cat_features=categorical_columns,
            eval_set=(X.iloc[valid_idx], y.iloc[valid_idx]),
            early_stopping_rounds=EARLY_STOPPING, verbose=False,
        )
        best = int(model.get_best_iteration())
        valid_prediction = model.predict_proba(X.iloc[valid_idx])[:, 1]
        oof[valid_idx] = valid_prediction.astype(np.float64)
        test_prediction += model.predict_proba(X_test)[:, 1].astype(np.float64) / 5
        score = float(roc_auc_score(y.iloc[valid_idx], valid_prediction))
        raw_importance = model.get_feature_importance(type="FeatureImportance")
        total_importance = float(raw_importance.sum())
        importances.append({
            feature: float(value / total_importance) if total_importance else 0.0
            for feature, value in zip(X.columns, raw_importance)
        })
        elapsed = perf_counter() - fold_start
        scores.append(score)
        best_iterations.append(best)
        fold_seconds.append(elapsed)
        print(f"Fold {fold}: AUC={score:.8f} best={best} time={elapsed:.2f}s", flush=True)

    total_seconds = perf_counter() - total_start
    mean_auc = float(np.mean(scores))
    std_auc = float(np.std(scores))
    overall_auc = float(roc_auc_score(y, oof))
    test_prediction = np.clip(test_prediction, 0, 1)
    oof_frame = pd.DataFrame({ID: train[ID], "y_true": y, "oof_prediction": oof})
    test_frame = pd.DataFrame({ID: sample[ID], "prediction": test_prediction})
    submission = test_frame.rename(columns={"prediction": TARGET})
    if oof_frame.isna().any().any() or not oof_frame["oof_prediction"].between(0, 1).all():
        raise ValueError("OOF EXP-020 invalida.")
    validate_submission(submission, test, sample)

    correlations: dict[str, dict[str, float]] = {}
    reference_aucs: dict[str, float] = {}
    for name in REFERENCE_PATHS:
        aligned = load_reference(oof_frame, name)
        reference_aucs[name] = float(roc_auc_score(aligned["y_true"], aligned["reference_prediction"]))
        correlations[name] = {
            "pearson": float(aligned["oof_prediction"].corr(aligned["reference_prediction"], method="pearson")),
            "spearman": float(aligned["oof_prediction"].corr(aligned["reference_prediction"], method="spearman")),
        }
    delta_vs_exp003 = overall_auc - reference_aucs["EXP-003"]
    delta_vs_exp016 = overall_auc - reference_aucs["EXP-016"]
    fold_deltas_vs_exp016 = (np.asarray(scores) - EXP016_FOLDS).tolist()

    exp016 = pd.read_csv(REFERENCE_PATHS["EXP-016"])
    if not exp016[ID].equals(train[ID]) or not exp016["y_true"].equals(y):
        raise ValueError("OOF EXP-016 no coincide con train.")
    regions: dict[str, dict[str, float | int | str]] = {}
    for region_column in ("clear_positive_zone", "clear_negative_zone", "ambiguous_zone"):
        mask = raw[region_column].eq(1)
        region_y = y.loc[mask]
        if region_y.nunique() < 2:
            regions[region_column] = {"rows": int(mask.sum()), "note": "single class; AUC unavailable"}
        else:
            exp020_auc = float(roc_auc_score(region_y, oof_frame.loc[mask, "oof_prediction"]))
            exp016_auc = float(roc_auc_score(region_y, exp016.loc[mask, "oof_prediction"]))
            regions[region_column] = {
                "rows": int(mask.sum()), "exp020_auc": exp020_auc,
                "exp016_auc": exp016_auc, "difference": exp020_auc - exp016_auc,
            }

    top25 = pd.DataFrame(importances).mean().sort_values(ascending=False).head(25)
    pearson_exp016 = correlations["EXP-016"]["pearson"]
    if overall_auc >= .96510:
        individual_recommendation = "Recomendar submission individual."
    elif overall_auc >= .96480:
        individual_recommendation = "Submission opcional; priorizar diversidad/ensemble."
    else:
        individual_recommendation = "No recomendar submission individual salvo diversidad extraordinaria."
    ensemble_recommendation = (
        "Recomendar proximo ensemble OOF EXP-016 + EXP-020."
        if overall_auc >= .96490 and pearson_exp016 <= .9935
        else "No cumple simultaneamente OOF >= 0.96490 y Pearson <= 0.9935."
    )
    if overall_auc >= .96500 and pearson_exp016 <= .9935:
        diversity_assessment = "Especialmente prometedor para ensemble."
    elif pearson_exp016 < .9935:
        diversity_assessment = "Diversidad alta, pero evaluar frente al rendimiento individual insuficiente."
    else:
        diversity_assessment = "Diversidad limitada frente a EXP-016."
    iteration_assessment = (
        "Al menos 4/5 folds >6200: CatBoost puede seguir limitado por iterations."
        if sum(value > 6200 for value in best_iterations) >= 4
        else "Los folds convergen antes de 6200 en su mayoria: 6500 es suficiente."
    )

    for directory in (PREDICTIONS, SUBMISSIONS, METRICS):
        directory.mkdir(parents=True, exist_ok=True)
    oof_frame.to_csv(OOF_PATH, index=False)
    test_frame.to_csv(TEST_PATH, index=False)
    submission.to_csv(SUBMISSION_PATH, index=False)
    validate_submission(pd.read_csv(SUBMISSION_PATH), test, sample)
    top25_text = "\n".join(
        f"{rank}. {feature}: {importance:.8f}"
        for rank, (feature, importance) in enumerate(top25.items(), 1)
    )
    lines = [
        f"experiment_id: {EXPERIMENT}", f"parameters: {MODEL_PARAMS}; early_stopping=250",
        f"exact_reused_threshold_features: {NEW_FEATURES}",
        f"categorical_features: {categorical_columns}",
        "categorical_treatment: __MISSING__ strings; CatBoost native cat_features",
        *(f"fold_{i}_roc_auc: {value:.8f}" for i, value in enumerate(scores, 1)),
        f"mean_roc_auc: {mean_auc:.8f}", f"std_roc_auc: {std_auc:.8f}",
        f"overall_oof_roc_auc: {overall_auc:.8f}",
        *(f"fold_{i}_best_iteration_zero_based: {value}" for i, value in enumerate(best_iterations, 1)),
        *(f"fold_{i}_seconds: {value:.2f}" for i, value in enumerate(fold_seconds, 1)),
        f"total_seconds: {total_seconds:.2f}",
        f"delta_vs_exp003_oof: {delta_vs_exp003:+.8f}",
        f"delta_vs_exp016_oof: {delta_vs_exp016:+.8f}",
        f"fold_deltas_vs_exp016: {fold_deltas_vs_exp016}",
        f"correlations: {correlations}", f"regional_comparison: {regions}",
        "top25_mean_normalized_feature_importance:", top25_text,
        f"iteration_assessment: {iteration_assessment}",
        f"diversity_assessment: {diversity_assessment}",
        f"individual_recommendation: {individual_recommendation}",
        f"ensemble_recommendation: {ensemble_recommendation}",
        "submission_validations: OK", "problems: none",
    ]
    METRICS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    update_log(mean_auc)

    print(f"Scores={scores}; mean={mean_auc:.8f}; std={std_auc:.8f}; overall={overall_auc:.8f}", flush=True)
    print(f"Best={best_iterations}; time={total_seconds:.2f}s; {iteration_assessment}", flush=True)
    print(f"Delta EXP-003={delta_vs_exp003:+.8f}; EXP-016={delta_vs_exp016:+.8f}", flush=True)
    print(f"Fold deltas EXP-016={fold_deltas_vs_exp016}", flush=True)
    print(f"Correlations={correlations}", flush=True)
    print(f"Regions={regions}", flush=True)
    print("Top25:\n" + top25_text, flush=True)
    print(f"Diversity={diversity_assessment}", flush=True)
    print(f"Individual={individual_recommendation}", flush=True)
    print(f"Ensemble={ensemble_recommendation}", flush=True)
    print(f"Submission={SUBMISSION_PATH}; validations=OK", flush=True)


if __name__ == "__main__":
    main()
