"""EXP-005: OOF-only linear blend of EXP-003 CatBoost and EXP-004 LightGBM."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"
METRICS_DIR = OUTPUTS_DIR / "metrics"
DATA_DIR = PROJECT_ROOT / "data"

CAT_OOF_PATH = PREDICTIONS_DIR / "oof_exp003_catboost.csv"
LGBM_OOF_PATH = PREDICTIONS_DIR / "oof_exp004_lightgbm.csv"
CAT_TEST_PATH = SUBMISSIONS_DIR / "submission_exp003_catboost_4000.csv"
LGBM_TEST_PATH = PREDICTIONS_DIR / "test_exp004_lightgbm.csv"
SAMPLE_PATH = DATA_DIR / "sample_submission.csv"
SUBMISSION_PATH = SUBMISSIONS_DIR / "submission_exp005_blend_cat_lgbm.csv"
METRICS_PATH = METRICS_DIR / "exp005_blend_metrics.txt"
LOG_PATH = METRICS_DIR / "experiment_log.csv"

ID_COLUMN = "id"
TARGET = "addicted_label"
IMPROVEMENT_THRESHOLD = 0.00005


def load_and_align_oof() -> pd.DataFrame:
    cat = pd.read_csv(CAT_OOF_PATH)
    lgbm = pd.read_csv(LGBM_OOF_PATH)
    expected = [ID_COLUMN, "y_true", "oof_prediction"]
    if cat.columns.tolist() != expected or lgbm.columns.tolist() != expected:
        raise ValueError(
            f"Esquema OOF inesperado: CatBoost={cat.columns.tolist()}, "
            f"LightGBM={lgbm.columns.tolist()}"
        )
    if len(cat) != len(lgbm):
        raise ValueError(f"Cantidad OOF distinta: CatBoost={len(cat)}, LightGBM={len(lgbm)}")
    if cat[ID_COLUMN].duplicated().any() or lgbm[ID_COLUMN].duplicated().any():
        raise ValueError("Hay IDs duplicados en las OOF.")
    if cat.isna().any().any() or lgbm.isna().any().any():
        raise ValueError("Hay NaN en las OOF.")

    aligned = cat.rename(
        columns={"y_true": "y_true_cat", "oof_prediction": "catboost_prediction"}
    ).merge(
        lgbm.rename(
            columns={"y_true": "y_true_lgbm", "oof_prediction": "lightgbm_prediction"}
        ),
        on=ID_COLUMN,
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    if len(aligned) != len(cat):
        raise ValueError("Los conjuntos de IDs OOF no coinciden.")
    if not aligned["y_true_cat"].equals(aligned["y_true_lgbm"]):
        raise ValueError("Los valores y_true no coinciden después de alinear por ID.")
    return aligned.rename(columns={"y_true_cat": "y_true"}).drop(columns="y_true_lgbm")


def score_weights(aligned: pd.DataFrame, weights: list[float]) -> pd.DataFrame:
    y_true = aligned["y_true"].to_numpy()
    cat = aligned["catboost_prediction"].to_numpy()
    lgbm = aligned["lightgbm_prediction"].to_numpy()
    rows = []
    for weight in weights:
        prediction = weight * cat + (1.0 - weight) * lgbm
        rows.append(
            {
                "weight_catboost": weight,
                "weight_lightgbm": 1.0 - weight,
                "oof_roc_auc": roc_auc_score(y_true, prediction),
            }
        )
    return pd.DataFrame(rows)


def validate_submission(submission: pd.DataFrame, sample: pd.DataFrame) -> None:
    if submission.columns.tolist() != [ID_COLUMN, TARGET]:
        raise ValueError(f"Columnas inválidas: {submission.columns.tolist()}")
    if len(submission) != 296_302 or len(submission) != len(sample):
        raise ValueError(f"Cantidad de filas inválida: {len(submission)}")
    if submission.isna().any().any():
        raise ValueError("El submission contiene NaN.")
    if not submission[TARGET].between(0, 1, inclusive="both").all():
        raise ValueError("El submission contiene probabilidades fuera de [0, 1].")
    if not submission[ID_COLUMN].equals(sample[ID_COLUMN]):
        raise ValueError("Los IDs no coinciden exactamente con sample_submission.")


def create_submission(weight_catboost: float) -> None:
    sample = pd.read_csv(SAMPLE_PATH)
    cat = pd.read_csv(CAT_TEST_PATH).rename(columns={TARGET: "catboost_prediction"})
    lgbm = pd.read_csv(LGBM_TEST_PATH).rename(columns={"prediction": "lightgbm_prediction"})
    if cat.columns.tolist() != [ID_COLUMN, "catboost_prediction"]:
        raise ValueError(f"Esquema inesperado en test CatBoost: {cat.columns.tolist()}")
    if lgbm.columns.tolist() != [ID_COLUMN, "lightgbm_prediction"]:
        raise ValueError(f"Esquema inesperado en test LightGBM: {lgbm.columns.tolist()}")
    if cat[ID_COLUMN].duplicated().any() or lgbm[ID_COLUMN].duplicated().any():
        raise ValueError("Hay IDs duplicados en predicciones de test.")
    aligned_test = sample[[ID_COLUMN]].merge(cat, on=ID_COLUMN, how="left", validate="one_to_one")
    aligned_test = aligned_test.merge(lgbm, on=ID_COLUMN, how="left", validate="one_to_one")
    if aligned_test.isna().any().any():
        raise ValueError("Faltan predicciones al alinear test por ID.")
    submission = pd.DataFrame(
        {
            ID_COLUMN: sample[ID_COLUMN].copy(),
            TARGET: (
                weight_catboost * aligned_test["catboost_prediction"]
                + (1.0 - weight_catboost) * aligned_test["lightgbm_prediction"]
            ),
        }
    )
    validate_submission(submission, sample)
    submission.to_csv(SUBMISSION_PATH, index=False)
    validate_submission(pd.read_csv(SUBMISSION_PATH), sample)


def update_experiment_log(best_auc: float) -> None:
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
    log = log.loc[log["experiment_id"] != "EXP-005"].copy()
    row = pd.DataFrame(
        [
            {
                "experiment_id": "EXP-005",
                "datetime": datetime.now().astimezone().isoformat(timespec="seconds"),
                "model": "CatBoost_LightGBM_Blend",
                "features": "original_features",
                "cv_strategy": "OOF_blend",
                "cv_roc_auc": f"{best_auc:.6f}",
                "kaggle_score": "",
                "notes": "linear blend optimized on OOF only",
            }
        ]
    )
    pd.concat([log, row], ignore_index=True).to_csv(LOG_PATH, index=False)


def main() -> None:
    aligned = load_and_align_oof()
    y_true = aligned["y_true"]
    cat_predictions = aligned["catboost_prediction"]
    lgbm_predictions = aligned["lightgbm_prediction"]
    cat_auc = float(roc_auc_score(y_true, cat_predictions))
    lgbm_auc = float(roc_auc_score(y_true, lgbm_predictions))
    pearson = float(cat_predictions.corr(lgbm_predictions, method="pearson"))
    spearman = float(cat_predictions.corr(lgbm_predictions, method="spearman"))

    coarse_weights = [round(value, 2) for value in np.arange(0.0, 1.0001, 0.05)]
    coarse_results = score_weights(aligned, coarse_weights)
    coarse_best = float(
        coarse_results.loc[coarse_results["oof_roc_auc"].idxmax(), "weight_catboost"]
    )
    refine_start = max(0.0, coarse_best - 0.05)
    refine_end = min(1.0, coarse_best + 0.05)
    refined_weights = [
        round(value, 2) for value in np.arange(refine_start, refine_end + 0.0001, 0.01)
    ]
    all_weights = sorted(set(coarse_weights + refined_weights))
    results = score_weights(aligned, all_weights)
    best_row = results.loc[results["oof_roc_auc"].idxmax()]
    best_weight = float(best_row["weight_catboost"])
    best_auc = float(best_row["oof_roc_auc"])
    improvement_vs_cat = best_auc - cat_auc
    improvement_vs_lgbm = best_auc - lgbm_auc
    qualifies = improvement_vs_cat >= IMPROVEMENT_THRESHOLD

    table_text = results.to_string(
        index=False,
        formatters={
            "weight_catboost": lambda value: f"{value:.2f}",
            "weight_lightgbm": lambda value: f"{value:.2f}",
            "oof_roc_auc": lambda value: f"{value:.8f}",
        },
    )
    report_lines = [
        "EXP-005 - CATBOOST/LIGHTGBM OOF BLEND",
        f"rows: {len(aligned)}",
        "alignment: IDs and y_true validated one-to-one",
        f"catboost_oof_auc: {cat_auc:.8f}",
        f"lightgbm_oof_auc: {lgbm_auc:.8f}",
        f"prediction_pearson: {pearson:.8f}",
        f"prediction_spearman: {spearman:.8f}",
        "",
        table_text,
        "",
        f"best_weight_catboost: {best_weight:.2f}",
        f"best_weight_lightgbm: {1.0 - best_weight:.2f}",
        f"best_blend_oof_auc: {best_auc:.8f}",
        f"improvement_vs_exp003: {improvement_vs_cat:+.8f}",
        f"improvement_vs_exp004: {improvement_vs_lgbm:+.8f}",
        f"required_improvement_vs_exp003: {IMPROVEMENT_THRESHOLD:.5f}",
        f"threshold_met: {qualifies}",
        f"submission_generated: {qualifies}",
    ]
    METRICS_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    if qualifies:
        create_submission(best_weight)
        update_experiment_log(best_auc)

    print(table_text)
    print(f"AUC CatBoost: {cat_auc:.8f}")
    print(f"AUC LightGBM: {lgbm_auc:.8f}")
    print(f"Pearson: {pearson:.8f}")
    print(f"Spearman: {spearman:.8f}")
    print(f"Mejor peso CatBoost: {best_weight:.2f}")
    print(f"Mejor AUC blend: {best_auc:.8f}")
    print(f"Mejora vs EXP-003: {improvement_vs_cat:+.8f}")
    print(f"Mejora vs EXP-004: {improvement_vs_lgbm:+.8f}")
    print(f"Submission generado: {qualifies}")


if __name__ == "__main__":
    main()
