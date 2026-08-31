"""EXP-007: OOF-only probability and rank ensemble search without training."""

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

OOF_PATHS = {
    "exp003": PREDICTIONS_DIR / "oof_exp003_catboost.csv",
    "exp004": PREDICTIONS_DIR / "oof_exp004_lightgbm.csv",
    "exp006": PREDICTIONS_DIR / "oof_exp006_lightgbm_features.csv",
}
TEST_PATHS = {
    "exp003": SUBMISSIONS_DIR / "submission_exp003_catboost_4000.csv",
    "exp004": PREDICTIONS_DIR / "test_exp004_lightgbm.csv",
    "exp006": PREDICTIONS_DIR / "test_exp006_lightgbm_features.csv",
}
SAMPLE_PATH = DATA_DIR / "sample_submission.csv"
EXP005_METRICS_PATH = METRICS_DIR / "exp005_blend_metrics.txt"
METRICS_PATH = METRICS_DIR / "exp007_ensemble_metrics.txt"
SUBMISSION_PATH = SUBMISSIONS_DIR / "submission_exp007_ensemble.csv"
LOG_PATH = METRICS_DIR / "experiment_log.csv"

ID_COLUMN = "id"
TARGET = "addicted_label"
MODEL_NAMES = ["exp003", "exp004", "exp006"]
THRESHOLD = 0.00003


def load_oof() -> pd.DataFrame:
    aligned: pd.DataFrame | None = None
    expected = [ID_COLUMN, "y_true", "oof_prediction"]
    for name in MODEL_NAMES:
        frame = pd.read_csv(OOF_PATHS[name])
        if frame.columns.tolist() != expected:
            raise ValueError(f"Esquema OOF inesperado para {name}: {frame.columns.tolist()}")
        if frame[ID_COLUMN].duplicated().any() or frame.isna().any().any():
            raise ValueError(f"IDs duplicados o NaN en OOF {name}.")
        renamed = frame.rename(
            columns={"y_true": f"y_{name}", "oof_prediction": name}
        )
        aligned = renamed if aligned is None else aligned.merge(
            renamed, on=ID_COLUMN, how="inner", validate="one_to_one", sort=False
        )
    assert aligned is not None
    expected_rows = len(pd.read_csv(OOF_PATHS["exp003"], usecols=[ID_COLUMN]))
    if len(aligned) != expected_rows:
        raise ValueError("Los conjuntos de IDs OOF no coinciden.")
    base_target = aligned["y_exp003"]
    for name in MODEL_NAMES[1:]:
        if not base_target.equals(aligned[f"y_{name}"]):
            raise ValueError(f"y_true de {name} no coincide con EXP-003.")
    return aligned[[ID_COLUMN, "y_exp003", *MODEL_NAMES]].rename(
        columns={"y_exp003": "y_true"}
    )


def normalized_rank(values: pd.Series) -> np.ndarray:
    ranks = values.rank(method="average").to_numpy(dtype=np.float64)
    if len(ranks) <= 1:
        return np.zeros_like(ranks)
    return (ranks - 1.0) / (len(ranks) - 1.0)


def best_two_model(
    y_true: np.ndarray, first: np.ndarray, second: np.ndarray
) -> tuple[float, tuple[float, float]]:
    best_auc = -np.inf
    best_weights = (0.0, 1.0)
    for integer_weight in range(101):
        w_first = integer_weight / 100.0
        auc = roc_auc_score(y_true, w_first * first + (1.0 - w_first) * second)
        if auc > best_auc:
            best_auc = float(auc)
            best_weights = (w_first, 1.0 - w_first)
    return best_auc, best_weights


def simplex_weights(step_units: int) -> list[tuple[float, float, float]]:
    weights = []
    for first in range(0, 101, step_units):
        for second in range(0, 101 - first, step_units):
            third = 100 - first - second
            if third % step_units == 0:
                weights.append((first / 100, second / 100, third / 100))
    return weights


def refine_weights(center: tuple[float, float, float]) -> list[tuple[float, float, float]]:
    center_units = tuple(round(weight * 100) for weight in center)
    candidates: set[tuple[float, float, float]] = set()
    for first in range(max(0, center_units[0] - 5), min(100, center_units[0] + 5) + 1):
        for second in range(max(0, center_units[1] - 5), min(100, center_units[1] + 5) + 1):
            third = 100 - first - second
            if third < 0 or abs(third - center_units[2]) > 5:
                continue
            candidates.add((first / 100, second / 100, third / 100))
    return sorted(candidates)


def best_three_model(
    y_true: np.ndarray, predictions: list[np.ndarray]
) -> tuple[float, tuple[float, float, float], float, tuple[float, float, float]]:
    coarse_best_auc = -np.inf
    coarse_best_weights = (0.0, 0.0, 1.0)
    for weights in simplex_weights(5):
        blended = sum(weight * prediction for weight, prediction in zip(weights, predictions))
        auc = roc_auc_score(y_true, blended)
        if auc > coarse_best_auc:
            coarse_best_auc = float(auc)
            coarse_best_weights = weights
    refined_best_auc = coarse_best_auc
    refined_best_weights = coarse_best_weights
    for weights in refine_weights(coarse_best_weights):
        blended = sum(weight * prediction for weight, prediction in zip(weights, predictions))
        auc = roc_auc_score(y_true, blended)
        if auc > refined_best_auc:
            refined_best_auc = float(auc)
            refined_best_weights = weights
    return refined_best_auc, refined_best_weights, coarse_best_auc, coarse_best_weights


def load_aligned_test(sample: pd.DataFrame) -> pd.DataFrame:
    aligned = sample[[ID_COLUMN]].copy()
    for name in MODEL_NAMES:
        frame = pd.read_csv(TEST_PATHS[name])
        prediction_column = TARGET if name == "exp003" else "prediction"
        expected = [ID_COLUMN, prediction_column]
        if frame.columns.tolist() != expected:
            raise ValueError(f"Esquema test inesperado para {name}: {frame.columns.tolist()}")
        if frame[ID_COLUMN].duplicated().any() or frame.isna().any().any():
            raise ValueError(f"IDs duplicados o NaN en test {name}.")
        aligned = aligned.merge(
            frame.rename(columns={prediction_column: name}),
            on=ID_COLUMN,
            how="left",
            validate="one_to_one",
            sort=False,
        )
    if aligned.isna().any().any() or len(aligned) != len(sample):
        raise ValueError("Predicciones faltantes al alinear test.")
    return aligned


def validate_submission(submission: pd.DataFrame, sample: pd.DataFrame) -> None:
    if submission.columns.tolist() != [ID_COLUMN, TARGET]:
        raise ValueError(f"Columnas inválidas: {submission.columns.tolist()}")
    if len(submission) != 296_302 or len(submission) != len(sample):
        raise ValueError(f"Cantidad de filas inválida: {len(submission)}")
    if submission.isna().any().any():
        raise ValueError("El submission contiene NaN.")
    if not submission[TARGET].between(0, 1, inclusive="both").all():
        raise ValueError("Valores del submission fuera de [0, 1].")
    if not submission[ID_COLUMN].equals(sample[ID_COLUMN]):
        raise ValueError("IDs del submission distintos a sample_submission.")


def parse_exp005_auc() -> float:
    for line in EXP005_METRICS_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("best_blend_oof_auc: "):
            return float(line.split(": ", 1)[1])
    raise ValueError("No se encontró best_blend_oof_auc en métricas EXP-005.")


def update_log(best_auc: float, method: str, weights: tuple[float, ...], qualifies: bool) -> None:
    columns = [
        "experiment_id", "datetime", "model", "features", "cv_strategy",
        "cv_roc_auc", "kaggle_score", "notes",
    ]
    log = pd.read_csv(LOG_PATH, dtype=str, keep_default_na=False)
    if log.columns.tolist() != columns:
        raise ValueError(f"Encabezado inesperado en {LOG_PATH}: {log.columns.tolist()}")
    if (log["experiment_id"] == "EXP-006").sum() != 1:
        raise ValueError("Se esperaba exactamente una fila EXP-006.")
    log.loc[log["experiment_id"] == "EXP-006", "kaggle_score"] = "0.96538"
    if qualifies:
        log = log.loc[log["experiment_id"] != "EXP-007"].copy()
        weight_text = "/".join(f"{weight:.2f}" for weight in weights)
        row = pd.DataFrame([{
            "experiment_id": "EXP-007",
            "datetime": datetime.now().astimezone().isoformat(timespec="seconds"),
            "model": "Ensemble",
            "features": "multi_model_existing_predictions",
            "cv_strategy": "OOF_ensemble_optimization",
            "cv_roc_auc": f"{best_auc:.6f}",
            "kaggle_score": "",
            "notes": f"{method}; weights exp003/exp004/exp006={weight_text}",
        }])
        log = pd.concat([log, row], ignore_index=True)
    log.to_csv(LOG_PATH, index=False)


def main() -> None:
    oof = load_oof()
    y_true = oof["y_true"].to_numpy()
    probabilities = {name: oof[name].to_numpy() for name in MODEL_NAMES}
    ranks = {name: normalized_rank(oof[name]) for name in MODEL_NAMES}
    individual_aucs = {
        name: float(roc_auc_score(y_true, probabilities[name])) for name in MODEL_NAMES
    }
    pearson = oof[MODEL_NAMES].corr(method="pearson")
    spearman = oof[MODEL_NAMES].corr(method="spearman")

    prob_two_auc, prob_two_weights = best_two_model(
        y_true, probabilities["exp003"], probabilities["exp006"]
    )
    prob_three_auc, prob_three_weights, prob_coarse_auc, prob_coarse_weights = best_three_model(
        y_true, [probabilities[name] for name in MODEL_NAMES]
    )
    rank_two_auc, rank_two_weights = best_two_model(
        y_true, ranks["exp003"], ranks["exp006"]
    )
    rank_three_auc, rank_three_weights, rank_coarse_auc, rank_coarse_weights = best_three_model(
        y_true, [ranks[name] for name in MODEL_NAMES]
    )

    candidates = [
        ("probability_two_exp003_exp006", prob_two_auc, (prob_two_weights[0], 0.0, prob_two_weights[1]), False),
        ("probability_three", prob_three_auc, prob_three_weights, False),
        ("rank_two_exp003_exp006", rank_two_auc, (rank_two_weights[0], 0.0, rank_two_weights[1]), True),
        ("rank_three", rank_three_auc, rank_three_weights, True),
    ]
    method, best_auc, best_weights, is_rank = max(candidates, key=lambda item: item[1])
    exp005_auc = parse_exp005_auc()
    improvement_vs_exp005 = best_auc - exp005_auc
    best_individual_auc = max(individual_aucs.values())
    improvement_vs_best_individual = best_auc - best_individual_auc
    qualifies = improvement_vs_exp005 >= THRESHOLD

    report_lines = [
        "EXP-007 - EXISTING MODEL ENSEMBLE OPTIMIZATION",
        *(f"{name}_oof_auc: {auc:.8f}" for name, auc in individual_aucs.items()),
        "", "pearson_correlation:", pearson.to_string(float_format=lambda x: f"{x:.8f}"),
        "", "spearman_correlation:", spearman.to_string(float_format=lambda x: f"{x:.8f}"),
        "",
        f"probability_two_auc: {prob_two_auc:.8f}",
        f"probability_two_weights_exp003_exp006: {prob_two_weights}",
        f"probability_three_coarse_auc: {prob_coarse_auc:.8f}",
        f"probability_three_coarse_weights: {prob_coarse_weights}",
        f"probability_three_refined_auc: {prob_three_auc:.8f}",
        f"probability_three_refined_weights_exp003_exp004_exp006: {prob_three_weights}",
        f"rank_two_auc: {rank_two_auc:.8f}",
        f"rank_two_weights_exp003_exp006: {rank_two_weights}",
        f"rank_three_coarse_auc: {rank_coarse_auc:.8f}",
        f"rank_three_coarse_weights: {rank_coarse_weights}",
        f"rank_three_refined_auc: {rank_three_auc:.8f}",
        f"rank_three_refined_weights_exp003_exp004_exp006: {rank_three_weights}",
        "",
        f"winning_method: {method}",
        f"winning_weights_exp003_exp004_exp006: {best_weights}",
        f"best_oof_auc: {best_auc:.8f}",
        f"exp005_oof_auc: {exp005_auc:.8f}",
        f"improvement_vs_exp005: {improvement_vs_exp005:+.8f}",
        f"improvement_vs_best_individual: {improvement_vs_best_individual:+.8f}",
        f"threshold: {THRESHOLD:.5f}",
        f"threshold_met: {qualifies}",
        f"submission_generated: {qualifies}",
    ]
    METRICS_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    if qualifies:
        sample = pd.read_csv(SAMPLE_PATH)
        aligned_test = load_aligned_test(sample)
        test_inputs = {
            name: normalized_rank(aligned_test[name]) if is_rank else aligned_test[name].to_numpy()
            for name in MODEL_NAMES
        }
        final_prediction = sum(
            weight * test_inputs[name] for weight, name in zip(best_weights, MODEL_NAMES)
        )
        submission = pd.DataFrame({ID_COLUMN: sample[ID_COLUMN], TARGET: final_prediction})
        validate_submission(submission, sample)
        submission.to_csv(SUBMISSION_PATH, index=False)
        validate_submission(pd.read_csv(SUBMISSION_PATH), sample)

    update_log(best_auc, method, best_weights, qualifies)

    print("Individual AUCs:", individual_aucs)
    print("Pearson:\n", pearson)
    print("Spearman:\n", spearman)
    print("Best probability two:", prob_two_auc, prob_two_weights)
    print("Best probability three:", prob_three_auc, prob_three_weights)
    print("Best rank two:", rank_two_auc, rank_two_weights)
    print("Best rank three:", rank_three_auc, rank_three_weights)
    print("Winner:", method, best_auc, best_weights)
    print("Improvement vs EXP-005:", improvement_vs_exp005)
    print("Submission generated:", qualifies)


if __name__ == "__main__":
    main()
