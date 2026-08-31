"""EXP-009: OOF-only ensemble search incorporating XGBoost."""

from __future__ import annotations

from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"
METRICS_DIR = OUTPUTS_DIR / "metrics"
DATA_DIR = PROJECT_ROOT / "data"

MODEL_NAMES = ["exp008", "exp003", "exp004", "exp006"]
OOF_PATHS = {
    "exp003": PREDICTIONS_DIR / "oof_exp003_catboost.csv",
    "exp004": PREDICTIONS_DIR / "oof_exp004_lightgbm.csv",
    "exp006": PREDICTIONS_DIR / "oof_exp006_lightgbm_features.csv",
    "exp008": PREDICTIONS_DIR / "oof_exp008_xgboost.csv",
}
TEST_PATHS = {
    "exp003": SUBMISSIONS_DIR / "submission_exp003_catboost_4000.csv",
    "exp004": PREDICTIONS_DIR / "test_exp004_lightgbm.csv",
    "exp006": PREDICTIONS_DIR / "test_exp006_lightgbm_features.csv",
    "exp008": PREDICTIONS_DIR / "test_exp008_xgboost.csv",
}
SAMPLE_PATH = DATA_DIR / "sample_submission.csv"
SUBMISSION_PATH = SUBMISSIONS_DIR / "submission_exp009_ensemble.csv"
METRICS_PATH = METRICS_DIR / "exp009_ensemble_metrics.txt"
LOG_PATH = METRICS_DIR / "experiment_log.csv"

EXP007_AUC = 0.96425749
IMPROVEMENT_THRESHOLD = 0.00003
NEAR_TIE_TOLERANCE = 0.00002


def load_oof() -> pd.DataFrame:
    aligned: pd.DataFrame | None = None
    expected = ["id", "y_true", "oof_prediction"]
    expected_rows: int | None = None
    for name in MODEL_NAMES:
        frame = pd.read_csv(OOF_PATHS[name])
        if frame.columns.tolist() != expected:
            raise ValueError(f"Esquema OOF inesperado para {name}: {frame.columns.tolist()}")
        if frame["id"].duplicated().any() or frame.isna().any().any():
            raise ValueError(f"IDs duplicados o NaN en OOF {name}.")
        if expected_rows is None:
            expected_rows = len(frame)
        elif len(frame) != expected_rows:
            raise ValueError(f"Cantidad de filas OOF distinta para {name}.")
        renamed = frame.rename(columns={"y_true": f"y_{name}", "oof_prediction": name})
        aligned = renamed if aligned is None else aligned.merge(
            renamed, on="id", how="inner", validate="one_to_one", sort=False
        )
    assert aligned is not None and expected_rows is not None
    if len(aligned) != expected_rows:
        raise ValueError("Los conjuntos de IDs OOF no coinciden.")
    base_y = aligned["y_exp008"]
    for name in MODEL_NAMES[1:]:
        if not base_y.equals(aligned[f"y_{name}"]):
            raise ValueError(f"y_true no coincide para {name}.")
    return aligned[["id", "y_exp008", *MODEL_NAMES]].rename(columns={"y_exp008": "y_true"})


def normalized_rank(values: np.ndarray) -> np.ndarray:
    ranks = pd.Series(values).rank(method="average").to_numpy(dtype=np.float64)
    return (ranks - 1.0) / (len(ranks) - 1.0)


def blend(predictions: dict[str, np.ndarray], models: tuple[str, ...], weights: tuple[float, ...]) -> np.ndarray:
    return sum(weight * predictions[model] for model, weight in zip(models, weights))


def simplex_weights(count: int, step: int) -> list[tuple[float, ...]]:
    total_units = 100 // step
    output: list[tuple[float, ...]] = []
    def build(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            output.append(tuple(value / total_units for value in [*prefix, remaining]))
            return
        for value in range(remaining + 1):
            build([*prefix, value], remaining - value, slots - 1)
    build([], total_units, count)
    return output


def refine_weights(center: tuple[float, ...], radius: int = 5) -> list[tuple[float, ...]]:
    center_units = tuple(round(value * 100) for value in center)
    candidates: list[tuple[float, ...]] = []
    def build(prefix: list[int], remaining: int, index: int) -> None:
        if index == len(center_units) - 1:
            if 0 <= remaining <= 100 and abs(remaining - center_units[index]) <= radius:
                candidates.append(tuple(value / 100 for value in [*prefix, remaining]))
            return
        low = max(0, center_units[index] - radius)
        high = min(100, center_units[index] + radius, remaining)
        for value in range(low, high + 1):
            build([*prefix, value], remaining - value, index + 1)
    build([], 100, 0)
    return candidates


def search_weights(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    models: tuple[str, ...],
    weights_to_try: list[tuple[float, ...]],
) -> tuple[float, tuple[float, ...]]:
    best_auc = -np.inf
    best_weights = weights_to_try[0]
    for weights in weights_to_try:
        auc = roc_auc_score(y_true, blend(predictions, models, weights))
        if auc > best_auc:
            best_auc = float(auc)
            best_weights = weights
    return best_auc, best_weights


def search_two(y_true: np.ndarray, predictions: dict[str, np.ndarray], models: tuple[str, str]) -> tuple[float, tuple[float, float]]:
    weights = [(value / 100, 1.0 - value / 100) for value in range(101)]
    return search_weights(y_true, predictions, models, weights)


def search_three(y_true: np.ndarray, predictions: dict[str, np.ndarray], models: tuple[str, str, str]) -> dict[str, object]:
    coarse_auc, coarse_weights = search_weights(y_true, predictions, models, simplex_weights(3, 5))
    refined_auc, refined_weights = search_weights(y_true, predictions, models, refine_weights(coarse_weights))
    return {
        "models": models, "auc": refined_auc, "weights": refined_weights,
        "coarse_auc": coarse_auc, "coarse_weights": coarse_weights,
    }


def fold_metrics(y_true: np.ndarray, prediction: np.ndarray) -> tuple[list[float], float]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = [
        float(roc_auc_score(y_true[valid_indices], prediction[valid_indices]))
        for _, valid_indices in cv.split(np.zeros(len(y_true)), y_true)
    ]
    return scores, float(np.std(scores))


def candidate(
    name: str, method: str, models: tuple[str, ...], weights: tuple[float, ...], auc: float,
    predictions: dict[str, np.ndarray], y_true: np.ndarray,
) -> dict[str, object]:
    prediction = blend(predictions, models, weights)
    folds, std = fold_metrics(y_true, prediction)
    return {
        "name": name, "method": method, "models": models, "weights": weights,
        "auc": auc, "prediction": prediction, "fold_scores": folds, "fold_std": std,
        "complexity": sum(weight > 0 for weight in weights),
    }


def load_test(sample: pd.DataFrame) -> pd.DataFrame:
    aligned = sample[["id"]].copy()
    for name in MODEL_NAMES:
        frame = pd.read_csv(TEST_PATHS[name])
        prediction_column = "addicted_label" if name == "exp003" else "prediction"
        if frame.columns.tolist() != ["id", prediction_column]:
            raise ValueError(f"Esquema test inesperado para {name}: {frame.columns.tolist()}")
        if frame["id"].duplicated().any() or frame.isna().any().any():
            raise ValueError(f"Predicciones test inválidas para {name}.")
        aligned = aligned.merge(
            frame.rename(columns={prediction_column: name}), on="id", how="left",
            validate="one_to_one", sort=False,
        )
    if aligned.isna().any().any() or len(aligned) != len(sample):
        raise ValueError("Faltan predicciones test después de alinear.")
    return aligned


def validate_submission(submission: pd.DataFrame, sample: pd.DataFrame) -> None:
    if submission.columns.tolist() != ["id", "addicted_label"]:
        raise ValueError(f"Columnas submission inválidas: {submission.columns.tolist()}")
    if len(submission) != 296_302 or len(submission) != len(sample):
        raise ValueError(f"Cantidad de filas inválida: {len(submission)}")
    if submission.isna().any().any() or not submission["addicted_label"].between(0, 1).all():
        raise ValueError("NaN o valores fuera de [0,1] en submission.")
    if not submission["id"].equals(sample["id"]):
        raise ValueError("IDs del submission distintos a sample_submission.")


def update_log(winner: dict[str, object], qualifies: bool) -> None:
    columns = [
        "experiment_id", "datetime", "model", "features", "cv_strategy",
        "cv_roc_auc", "kaggle_score", "notes",
    ]
    log = pd.read_csv(LOG_PATH, dtype=str, keep_default_na=False)
    if log.columns.tolist() != columns or (log["experiment_id"] == "EXP-008").sum() != 1:
        raise ValueError("Estado inesperado de experiment_log.csv.")
    log.loc[log["experiment_id"] == "EXP-008", "kaggle_score"] = "0.96587"
    if qualifies:
        log = log.loc[log["experiment_id"] != "EXP-009"].copy()
        weights_text = "/".join(f"{weight:.2f}" for weight in winner["weights"])
        models_text = "/".join(winner["models"])
        row = pd.DataFrame([{
            "experiment_id": "EXP-009",
            "datetime": datetime.now().astimezone().isoformat(timespec="seconds"),
            "model": "Ensemble_with_XGBoost",
            "features": "existing_model_predictions",
            "cv_strategy": "OOF_ensemble_optimization",
            "cv_roc_auc": f"{winner['auc']:.6f}",
            "kaggle_score": "",
            "notes": f"{winner['method']}; models={models_text}; weights={weights_text}",
        }])
        log = pd.concat([log, row], ignore_index=True)
    log.to_csv(LOG_PATH, index=False)


def main() -> None:
    oof = load_oof()
    y_true = oof["y_true"].to_numpy()
    probabilities = {name: oof[name].to_numpy() for name in MODEL_NAMES}
    ranks = {name: normalized_rank(probabilities[name]) for name in MODEL_NAMES}
    individual_aucs = {name: float(roc_auc_score(y_true, probabilities[name])) for name in MODEL_NAMES}
    pearson = oof[MODEL_NAMES].corr(method="pearson")
    spearman = oof[MODEL_NAMES].corr(method="spearman")

    prob_two_xgb_cat = search_two(y_true, probabilities, ("exp008", "exp003"))
    prob_two_xgb_lgbf = search_two(y_true, probabilities, ("exp008", "exp006"))
    prob_three = search_three(y_true, probabilities, ("exp008", "exp003", "exp006"))

    four_models = ("exp008", "exp003", "exp004", "exp006")
    prob_four_coarse_auc, prob_four_coarse_weights = search_weights(
        y_true, probabilities, four_models, simplex_weights(4, 5)
    )
    four_refined = prob_four_coarse_auc >= float(prob_three["auc"]) + 0.00002
    if four_refined:
        prob_four_auc, prob_four_weights = search_weights(
            y_true, probabilities, four_models, refine_weights(prob_four_coarse_weights)
        )
    else:
        prob_four_auc, prob_four_weights = prob_four_coarse_auc, prob_four_coarse_weights

    rank_two_xgb_cat = search_two(y_true, ranks, ("exp008", "exp003"))
    rank_two_xgb_lgbf = search_two(y_true, ranks, ("exp008", "exp006"))
    rank_three = search_three(y_true, ranks, ("exp008", "exp003", "exp006"))

    probability_candidates = [
        candidate("prob_xgb_cat", "probability", ("exp008", "exp003"), prob_two_xgb_cat[1], prob_two_xgb_cat[0], probabilities, y_true),
        candidate("prob_xgb_exp006", "probability", ("exp008", "exp006"), prob_two_xgb_lgbf[1], prob_two_xgb_lgbf[0], probabilities, y_true),
        candidate("prob_three", "probability", tuple(prob_three["models"]), tuple(prob_three["weights"]), float(prob_three["auc"]), probabilities, y_true),
        candidate("prob_four", "probability", four_models, prob_four_weights, prob_four_auc, probabilities, y_true),
    ]
    rank_candidates = [
        candidate("rank_xgb_cat", "rank", ("exp008", "exp003"), rank_two_xgb_cat[1], rank_two_xgb_cat[0], ranks, y_true),
        candidate("rank_xgb_exp006", "rank", ("exp008", "exp006"), rank_two_xgb_lgbf[1], rank_two_xgb_lgbf[0], ranks, y_true),
        candidate("rank_three", "rank", tuple(rank_three["models"]), tuple(rank_three["weights"]), float(rank_three["auc"]), ranks, y_true),
    ]
    best_probability = max(probability_candidates, key=lambda item: item["auc"])
    best_rank = max(rank_candidates, key=lambda item: item["auc"])
    all_candidates = probability_candidates + rank_candidates
    raw_best_auc = max(float(item["auc"]) for item in all_candidates)
    near_ties = [item for item in all_candidates if raw_best_auc - float(item["auc"]) < NEAR_TIE_TOLERANCE]
    winner = sorted(
        near_ties,
        key=lambda item: (int(item["complexity"]), float(item["fold_std"]), -float(item["auc"])),
    )[0]

    xgb_fold_scores, xgb_fold_std = fold_metrics(y_true, probabilities["exp008"])
    improvement_vs_exp007 = float(winner["auc"]) - EXP007_AUC
    improvement_vs_exp008 = float(winner["auc"]) - individual_aucs["exp008"]
    qualifies = improvement_vs_exp008 >= IMPROVEMENT_THRESHOLD

    def describe(item: dict[str, object]) -> str:
        return (
            f"name={item['name']}; method={item['method']}; models={item['models']}; "
            f"weights={item['weights']}; auc={item['auc']:.8f}; "
            f"fold_scores={[round(x, 8) for x in item['fold_scores']]}; std={item['fold_std']:.8f}"
        )

    report_lines = [
        "EXP-009 - OOF ENSEMBLE WITH XGBOOST",
        *(f"{name}_individual_auc: {auc:.8f}" for name, auc in individual_aucs.items()),
        "", "pearson_correlation:", pearson.to_string(float_format=lambda x: f"{x:.8f}"),
        "", "spearman_correlation:", spearman.to_string(float_format=lambda x: f"{x:.8f}"),
        "", "candidate_searches:",
        f"prob_two_xgb_cat: auc={prob_two_xgb_cat[0]:.8f}; weights={prob_two_xgb_cat[1]}",
        f"prob_two_xgb_exp006: auc={prob_two_xgb_lgbf[0]:.8f}; weights={prob_two_xgb_lgbf[1]}",
        f"prob_three: {prob_three}",
        f"prob_four_coarse: auc={prob_four_coarse_auc:.8f}; weights={prob_four_coarse_weights}",
        f"prob_four_refined={four_refined}; final_auc={prob_four_auc:.8f}; final_weights={prob_four_weights}",
        f"rank_two_xgb_cat: auc={rank_two_xgb_cat[0]:.8f}; weights={rank_two_xgb_cat[1]}",
        f"rank_two_xgb_exp006: auc={rank_two_xgb_lgbf[0]:.8f}; weights={rank_two_xgb_lgbf[1]}",
        f"rank_three: {rank_three}",
        "", "finalists:",
        f"exp008_individual: fold_scores={xgb_fold_scores}; std={xgb_fold_std:.8f}",
        f"best_probability: {describe(best_probability)}",
        f"best_rank: {describe(best_rank)}",
        "", f"raw_best_auc: {raw_best_auc:.8f}",
        f"selection_tolerance: {NEAR_TIE_TOLERANCE:.5f}",
        f"winning_candidate: {describe(winner)}",
        f"improvement_vs_exp007: {improvement_vs_exp007:+.8f}",
        f"improvement_vs_exp008: {improvement_vs_exp008:+.8f}",
        f"required_improvement_vs_exp008: {IMPROVEMENT_THRESHOLD:.5f}",
        f"threshold_met: {qualifies}",
        f"submission_generated: {qualifies}",
    ]
    METRICS_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    if qualifies:
        sample = pd.read_csv(SAMPLE_PATH)
        aligned_test = load_test(sample)
        test_predictions = {
            name: normalized_rank(aligned_test[name].to_numpy())
            if winner["method"] == "rank" else aligned_test[name].to_numpy()
            for name in winner["models"]
        }
        final_prediction = blend(test_predictions, tuple(winner["models"]), tuple(winner["weights"]))
        submission = pd.DataFrame({"id": sample["id"], "addicted_label": final_prediction})
        validate_submission(submission, sample)
        submission.to_csv(SUBMISSION_PATH, index=False)
        validate_submission(pd.read_csv(SUBMISSION_PATH), sample)

    update_log(winner, qualifies)
    print("Individual AUCs:", individual_aucs)
    print("Pearson:\n", pearson)
    print("Spearman:\n", spearman)
    print("Best probability:", describe(best_probability))
    print("Best rank:", describe(best_rank))
    print("Winner:", describe(winner))
    print("Improvement vs EXP-007:", improvement_vs_exp007)
    print("Improvement vs EXP-008:", improvement_vs_exp008)
    print("Submission generated:", qualifies)


if __name__ == "__main__":
    main()
