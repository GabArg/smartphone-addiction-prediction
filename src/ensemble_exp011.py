"""EXP-011: determine whether EXP-010 improves the EXP-009 ensemble."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ensemble_exp009 import (
    blend,
    fold_metrics,
    normalized_rank,
    refine_weights,
    search_weights,
    simplex_weights,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"
METRICS_DIR = OUTPUTS_DIR / "metrics"
DATA_DIR = PROJECT_ROOT / "data"

MODELS = ["exp003", "exp006", "exp008", "exp010"]
OOF_PATHS = {
    "exp003": PREDICTIONS_DIR / "oof_exp003_catboost.csv",
    "exp006": PREDICTIONS_DIR / "oof_exp006_lightgbm_features.csv",
    "exp008": PREDICTIONS_DIR / "oof_exp008_xgboost.csv",
    "exp010": PREDICTIONS_DIR / "oof_exp010_xgboost_features.csv",
}
TEST_PATHS = {
    "exp003": SUBMISSIONS_DIR / "submission_exp003_catboost_4000.csv",
    "exp006": PREDICTIONS_DIR / "test_exp006_lightgbm_features.csv",
    "exp008": PREDICTIONS_DIR / "test_exp008_xgboost.csv",
    "exp010": PREDICTIONS_DIR / "test_exp010_xgboost_features.csv",
}
SAMPLE_PATH = DATA_DIR / "sample_submission.csv"
SUBMISSION_PATH = SUBMISSIONS_DIR / "submission_exp011_ensemble.csv"
METRICS_PATH = METRICS_DIR / "exp011_ensemble_metrics.txt"
LOG_PATH = METRICS_DIR / "experiment_log.csv"

BASELINE_MODELS = ("exp008", "exp003", "exp006")
BASELINE_WEIGHTS = (0.53, 0.23, 0.24)
IMPROVEMENT_THRESHOLD = 0.00003
NEAR_TIE_TOLERANCE = 0.00002


def load_oof() -> pd.DataFrame:
    aligned: pd.DataFrame | None = None
    expected_rows: int | None = None
    for model in MODELS:
        frame = pd.read_csv(OOF_PATHS[model])
        if frame.columns.tolist() != ["id", "y_true", "oof_prediction"]:
            raise ValueError(f"Esquema OOF inesperado para {model}.")
        if frame["id"].duplicated().any() or frame.isna().any().any():
            raise ValueError(f"IDs duplicados o NaN en OOF {model}.")
        expected_rows = len(frame) if expected_rows is None else expected_rows
        if len(frame) != expected_rows:
            raise ValueError(f"Cantidad de filas distinta para {model}.")
        renamed = frame.rename(columns={"y_true": f"y_{model}", "oof_prediction": model})
        aligned = renamed if aligned is None else aligned.merge(
            renamed, on="id", how="inner", validate="one_to_one", sort=False
        )
    assert aligned is not None and expected_rows is not None
    if len(aligned) != expected_rows:
        raise ValueError("Los conjuntos de IDs OOF no coinciden.")
    base_y = aligned["y_exp003"]
    for model in MODELS[1:]:
        if not base_y.equals(aligned[f"y_{model}"]):
            raise ValueError(f"y_true no coincide para {model}.")
    return aligned[["id", "y_exp003", *MODELS]].rename(columns={"y_exp003": "y_true"})


def search_refined(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    models: tuple[str, ...],
) -> dict[str, object]:
    coarse_auc, coarse_weights = search_weights(
        y_true, predictions, models, simplex_weights(len(models), 5)
    )
    refined_auc, refined_weights = search_weights(
        y_true, predictions, models, refine_weights(coarse_weights)
    )
    return {
        "models": models,
        "auc": refined_auc,
        "weights": refined_weights,
        "coarse_auc": coarse_auc,
        "coarse_weights": coarse_weights,
    }


def small_exp010_search(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    current: dict[str, object],
) -> dict[str, object]:
    models = tuple(current["models"])
    weights = tuple(current["weights"])
    exp010_index = models.index("exp010")
    center_units = [round(weight * 100) for weight in weights]
    candidates: set[tuple[float, ...]] = set()
    for exp010_units in range(0, 6):
        ranges = []
        for index, center in enumerate(center_units):
            if index == exp010_index:
                ranges.append([exp010_units])
            else:
                ranges.append(range(max(0, center - 5), min(100, center + 5) + 1))
        for first in ranges[0]:
            for second in ranges[1]:
                for third in ranges[2]:
                    fourth = 100 - first - second - third
                    if fourth in ranges[3]:
                        candidates.add((first / 100, second / 100, third / 100, fourth / 100))
    auc, best_weights = search_weights(y_true, predictions, models, sorted(candidates))
    if auc > float(current["auc"]):
        current = {**current, "auc": auc, "weights": best_weights, "small_weight_search": True}
    else:
        current = {**current, "small_weight_search": True}
    return current


def make_candidate(
    name: str,
    method: str,
    result: dict[str, object],
    predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
) -> dict[str, object]:
    models = tuple(result["models"])
    weights = tuple(result["weights"])
    prediction = blend(predictions, models, weights)
    fold_scores, fold_std = fold_metrics(y_true, prediction)
    return {
        "name": name,
        "method": method,
        "models": models,
        "weights": weights,
        "auc": float(result["auc"]),
        "prediction": prediction,
        "fold_scores": fold_scores,
        "fold_std": fold_std,
        "complexity": sum(weight > 0 for weight in weights),
        "both_xgboost": (
            weights[models.index("exp008")] > 0 and weights[models.index("exp010")] > 0
            if "exp008" in models and "exp010" in models else False
        ),
    }


def load_test(sample: pd.DataFrame) -> pd.DataFrame:
    aligned = sample[["id"]].copy()
    for model in MODELS:
        frame = pd.read_csv(TEST_PATHS[model])
        prediction_column = "addicted_label" if model == "exp003" else "prediction"
        if frame.columns.tolist() != ["id", prediction_column]:
            raise ValueError(f"Esquema test inesperado para {model}.")
        if frame["id"].duplicated().any() or frame.isna().any().any():
            raise ValueError(f"Predicciones test inválidas para {model}.")
        aligned = aligned.merge(
            frame.rename(columns={prediction_column: model}),
            on="id", how="left", validate="one_to_one", sort=False,
        )
    if aligned.isna().any().any() or len(aligned) != len(sample):
        raise ValueError("Predicciones faltantes después de alinear test.")
    return aligned


def validate_submission(submission: pd.DataFrame, sample: pd.DataFrame) -> None:
    if submission.columns.tolist() != ["id", "addicted_label"]:
        raise ValueError(f"Columnas inválidas: {submission.columns.tolist()}")
    if len(submission) != 296_302 or len(submission) != len(sample):
        raise ValueError(f"Cantidad de filas inválida: {len(submission)}")
    if submission.isna().any().any() or not submission["addicted_label"].between(0, 1).all():
        raise ValueError("NaN o valores fuera de [0,1] en submission.")
    if not submission["id"].equals(sample["id"]):
        raise ValueError("IDs distintos a sample_submission.")


def update_log(winner: dict[str, object]) -> None:
    columns = [
        "experiment_id", "datetime", "model", "features", "cv_strategy",
        "cv_roc_auc", "kaggle_score", "notes",
    ]
    log = pd.read_csv(LOG_PATH, dtype=str, keep_default_na=False)
    if log.columns.tolist() != columns:
        raise ValueError("Encabezado inesperado en experiment_log.csv.")
    log = log.loc[log["experiment_id"] != "EXP-011"].copy()
    weights_text = "/".join(f"{weight:.2f}" for weight in winner["weights"])
    models_text = "/".join(winner["models"])
    row = pd.DataFrame([{
        "experiment_id": "EXP-011",
        "datetime": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": "Ensemble_with_XGBoost_Features",
        "features": "existing_model_predictions",
        "cv_strategy": "OOF_ensemble_optimization",
        "cv_roc_auc": f"{winner['auc']:.6f}",
        "kaggle_score": "",
        "notes": f"{winner['method']}; models={models_text}; weights={weights_text}",
    }])
    pd.concat([log, row], ignore_index=True).to_csv(LOG_PATH, index=False)


def main() -> None:
    oof = load_oof()
    y_true = oof["y_true"].to_numpy()
    probabilities = {model: oof[model].to_numpy() for model in MODELS}
    ranks = {model: normalized_rank(probabilities[model]) for model in MODELS}
    individual_aucs = {
        model: float(roc_auc_score(y_true, probabilities[model])) for model in MODELS
    }
    pearson = oof[MODELS].corr(method="pearson")
    spearman = oof[MODELS].corr(method="spearman")

    baseline_prediction = blend(ranks, BASELINE_MODELS, BASELINE_WEIGHTS)
    baseline_auc = float(roc_auc_score(y_true, baseline_prediction))
    baseline_folds, baseline_std = fold_metrics(y_true, baseline_prediction)

    replacement_models = ("exp010", "exp003", "exp006")
    joint_models = ("exp008", "exp010", "exp003", "exp006")
    replacement_prob = search_refined(y_true, probabilities, replacement_models)
    joint_prob = search_refined(y_true, probabilities, joint_models)
    if tuple(joint_prob["weights"])[joint_models.index("exp010")] < 0.05:
        joint_prob = small_exp010_search(y_true, probabilities, joint_prob)

    probability_results = [
        make_candidate("replacement_probability", "probability", replacement_prob, probabilities, y_true),
        make_candidate("joint_probability", "probability", joint_prob, probabilities, y_true),
    ]
    best_probability = max(probability_results, key=lambda item: item["auc"])

    rank_results: list[dict[str, object]] = []
    if float(best_probability["auc"]) > baseline_auc:
        replacement_rank = search_refined(y_true, ranks, replacement_models)
        joint_rank = search_refined(y_true, ranks, joint_models)
        if tuple(joint_rank["weights"])[joint_models.index("exp010")] < 0.05:
            joint_rank = small_exp010_search(y_true, ranks, joint_rank)
        rank_results = [
            make_candidate("replacement_rank", "rank", replacement_rank, ranks, y_true),
            make_candidate("joint_rank", "rank", joint_rank, ranks, y_true),
        ]

    baseline_candidate = {
        "name": "exp009_baseline", "method": "rank", "models": BASELINE_MODELS,
        "weights": BASELINE_WEIGHTS, "auc": baseline_auc, "prediction": baseline_prediction,
        "fold_scores": baseline_folds, "fold_std": baseline_std, "complexity": 3,
        "both_xgboost": False,
    }
    all_candidates = [baseline_candidate, *probability_results, *rank_results]
    raw_best_auc = max(float(item["auc"]) for item in all_candidates)
    near_ties = [item for item in all_candidates if raw_best_auc - float(item["auc"]) < NEAR_TIE_TOLERANCE]
    winner = sorted(
        near_ties,
        key=lambda item: (
            int(item["complexity"]), float(item["fold_std"]),
            bool(item["both_xgboost"]), -float(item["auc"]),
        ),
    )[0]
    improvement = float(winner["auc"]) - baseline_auc
    qualifies = improvement >= IMPROVEMENT_THRESHOLD

    def describe(item: dict[str, object]) -> str:
        return (
            f"name={item['name']}; method={item['method']}; models={item['models']}; "
            f"weights={item['weights']}; auc={item['auc']:.8f}; "
            f"fold_scores={[round(value, 8) for value in item['fold_scores']]}; "
            f"std={item['fold_std']:.8f}"
        )

    report_lines = [
        "EXP-011 - DOES EXP-010 IMPROVE EXP-009?",
        *(f"{model}_individual_auc: {auc:.8f}" for model, auc in individual_aucs.items()),
        "", "pearson_correlation:", pearson.to_string(float_format=lambda x: f"{x:.8f}"),
        "", "spearman_correlation:", spearman.to_string(float_format=lambda x: f"{x:.8f}"),
        "", f"baseline_exp009: {describe(baseline_candidate)}",
        f"replacement_probability_search: {replacement_prob}",
        f"joint_probability_search: {joint_prob}",
        *(f"probability_finalist: {describe(item)}" for item in probability_results),
        *(f"rank_finalist: {describe(item)}" for item in rank_results),
        "", f"raw_best_auc: {raw_best_auc:.8f}",
        f"near_tie_tolerance: {NEAR_TIE_TOLERANCE:.5f}",
        f"winner: {describe(winner)}",
        f"winning_exp010_weight: {winner['weights'][winner['models'].index('exp010')] if 'exp010' in winner['models'] else 0.0:.2f}",
        f"improvement_vs_exp009: {improvement:+.8f}",
        f"required_improvement: {IMPROVEMENT_THRESHOLD:.5f}",
        f"threshold_met: {qualifies}",
        f"submission_generated: {qualifies}",
    ]
    METRICS_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    if qualifies:
        sample = pd.read_csv(SAMPLE_PATH)
        test = load_test(sample)
        test_inputs = {
            model: normalized_rank(test[model].to_numpy())
            if winner["method"] == "rank" else test[model].to_numpy()
            for model in winner["models"]
        }
        final_prediction = blend(test_inputs, tuple(winner["models"]), tuple(winner["weights"]))
        submission = pd.DataFrame({"id": sample["id"], "addicted_label": final_prediction})
        validate_submission(submission, sample)
        submission.to_csv(SUBMISSION_PATH, index=False)
        validate_submission(pd.read_csv(SUBMISSION_PATH), sample)
        update_log(winner)

    print("Individual AUCs:", individual_aucs)
    print("Pearson:\n", pearson)
    print("Spearman:\n", spearman)
    print("Baseline:", describe(baseline_candidate))
    print("Replacement:", describe(max(
        [item for item in probability_results + rank_results if "replacement" in item["name"]],
        key=lambda item: item["auc"],
    )))
    print("Joint:", describe(max(
        [item for item in probability_results + rank_results if "joint" in item["name"]],
        key=lambda item: item["auc"],
    )))
    print("Winner:", describe(winner))
    print("Improvement vs EXP-009:", improvement)
    print("Submission generated:", qualifies)


if __name__ == "__main__":
    main()
