"""EXP-013: OOF-only ensemble search centered on EXP-012."""

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

MODELS = ["exp012", "exp003", "exp006", "exp008", "exp010"]
OOF_PATHS = {
    "exp012": PREDICTIONS_DIR / "oof_exp012_xgboost_thresholds.csv",
    "exp003": PREDICTIONS_DIR / "oof_exp003_catboost.csv",
    "exp006": PREDICTIONS_DIR / "oof_exp006_lightgbm_features.csv",
    "exp008": PREDICTIONS_DIR / "oof_exp008_xgboost.csv",
    "exp010": PREDICTIONS_DIR / "oof_exp010_xgboost_features.csv",
}
TEST_PATHS = {
    "exp012": PREDICTIONS_DIR / "test_exp012_xgboost_thresholds.csv",
    "exp003": SUBMISSIONS_DIR / "submission_exp003_catboost_4000.csv",
    "exp006": PREDICTIONS_DIR / "test_exp006_lightgbm_features.csv",
    "exp008": PREDICTIONS_DIR / "test_exp008_xgboost.csv",
    "exp010": PREDICTIONS_DIR / "test_exp010_xgboost_features.csv",
}
SAMPLE_PATH = DATA_DIR / "sample_submission.csv"
SUBMISSION_PATH = SUBMISSIONS_DIR / "submission_exp013_ensemble.csv"
METRICS_PATH = METRICS_DIR / "exp013_ensemble_metrics.txt"
LOG_PATH = METRICS_DIR / "experiment_log.csv"

PAIR_SETS = [
    ("exp012", "exp003"), ("exp012", "exp006"),
    ("exp012", "exp008"), ("exp012", "exp010"),
]
TRIPLE_SETS = [
    ("exp012", "exp003", "exp006"),
    ("exp012", "exp003", "exp008"),
    ("exp012", "exp003", "exp010"),
    ("exp012", "exp006", "exp008"),
    ("exp012", "exp006", "exp010"),
]
IMPROVEMENT_THRESHOLD = 0.00003
STRONG_IMPROVEMENT = 0.00010
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
            raise ValueError(f"Cantidad OOF distinta para {model}.")
        renamed = frame.rename(columns={"y_true": f"y_{model}", "oof_prediction": model})
        aligned = renamed if aligned is None else aligned.merge(
            renamed, on="id", how="inner", validate="one_to_one", sort=False
        )
    assert aligned is not None and expected_rows is not None
    if len(aligned) != expected_rows:
        raise ValueError("Los IDs OOF no coinciden entre modelos.")
    base_y = aligned["y_exp012"]
    for model in MODELS[1:]:
        if not base_y.equals(aligned[f"y_{model}"]):
            raise ValueError(f"y_true no coincide para {model}.")
    return aligned[["id", "y_exp012", *MODELS]].rename(columns={"y_exp012": "y_true"})


def search_two(
    y_true: np.ndarray, predictions: dict[str, np.ndarray], models: tuple[str, str]
) -> dict[str, object]:
    weights = [(value / 100, 1.0 - value / 100) for value in range(101)]
    auc, best_weights = search_weights(y_true, predictions, models, weights)
    return {"models": models, "weights": best_weights, "auc": auc, "coarse_auc": auc}


def search_refined(
    y_true: np.ndarray, predictions: dict[str, np.ndarray], models: tuple[str, ...]
) -> dict[str, object]:
    coarse_auc, coarse_weights = search_weights(
        y_true, predictions, models, simplex_weights(len(models), 5)
    )
    refined_auc, refined_weights = search_weights(
        y_true, predictions, models, refine_weights(coarse_weights)
    )
    return {
        "models": models, "weights": refined_weights, "auc": refined_auc,
        "coarse_weights": coarse_weights, "coarse_auc": coarse_auc,
    }


def make_candidate(
    name: str, method: str, result: dict[str, object],
    predictions: dict[str, np.ndarray], y_true: np.ndarray,
) -> dict[str, object]:
    models = tuple(result["models"])
    weights = tuple(result["weights"])
    prediction = blend(predictions, models, weights)
    folds, std = fold_metrics(y_true, prediction)
    return {
        "name": name, "method": method, "models": models, "weights": weights,
        "auc": float(result["auc"]), "prediction": prediction,
        "fold_scores": folds, "fold_std": std,
        "complexity": sum(weight > 0 for weight in weights),
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
        raise ValueError("Faltan predicciones después de alinear test.")
    return aligned


def validate_submission(submission: pd.DataFrame, sample: pd.DataFrame) -> None:
    if submission.columns.tolist() != ["id", "addicted_label"]:
        raise ValueError(f"Columnas submission inválidas: {submission.columns.tolist()}")
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
    if log.columns.tolist() != columns or (log["experiment_id"] == "EXP-012").sum() != 1:
        raise ValueError("Estado inesperado de experiment_log.csv.")
    log.loc[log["experiment_id"] == "EXP-012", "kaggle_score"] = "0.96696"
    log = log.loc[log["experiment_id"] != "EXP-013"].copy()
    model_text = "/".join(winner["models"])
    weight_text = "/".join(f"{weight:.2f}" for weight in winner["weights"])
    row = pd.DataFrame([{
        "experiment_id": "EXP-013",
        "datetime": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": "Ensemble_with_XGBoost_Thresholds",
        "features": "existing_model_predictions",
        "cv_strategy": "OOF_ensemble_optimization",
        "cv_roc_auc": f"{winner['auc']:.6f}",
        "kaggle_score": "",
        "notes": f"{winner['method']}; models={model_text}; weights={weight_text}",
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

    baseline_folds, baseline_std = fold_metrics(y_true, probabilities["exp012"])
    baseline = {
        "name": "exp012_individual", "method": "probability", "models": ("exp012",),
        "weights": (1.0,), "auc": individual_aucs["exp012"],
        "prediction": probabilities["exp012"], "fold_scores": baseline_folds,
        "fold_std": baseline_std, "complexity": 1,
    }

    pair_results = [search_two(y_true, probabilities, models) for models in PAIR_SETS]
    pair_candidates = [
        make_candidate(f"prob_pair_{'_'.join(result['models'])}", "probability", result, probabilities, y_true)
        for result in pair_results
    ]
    triple_results = [search_refined(y_true, probabilities, models) for models in TRIPLE_SETS]
    triple_candidates = [
        make_candidate(f"prob_triple_{'_'.join(result['models'])}", "probability", result, probabilities, y_true)
        for result in triple_results
    ]
    best_pair = max(pair_candidates, key=lambda item: item["auc"])
    best_triple = max(triple_candidates, key=lambda item: item["auc"])

    multi_candidates: list[dict[str, object]] = []
    multi_tested = False
    five_tested = False
    best_triple_weights = tuple(best_triple["weights"])
    if sum(weight >= 0.05 for weight in best_triple_weights) >= 3:
        multi_tested = True
        outside = [model for model in MODELS if model not in best_triple["models"]]
        four_results = []
        for extra_model in outside:
            models = tuple([*best_triple["models"], extra_model])
            four_results.append(search_refined(y_true, probabilities, models))
        multi_candidates = [
            make_candidate(f"prob_four_{'_'.join(result['models'])}", "probability", result, probabilities, y_true)
            for result in four_results
        ]
        best_four = max(multi_candidates, key=lambda item: item["auc"])
        added_model = best_four["models"][-1]
        added_weight = best_four["weights"][-1]
        if float(best_four["auc"]) >= float(best_triple["auc"]) + 0.00002 and added_weight >= 0.05:
            five_tested = True
            five_result = search_refined(y_true, probabilities, tuple(MODELS))
            multi_candidates.append(
                make_candidate("prob_five", "probability", five_result, probabilities, y_true)
            )

    probability_candidates = [*pair_candidates, *triple_candidates, *multi_candidates]
    best_probability = max(probability_candidates, key=lambda item: item["auc"])

    rank_model_sets = {
        ("exp012", "exp003"),
        ("exp012", "exp003", "exp006"),
        tuple(best_probability["models"]),
    }
    rank_candidates: list[dict[str, object]] = []
    for models in rank_model_sets:
        result = search_two(y_true, ranks, models) if len(models) == 2 else search_refined(
            y_true, ranks, models
        )
        rank_candidates.append(
            make_candidate(f"rank_{'_'.join(models)}", "rank", result, ranks, y_true)
        )
    best_rank = max(rank_candidates, key=lambda item: item["auc"])

    all_candidates = [baseline, *probability_candidates, *rank_candidates]
    raw_best_auc = max(float(item["auc"]) for item in all_candidates)
    near_ties = [item for item in all_candidates if raw_best_auc - float(item["auc"]) < NEAR_TIE_TOLERANCE]
    winner = sorted(
        near_ties,
        key=lambda item: (
            float(item["fold_std"]), int(item["complexity"]),
            0 if item["method"] == "probability" else 1, -float(item["auc"]),
        ),
    )[0]
    improvement = float(winner["auc"]) - float(baseline["auc"])
    fold_improvements = [
        new - old for new, old in zip(winner["fold_scores"], baseline["fold_scores"])
    ]
    std_change = float(winner["fold_std"]) - float(baseline["fold_std"])
    improved_folds = sum(value > 0 for value in fold_improvements)
    clear_variance_reduction = std_change <= -0.00002
    qualifies = (
        improvement >= STRONG_IMPROVEMENT
        or (
            improvement >= IMPROVEMENT_THRESHOLD
            and (improved_folds >= 4 or clear_variance_reduction)
        )
    )

    def describe(item: dict[str, object]) -> str:
        return (
            f"name={item['name']}; method={item['method']}; models={item['models']}; "
            f"weights={item['weights']}; auc={item['auc']:.8f}; "
            f"fold_scores={[round(value, 8) for value in item['fold_scores']]}; "
            f"std={item['fold_std']:.8f}"
        )

    report_lines = [
        "EXP-013 - ENSEMBLE CENTERED ON EXP-012",
        *(f"{model}_individual_auc: {auc:.8f}" for model, auc in individual_aucs.items()),
        "", "pearson_correlation:", pearson.to_string(float_format=lambda x: f"{x:.8f}"),
        "", "spearman_correlation:", spearman.to_string(float_format=lambda x: f"{x:.8f}"),
        "", f"exp012_baseline: {describe(baseline)}",
        f"exp011_reference_oof: 0.96483039",
        "", "probability_pair_results:",
        *(describe(item) for item in pair_candidates),
        "probability_triple_results:",
        *(describe(item) for item in triple_candidates),
        f"four_model_search_performed: {multi_tested}",
        f"five_model_search_performed: {five_tested}",
        *(f"multi_model_result: {describe(item)}" for item in multi_candidates),
        "", f"best_probability: {describe(best_probability)}",
        *(f"rank_result: {describe(item)}" for item in rank_candidates),
        f"best_rank: {describe(best_rank)}",
        "", f"raw_best_auc: {raw_best_auc:.8f}",
        f"near_tie_tolerance: {NEAR_TIE_TOLERANCE:.5f}",
        f"winner: {describe(winner)}",
        f"improvement_vs_exp012: {improvement:+.8f}",
        f"fold_improvements_vs_exp012: {fold_improvements}",
        f"improved_fold_count: {improved_folds}",
        f"std_change_vs_exp012: {std_change:+.8f}",
        f"threshold_met: {qualifies}",
        f"submission_generated: {qualifies}",
    ]
    METRICS_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    # EXP-012 public score is historical metadata and is updated regardless of EXP-013 qualification.
    log = pd.read_csv(LOG_PATH, dtype=str, keep_default_na=False)
    if (log["experiment_id"] == "EXP-012").sum() != 1:
        raise ValueError("Se esperaba exactamente una fila EXP-012 en el log.")
    log.loc[log["experiment_id"] == "EXP-012", "kaggle_score"] = "0.96696"
    log.to_csv(LOG_PATH, index=False)

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
    print("EXP-012 correlations Pearson:", pearson.loc["exp012"].to_dict())
    print("EXP-012 correlations Spearman:", spearman.loc["exp012"].to_dict())
    print("Best pair:", describe(best_pair))
    print("Best triple:", describe(best_triple))
    print("Multi-model tested/results:", multi_tested, five_tested, [describe(item) for item in multi_candidates])
    print("Best probability:", describe(best_probability))
    print("Best rank:", describe(best_rank))
    print("Winner:", describe(winner))
    print("Improvement vs EXP-012:", improvement)
    print("Std change:", std_change)
    print("Submission generated:", qualifies)


if __name__ == "__main__":
    main()
