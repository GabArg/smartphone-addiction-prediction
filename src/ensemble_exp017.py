"""EXP-017: OOF-only ensemble search centered on EXP-016; no model training."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ensemble_exp009 import (
    blend, fold_metrics, normalized_rank, refine_weights, search_weights, simplex_weights,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
PREDICTIONS = OUTPUTS / "predictions"
SUBMISSIONS = OUTPUTS / "submissions"
METRICS = OUTPUTS / "metrics"
DATA = ROOT / "data"

MODELS = ["exp016", "exp015", "exp012", "exp003", "exp006", "exp008", "exp010"]
OOF_PATHS = {
    "exp016": PREDICTIONS / "oof_exp016_xgboost_depth5_9000.csv",
    "exp015": PREDICTIONS / "oof_exp015_xgboost_depth5.csv",
    "exp012": PREDICTIONS / "oof_exp012_xgboost_thresholds.csv",
    "exp003": PREDICTIONS / "oof_exp003_catboost.csv",
    "exp006": PREDICTIONS / "oof_exp006_lightgbm_features.csv",
    "exp008": PREDICTIONS / "oof_exp008_xgboost.csv",
    "exp010": PREDICTIONS / "oof_exp010_xgboost_features.csv",
}
TEST_PATHS = {
    "exp016": PREDICTIONS / "test_exp016_xgboost_depth5_9000.csv",
    "exp015": PREDICTIONS / "test_exp015_xgboost_depth5.csv",
    "exp012": PREDICTIONS / "test_exp012_xgboost_thresholds.csv",
    "exp003": SUBMISSIONS / "submission_exp003_catboost_4000.csv",
    "exp006": PREDICTIONS / "test_exp006_lightgbm_features.csv",
    "exp008": PREDICTIONS / "test_exp008_xgboost.csv",
    "exp010": PREDICTIONS / "test_exp010_xgboost_features.csv",
}
PAIR_SETS = [
    ("exp016", "exp003"), ("exp016", "exp006"), ("exp016", "exp015"),
    ("exp016", "exp012"), ("exp016", "exp008"), ("exp016", "exp010"),
]
TRIPLE_SETS = [
    ("exp016", "exp003", "exp006"),
    ("exp016", "exp003", "exp015"),
    ("exp016", "exp003", "exp012"),
    ("exp016", "exp006", "exp015"),
    ("exp016", "exp003", "exp010"),
    ("exp016", "exp003", "exp008"),
]
SUBMISSION_PATH = SUBMISSIONS / "submission_exp017_ensemble.csv"
METRICS_PATH = METRICS / "exp017_ensemble_metrics.txt"
LOG_PATH = METRICS / "experiment_log.csv"
NEAR_TIE = 0.00002


def load_oof() -> pd.DataFrame:
    aligned: pd.DataFrame | None = None
    expected_rows: int | None = None
    for model in MODELS:
        frame = pd.read_csv(OOF_PATHS[model])
        if frame.columns.tolist() != ["id", "y_true", "oof_prediction"]:
            raise ValueError(f"Esquema OOF inesperado para {model}.")
        if frame.isna().any().any() or frame["id"].duplicated().any():
            raise ValueError(f"NaN o IDs duplicados en OOF {model}.")
        expected_rows = len(frame) if expected_rows is None else expected_rows
        if len(frame) != expected_rows:
            raise ValueError(f"Cantidad OOF distinta para {model}.")
        renamed = frame.rename(columns={"y_true": f"y_{model}", "oof_prediction": model})
        aligned = renamed if aligned is None else aligned.merge(
            renamed, on="id", how="inner", validate="one_to_one", sort=False
        )
    assert aligned is not None and expected_rows is not None
    if len(aligned) != expected_rows:
        raise ValueError("IDs OOF no coinciden.")
    target = aligned["y_exp016"]
    for model in MODELS[1:]:
        if not target.equals(aligned[f"y_{model}"]):
            raise ValueError(f"y_true no coincide para {model}.")
    return aligned[["id", "y_exp016", *MODELS]].rename(columns={"y_exp016": "y_true"})


def load_test(sample: pd.DataFrame) -> pd.DataFrame:
    aligned = sample[["id"]].copy()
    for model in MODELS:
        frame = pd.read_csv(TEST_PATHS[model])
        prediction_column = "addicted_label" if model == "exp003" else "prediction"
        if frame.columns.tolist() != ["id", prediction_column]:
            raise ValueError(f"Esquema test inesperado para {model}.")
        if frame.isna().any().any() or frame["id"].duplicated().any():
            raise ValueError(f"NaN o IDs duplicados en test {model}.")
        aligned = aligned.merge(
            frame.rename(columns={prediction_column: model}),
            on="id", how="left", validate="one_to_one", sort=False,
        )
    if len(aligned) != len(sample) or aligned.isna().any().any():
        raise ValueError("Alineacion test incompleta.")
    return aligned


def search_pair(y: np.ndarray, predictions: dict[str, np.ndarray],
                models: tuple[str, str]) -> dict[str, object]:
    weights = [(w / 100, 1 - w / 100) for w in range(101)]
    auc, best = search_weights(y, predictions, models, weights)
    return {"models": models, "weights": best, "auc": auc}


def search_refined(y: np.ndarray, predictions: dict[str, np.ndarray],
                   models: tuple[str, ...]) -> dict[str, object]:
    coarse_auc, coarse = search_weights(y, predictions, models, simplex_weights(len(models), 5))
    fine_auc, fine = search_weights(y, predictions, models, refine_weights(coarse))
    return {
        "models": models, "weights": fine, "auc": fine_auc,
        "coarse_auc": coarse_auc, "coarse_weights": coarse,
    }


def make_candidate(name: str, method: str, result: dict[str, object],
                   source: dict[str, np.ndarray], y: np.ndarray) -> dict[str, object]:
    models = tuple(result["models"])
    weights = tuple(result["weights"])
    prediction = blend(source, models, weights)
    folds, std = fold_metrics(y, prediction)
    return {
        "name": name, "method": method, "models": models, "weights": weights,
        "auc": float(result["auc"]), "prediction": prediction,
        "fold_scores": folds, "fold_std": std,
        "complexity": sum(w > 0 for w in weights),
        "exp016_weight": weights[models.index("exp016")] if "exp016" in models else 0.0,
    }


def choose(candidates: list[dict[str, object]]) -> dict[str, object]:
    best_auc = max(float(c["auc"]) for c in candidates)
    near = [c for c in candidates if best_auc - float(c["auc"]) < NEAR_TIE]
    return min(
        near,
        key=lambda c: (
            float(c["fold_std"]), int(c["complexity"]), -float(c["exp016_weight"]),
            0 if c["method"] == "probability" else 1,
        ),
    )


def validate_submission(frame: pd.DataFrame, sample: pd.DataFrame) -> None:
    if frame.columns.tolist() != ["id", "addicted_label"]:
        raise ValueError("Columnas submission invalidas.")
    if len(frame) != 296_302 or len(frame) != len(sample):
        raise ValueError("Cantidad submission invalida.")
    if frame.isna().any().any() or not frame["addicted_label"].between(0, 1).all():
        raise ValueError("NaN o valores fuera de [0,1].")
    if not frame["id"].equals(sample["id"]):
        raise ValueError("IDs submission no coinciden con sample.")


def update_log(winner: dict[str, object], qualifies: bool) -> None:
    columns = [
        "experiment_id", "datetime", "model", "features", "cv_strategy",
        "cv_roc_auc", "kaggle_score", "notes",
    ]
    log = pd.read_csv(LOG_PATH, dtype=str, keep_default_na=False)
    if log.columns.tolist() != columns or (log["experiment_id"] == "EXP-016").sum() != 1:
        raise ValueError("Estado inesperado de experiment_log.csv.")
    log.loc[log["experiment_id"] == "EXP-016", "kaggle_score"] = "0.96730"
    if qualifies:
        log = log.loc[log["experiment_id"] != "EXP-017"].copy()
        models = "/".join(winner["models"])
        weights = "/".join(f"{w:.2f}" for w in winner["weights"])
        row = pd.DataFrame([{
            "experiment_id": "EXP-017",
            "datetime": datetime.now().astimezone().isoformat(timespec="seconds"),
            "model": "Ensemble_with_EXP016",
            "features": "existing_model_predictions",
            "cv_strategy": "OOF_ensemble_optimization",
            "cv_roc_auc": f"{winner['auc']:.6f}",
            "kaggle_score": "",
            "notes": f"{winner['method']}; models={models}; weights={weights}",
        }])
        log = pd.concat([log, row], ignore_index=True)
    log.to_csv(LOG_PATH, index=False)


def main() -> None:
    oof = load_oof()
    y = oof["y_true"].to_numpy()
    probabilities = {model: oof[model].to_numpy(dtype=np.float64) for model in MODELS}
    ranks = {model: normalized_rank(probabilities[model]) for model in MODELS}
    individual = {model: float(roc_auc_score(y, probabilities[model])) for model in MODELS}
    pearson = oof[MODELS].corr(method="pearson")
    spearman = oof[MODELS].corr(method="spearman")

    baseline_folds, baseline_std = fold_metrics(y, probabilities["exp016"])
    baseline = {
        "name": "exp016", "method": "probability", "models": ("exp016",),
        "weights": (1.0,), "auc": individual["exp016"],
        "prediction": probabilities["exp016"], "fold_scores": baseline_folds,
        "fold_std": baseline_std, "complexity": 1, "exp016_weight": 1.0,
    }

    pair_results = [search_pair(y, probabilities, models) for models in PAIR_SETS]
    pair_candidates = [
        make_candidate(f"prob_pair_{'_'.join(r['models'])}", "probability", r, probabilities, y)
        for r in pair_results
    ]
    triple_results = [search_refined(y, probabilities, models) for models in TRIPLE_SETS]
    triple_candidates = [
        make_candidate(f"prob_triple_{'_'.join(r['models'])}", "probability", r, probabilities, y)
        for r in triple_results
    ]
    best_pair = max(pair_candidates, key=lambda c: c["auc"])
    best_triple = max(triple_candidates, key=lambda c: c["auc"])

    four_candidates: list[dict[str, object]] = []
    four_tested = float(best_triple["auc"]) >= individual["exp016"] + 0.00005
    if four_tested:
        for extra in [m for m in MODELS if m not in best_triple["models"]]:
            models = (*best_triple["models"], extra)
            result = search_refined(y, probabilities, models)
            four_candidates.append(
                make_candidate(f"prob_four_{'_'.join(models)}", "probability", result, probabilities, y)
            )

    probability_candidates = [*pair_candidates, *triple_candidates, *four_candidates]
    best_probability = max(probability_candidates, key=lambda c: c["auc"])
    top_probability = sorted(probability_candidates, key=lambda c: c["auc"], reverse=True)[:3]

    rank_candidates: list[dict[str, object]] = []
    seen: set[tuple[str, ...]] = set()
    for probability_candidate in top_probability:
        models = tuple(probability_candidate["models"])
        if models in seen:
            continue
        seen.add(models)
        center = tuple(probability_candidate["weights"])
        auc, weights = search_weights(y, ranks, models, refine_weights(center))
        result = {"models": models, "weights": weights, "auc": auc}
        rank_candidates.append(
            make_candidate(f"rank_{'_'.join(models)}", "rank", result, ranks, y)
        )
    best_rank = max(rank_candidates, key=lambda c: c["auc"])
    winner = choose([baseline, *probability_candidates, *rank_candidates])

    delta = float(winner["auc"]) - individual["exp016"]
    fold_deltas = np.asarray(winner["fold_scores"]) - np.asarray(baseline_folds)
    improved_folds = int(sum(fold_deltas > 0))
    std_change = float(winner["fold_std"]) - baseline_std
    if delta >= 0.00010:
        qualifies = True
        decision = "Mejora >= +0.00010: generar submission."
    elif delta >= 0.00003 and improved_folds >= 4 and std_change <= 0.00005:
        qualifies = True
        decision = "Mejora marginal, >=4 folds y std controlado: generar submission."
    else:
        qualifies = False
        decision = "No supera criterios: no generar submission."

    sample = pd.read_csv(DATA / "sample_submission.csv")
    if qualifies:
        test = load_test(sample)
        if winner["method"] == "probability":
            test_source = {m: test[m].to_numpy(dtype=np.float64) for m in MODELS}
        else:
            test_source = {m: normalized_rank(test[m].to_numpy(dtype=np.float64)) for m in MODELS}
        final_prediction = blend(test_source, tuple(winner["models"]), tuple(winner["weights"]))
        submission = pd.DataFrame({"id": sample["id"], "addicted_label": final_prediction})
        validate_submission(submission, sample)
        SUBMISSIONS.mkdir(parents=True, exist_ok=True)
        submission.to_csv(SUBMISSION_PATH, index=False)
        validate_submission(pd.read_csv(SUBMISSION_PATH), sample)

    def summary(c: dict[str, object]) -> str:
        return (
            f"{c['name']}: method={c['method']}; models={c['models']}; "
            f"weights={c['weights']}; auc={c['auc']:.10f}; "
            f"folds={c['fold_scores']}; std={c['fold_std']:.10f}"
        )

    best_four = max(four_candidates, key=lambda c: c["auc"]) if four_candidates else None
    lines = [
        "EXP-017 OOF ensemble optimization (no training; no leaderboard optimization)",
        f"individual_aucs: {individual}",
        "pearson_matrix:", pearson.to_string(),
        "spearman_matrix:", spearman.to_string(),
        "pair_results:", *(summary(c) for c in pair_candidates),
        "triple_results:", *(summary(c) for c in triple_candidates),
        f"four_model_tested: {four_tested}",
        *( ["four_model_results:", *(summary(c) for c in four_candidates)] if four_tested else []),
        "rank_results:", *(summary(c) for c in rank_candidates),
        f"baseline: {summary(baseline)}",
        f"best_pair: {summary(best_pair)}",
        f"best_triple: {summary(best_triple)}",
        f"best_four: {summary(best_four) if best_four else 'not_tested'}",
        f"best_probability: {summary(best_probability)}",
        f"best_rank: {summary(best_rank)}",
        f"winner: {summary(winner)}",
        f"winner_delta_vs_exp016: {delta:+.10f}",
        f"winner_fold_deltas_vs_exp016: {fold_deltas.tolist()}",
        f"winner_std_change_vs_exp016: {std_change:+.10f}",
        f"decision: {decision}",
        f"submission_generated: {qualifies}",
        f"submission_path: {SUBMISSION_PATH if qualifies else 'none'}",
        "problems: none",
    ]
    METRICS.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    update_log(winner, qualifies)

    print(f"Individual={individual}", flush=True)
    print("Pearson:\n" + pearson.to_string(), flush=True)
    print("Spearman:\n" + spearman.to_string(), flush=True)
    print("Best pair=" + summary(best_pair), flush=True)
    print("Best triple=" + summary(best_triple), flush=True)
    print("Best four=" + (summary(best_four) if best_four else "not tested"), flush=True)
    print("Best probability=" + summary(best_probability), flush=True)
    print("Best rank=" + summary(best_rank), flush=True)
    print("Winner=" + summary(winner), flush=True)
    print(f"Delta={delta:+.10f}; fold deltas={fold_deltas}; std change={std_change:+.10f}", flush=True)
    print(f"Decision={decision}; submission={SUBMISSION_PATH if qualifies else 'none'}", flush=True)


if __name__ == "__main__":
    main()
