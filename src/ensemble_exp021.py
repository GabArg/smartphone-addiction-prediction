"""EXP-021: OOF-only ensemble of EXP-016 and EXP-020; no model training."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from train_xgboost_exp012_threshold_features import add_threshold_features


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
PREDICTIONS = OUTPUTS / "predictions"
SUBMISSIONS = OUTPUTS / "submissions"
METRICS = OUTPUTS / "metrics"

OOF_PATHS = {
    "exp016": PREDICTIONS / "oof_exp016_xgboost_depth5_9000.csv",
    "exp020": PREDICTIONS / "oof_exp020_catboost_thresholds.csv",
    "exp006": PREDICTIONS / "oof_exp006_lightgbm_features.csv",
}
TEST_PATHS = {
    "exp016": PREDICTIONS / "test_exp016_xgboost_depth5_9000.csv",
    "exp020": PREDICTIONS / "test_exp020_catboost_thresholds.csv",
    "exp006": PREDICTIONS / "test_exp006_lightgbm_features.csv",
}
SUBMISSION_PATH = SUBMISSIONS / "submission_exp021_ensemble.csv"
METRICS_PATH = METRICS / "exp021_ensemble_metrics.txt"
LOG_PATH = METRICS / "experiment_log.csv"
EXPECTED = {"exp016": 0.96570173, "exp020": 0.96487898, "pearson": 0.993363}
EXP016_REFERENCE_FOLDS = np.array([
    0.96502029, 0.96566446, 0.96588369, 0.96645502, 0.96548740,
])


def load_oof(models: tuple[str, ...]) -> pd.DataFrame:
    aligned: pd.DataFrame | None = None
    expected_rows: int | None = None
    for model in models:
        frame = pd.read_csv(OOF_PATHS[model])
        if frame.columns.tolist() != ["id", "y_true", "oof_prediction"]:
            raise ValueError(f"Esquema OOF inesperado: {model}")
        if frame.isna().any().any() or frame["id"].duplicated().any():
            raise ValueError(f"NaN o IDs duplicados en OOF: {model}")
        expected_rows = len(frame) if expected_rows is None else expected_rows
        if len(frame) != expected_rows:
            raise ValueError(f"Cantidad de filas OOF diferente: {model}")
        renamed = frame.rename(columns={"y_true": f"y_{model}", "oof_prediction": model})
        aligned = renamed if aligned is None else aligned.merge(
            renamed, on="id", how="inner", validate="one_to_one", sort=False
        )
    assert aligned is not None and expected_rows is not None
    if len(aligned) != expected_rows:
        raise ValueError("Los IDs OOF no coinciden entre modelos.")
    base_y = aligned[f"y_{models[0]}"]
    for model in models[1:]:
        if not base_y.equals(aligned[f"y_{model}"]):
            raise ValueError(f"y_true no coincide: {model}")
    return aligned[["id", f"y_{models[0]}", *models]].rename(
        columns={f"y_{models[0]}": "y_true"}
    )


def load_test(sample: pd.DataFrame, models: tuple[str, ...]) -> pd.DataFrame:
    aligned = sample[["id"]].copy()
    for model in models:
        frame = pd.read_csv(TEST_PATHS[model])
        if frame.columns.tolist() != ["id", "prediction"]:
            raise ValueError(f"Esquema test inesperado: {model}")
        if frame.isna().any().any() or frame["id"].duplicated().any():
            raise ValueError(f"NaN o IDs duplicados en test: {model}")
        aligned = aligned.merge(frame.rename(columns={"prediction": model}), on="id", how="left",
                                validate="one_to_one", sort=False)
    if len(aligned) != len(sample) or aligned.isna().any().any():
        raise ValueError("Alineacion test incompleta.")
    return aligned


def normalized_rank(values: np.ndarray) -> np.ndarray:
    ranks = pd.Series(values).rank(method="average").to_numpy(dtype=np.float64)
    return (ranks - 1.0) / (len(ranks) - 1.0)


def search_pair(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[dict[str, float], list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    for w in np.arange(0.50, 1.00001, 0.01):
        auc = float(roc_auc_score(y, w * a + (1.0 - w) * b))
        rows.append({"stage": "coarse", "w_exp016": float(w), "w_exp020": float(1-w), "auc": auc})
    coarse = max(rows, key=lambda r: (r["auc"], r["w_exp016"]))
    low, high = max(0.50, coarse["w_exp016"] - 0.03), min(1.0, coarse["w_exp016"] + 0.03)
    for w in np.arange(low, high + 0.000001, 0.0025):
        auc = float(roc_auc_score(y, w * a + (1.0 - w) * b))
        rows.append({"stage": "fine", "w_exp016": float(w), "w_exp020": float(1-w), "auc": auc})
    best = max(rows, key=lambda r: (r["auc"], r["w_exp016"]))
    return best, rows


def fold_scores(y: np.ndarray, prediction: np.ndarray) -> tuple[list[float], float]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = [float(roc_auc_score(y[valid], prediction[valid]))
              for _, valid in cv.split(np.zeros(len(y)), y)]
    return scores, float(np.std(scores))


def regional_scores(aligned: pd.DataFrame, prediction: np.ndarray) -> dict[str, dict[str, float | int]]:
    raw = pd.read_csv(DATA / "train.csv")
    threshold = add_threshold_features(raw.drop(columns=["addicted_label"]))
    regions = pd.DataFrame({"id": raw["id"]})
    for name in ["clear_positive_zone", "clear_negative_zone", "ambiguous_zone"]:
        regions[name] = threshold[name]
    merged = aligned[["id", "y_true", "exp016"]].copy()
    merged["ensemble"] = prediction
    merged = merged.merge(regions, on="id", how="left", validate="one_to_one", sort=False)
    output: dict[str, dict[str, float | int]] = {}
    for name in ["clear_positive_zone", "clear_negative_zone", "ambiguous_zone"]:
        part = merged.loc[merged[name].eq(1)]
        if part["y_true"].nunique() < 2:
            auc016 = auc_ens = float("nan")
        else:
            auc016 = float(roc_auc_score(part["y_true"], part["exp016"]))
            auc_ens = float(roc_auc_score(part["y_true"], part["ensemble"]))
        output[name] = {"rows": len(part), "exp016_auc": auc016,
                        "ensemble_auc": auc_ens, "delta": auc_ens - auc016}
    return output


def validate_submission(frame: pd.DataFrame, sample: pd.DataFrame) -> None:
    if frame.columns.tolist() != ["id", "addicted_label"]:
        raise ValueError("Columnas de submission invalidas.")
    if len(frame) != 296_302 or len(frame) != len(sample):
        raise ValueError("Cantidad de filas de submission invalida.")
    if not frame["id"].equals(sample["id"]):
        raise ValueError("IDs de submission no coinciden con sample_submission.")
    if frame.isna().any().any() or not frame["addicted_label"].between(0, 1).all():
        raise ValueError("Submission con NaN o probabilidades fuera de [0,1].")


def update_log(method: str, weights: tuple[float, ...], auc: float, models: tuple[str, ...]) -> None:
    columns = ["experiment_id", "datetime", "model", "features", "cv_strategy",
               "cv_roc_auc", "kaggle_score", "notes"]
    log = pd.read_csv(LOG_PATH, dtype=str, keep_default_na=False)
    if log.columns.tolist() != columns:
        raise ValueError("Esquema inesperado de experiment_log.csv.")
    log = log.loc[log["experiment_id"] != "EXP-021"].copy()
    detail = ", ".join(f"{m}={w:.4f}" for m, w in zip(models, weights))
    row = pd.DataFrame([{
        "experiment_id": "EXP-021",
        "datetime": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": "Ensemble_EXP016_EXP020",
        "features": "existing_model_predictions",
        "cv_strategy": "OOF_ensemble_optimization",
        "cv_roc_auc": f"{auc:.8f}",
        "kaggle_score": "",
        "notes": f"{method}; {detail}",
    }])
    pd.concat([log, row], ignore_index=True).to_csv(LOG_PATH, index=False)


def main() -> None:
    aligned = load_oof(("exp016", "exp020"))
    y = aligned["y_true"].to_numpy()
    p16 = aligned["exp016"].to_numpy(dtype=np.float64)
    p20 = aligned["exp020"].to_numpy(dtype=np.float64)
    auc16 = float(roc_auc_score(y, p16))
    auc20 = float(roc_auc_score(y, p20))
    pearson = float(pd.Series(p16).corr(pd.Series(p20), method="pearson"))
    spearman = float(pd.Series(p16).corr(pd.Series(p20), method="spearman"))
    if abs(auc16 - EXPECTED["exp016"]) > 1e-5 or abs(auc20 - EXPECTED["exp020"]) > 1e-5 or abs(pearson - EXPECTED["pearson"]) > 1e-4:
        raise ValueError("Los controles de AUC/correlacion difieren materialmente de los esperados.")

    prob_best, prob_rows = search_pair(y, p16, p20)
    r16, r20 = normalized_rank(p16), normalized_rank(p20)
    rank_best, rank_rows = search_pair(y, r16, r20)
    candidates = []
    for method, best, a, b in [("probability", prob_best, p16, p20), ("rank", rank_best, r16, r20)]:
        pred = best["w_exp016"] * a + best["w_exp020"] * b
        folds, std = fold_scores(y, pred)
        candidates.append({"method": method, "weights": (best["w_exp016"], best["w_exp020"]),
                           "auc": best["auc"], "prediction": pred, "folds": folds, "std": std})

    best_auc = max(c["auc"] for c in candidates)
    near = [c for c in candidates if best_auc - c["auc"] < 0.00002]
    baseline_folds, baseline_std = fold_scores(y, p16)
    for candidate in near:
        candidate["improved_folds"] = int(sum(np.asarray(candidate["folds"]) > np.asarray(baseline_folds)))
    winner = min(near, key=lambda c: (c["std"], -c["improved_folds"], 0 if c["method"] == "probability" else 1))

    best_two_delta = winner["auc"] - auc16
    third_evaluated = best_two_delta >= 0.00005
    third_result: dict[str, object] | None = None
    if third_evaluated:
        extended = load_oof(("exp016", "exp020", "exp006"))
        if not aligned[["id", "y_true"]].equals(extended[["id", "y_true"]]):
            raise ValueError("EXP-006 no alinea con las OOF principales.")
        p6 = extended["exp006"].to_numpy(dtype=np.float64)
        center20 = winner["weights"][1]
        rows3 = []
        for w6 in np.arange(0.0, 0.15001, 0.02):
            for w20 in np.arange(max(0.0, center20 - 0.06), min(1.0-w6, center20 + 0.06) + 1e-9, 0.02):
                w16 = 1.0 - w20 - w6
                pred = w16*p16 + w20*p20 + w6*p6
                rows3.append((float(roc_auc_score(y, pred)), w16, w20, w6, pred))
        best3 = max(rows3, key=lambda row: (row[0], row[1]))
        folds3, std3 = fold_scores(y, best3[4])
        third_result = {"auc": best3[0], "weights": best3[1:4], "prediction": best3[4],
                        "folds": folds3, "std": std3, "delta_vs_two": best3[0]-winner["auc"]}
        if third_result["delta_vs_two"] >= 0.00002:
            winner = {"method": "probability", "weights": third_result["weights"],
                      "auc": third_result["auc"], "prediction": third_result["prediction"],
                      "folds": third_result["folds"], "std": third_result["std"]}

    delta = winner["auc"] - auc16
    fold_delta = np.asarray(winner["folds"]) - np.asarray(baseline_folds)
    improved_folds = int(sum(fold_delta > 0))
    regions = regional_scores(aligned, winner["prediction"])
    no_clear_region_degradation = all(r["delta"] >= -0.00005 for r in regions.values())
    if delta >= 0.00005:
        qualifies = True
        decision = "Mejora >= +0.00005."
    elif delta >= 0.00002 and improved_folds >= 4 and no_clear_region_degradation:
        qualifies = True
        decision = "Mejora +0.00002 a +0.00005, >=4 folds y sin degradacion regional clara."
    else:
        qualifies = False
        decision = "No supera los criterios de EXP-021."

    models = ("exp016", "exp020", "exp006") if len(winner["weights"]) == 3 else ("exp016", "exp020")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    if qualifies:
        test = load_test(sample, models)
        if winner["method"] == "rank":
            source = {m: normalized_rank(test[m].to_numpy(dtype=np.float64)) for m in models}
        else:
            source = {m: test[m].to_numpy(dtype=np.float64) for m in models}
        test_prediction = sum(w*source[m] for m, w in zip(models, winner["weights"]))
        submission = pd.DataFrame({"id": sample["id"], "addicted_label": test_prediction})
        validate_submission(submission, sample)
        SUBMISSIONS.mkdir(parents=True, exist_ok=True)
        submission.to_csv(SUBMISSION_PATH, index=False)
        validate_submission(pd.read_csv(SUBMISSION_PATH), sample)
        update_log(winner["method"], winner["weights"], winner["auc"], models)

    def candidate_text(c: dict[str, object]) -> str:
        return (f"method={c['method']}; weights={c['weights']}; auc={c['auc']:.10f}; "
                f"folds={c['folds']}; std={c['std']:.10f}")

    lines = [
        "EXP-021 OOF ensemble optimization; no model training; no leaderboard optimization",
        f"auc_exp016: {auc16:.10f}", f"auc_exp020: {auc20:.10f}",
        f"pearson: {pearson:.10f}", f"spearman: {spearman:.10f}",
        "probability_weights_tried:", pd.DataFrame(prob_rows).to_string(index=False),
        "rank_weights_tried:", pd.DataFrame(rank_rows).to_string(index=False),
        f"baseline_folds: {baseline_folds}; baseline_std: {baseline_std:.10f}",
        f"best_probability: {candidate_text(candidates[0])}",
        f"best_rank: {candidate_text(candidates[1])}",
        f"third_model_evaluated: {third_evaluated}",
        f"third_model_result: {third_result}",
        f"winner: {candidate_text(winner)}", f"fold_deltas: {fold_delta.tolist()}",
        f"regions: {regions}", f"delta_vs_exp016: {delta:+.10f}",
        f"optimal_exp020_weight: {winner['weights'][1]:.6f}", f"decision: {decision}",
        f"submission_generated: {qualifies}",
        f"submission_path: {SUBMISSION_PATH if qualifies else 'none'}", "problems: none",
    ]
    METRICS.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"AUC EXP-016={auc16:.10f}; EXP-020={auc20:.10f}")
    print(f"Pearson={pearson:.10f}; Spearman={spearman:.10f}")
    print("Best probability: " + candidate_text(candidates[0]))
    print("Best rank: " + candidate_text(candidates[1]))
    print(f"Baseline folds={baseline_folds}; std={baseline_std:.10f}")
    print("Winner: " + candidate_text(winner))
    print(f"Fold delta={fold_delta.tolist()}; regions={regions}")
    print(f"Delta={delta:+.10f}; third evaluated={third_evaluated}; third result={third_result}")
    print(f"Decision={decision}; submission={SUBMISSION_PATH if qualifies else 'none'}")


if __name__ == "__main__":
    main()
