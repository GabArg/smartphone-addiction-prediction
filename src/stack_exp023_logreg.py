"""EXP-023 diagnostic: nested logistic stacking over existing OOF predictions."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_xgboost_exp012_threshold_features import add_threshold_features


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
PREDICTIONS = OUTPUTS / "predictions"
SUBMISSIONS = OUTPUTS / "submissions"
REPORTS = OUTPUTS / "reports"
METRICS = OUTPUTS / "metrics"
OOF_OUT = PREDICTIONS / "oof_exp023_stack_logreg.csv"
TEST_OUT = PREDICTIONS / "test_exp023_stack_logreg.csv"
SUBMISSION_OUT = SUBMISSIONS / "submission_exp023_stack_logreg.csv"
METRICS_OUT = METRICS / "exp023_stack_logreg_metrics.txt"
COEF_OUT = REPORTS / "exp023_stack_coefficients.csv"
LOG_PATH = METRICS / "experiment_log.csv"

MODELS = ["exp003", "exp006", "exp008", "exp012", "exp016", "exp019", "exp022"]
OOF_PATHS = {
    "exp003": PREDICTIONS / "oof_exp003_catboost.csv",
    "exp006": PREDICTIONS / "oof_exp006_lightgbm_features.csv",
    "exp008": PREDICTIONS / "oof_exp008_xgboost.csv",
    "exp012": PREDICTIONS / "oof_exp012_xgboost_thresholds.csv",
    "exp016": PREDICTIONS / "oof_exp016_xgboost_depth5_9000.csv",
    "exp019": PREDICTIONS / "oof_exp019_lightgbm_thresholds.csv",
    "exp022": PREDICTIONS / "oof_exp022_catboost_thresholds_9000.csv",
}
TEST_PATHS = {
    "exp003": SUBMISSIONS / "submission_exp003_catboost_4000.csv",
    "exp006": PREDICTIONS / "test_exp006_lightgbm_features.csv",
    "exp008": PREDICTIONS / "test_exp008_xgboost.csv",
    "exp012": PREDICTIONS / "test_exp012_xgboost_thresholds.csv",
    "exp016": PREDICTIONS / "test_exp016_xgboost_depth5_9000.csv",
    "exp019": PREDICTIONS / "test_exp019_lightgbm_thresholds.csv",
    "exp022": PREDICTIONS / "test_exp022_catboost_thresholds_9000.csv",
}
EXP020_OOF = PREDICTIONS / "oof_exp020_catboost_thresholds.csv"
CS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
EXP016_AUC = 0.9657017303568276
EXP021_AUC = 0.9658597975

VARIANTS = {
    "META_PROBABILITIES": (MODELS, "probability"),
    "META_LOGITS": (MODELS, "logit"),
    "META_REDUCED": (["exp016", "exp022", "exp012", "exp006"], "probability"),
    "META_2MODEL": (["exp016", "exp022"], "probability"),
}


def load_oof() -> pd.DataFrame:
    aligned: pd.DataFrame | None = None
    expected_rows: int | None = None
    for model in MODELS:
        frame = pd.read_csv(OOF_PATHS[model])
        if frame.columns.tolist() != ["id", "y_true", "oof_prediction"]:
            raise ValueError(f"Esquema OOF inesperado: {model}")
        if frame.isna().any().any() or frame["id"].duplicated().any():
            raise ValueError(f"NaN o IDs duplicados: {model}")
        expected_rows = len(frame) if expected_rows is None else expected_rows
        if len(frame) != expected_rows:
            raise ValueError(f"Longitud OOF diferente: {model}")
        renamed = frame.rename(columns={"y_true": f"y_{model}", "oof_prediction": model})
        aligned = renamed if aligned is None else aligned.merge(
            renamed, on="id", how="inner", validate="one_to_one", sort=False
        )
    assert aligned is not None and expected_rows is not None
    if len(aligned) != expected_rows:
        raise ValueError("Los IDs OOF no coinciden.")
    y = aligned["y_exp003"]
    for model in MODELS[1:]:
        if not y.equals(aligned[f"y_{model}"]):
            raise ValueError(f"y_true no coincide: {model}")
    return aligned[["id", "y_exp003", *MODELS]].rename(columns={"y_exp003": "y_true"})


def load_test(sample: pd.DataFrame) -> pd.DataFrame:
    aligned = sample[["id"]].copy()
    for model in MODELS:
        frame = pd.read_csv(TEST_PATHS[model])
        prediction_col = "addicted_label" if model == "exp003" else "prediction"
        if frame.columns.tolist() != ["id", prediction_col]:
            raise ValueError(f"Esquema test inesperado: {model}")
        if frame.isna().any().any() or frame["id"].duplicated().any():
            raise ValueError(f"Test invalido: {model}")
        aligned = aligned.merge(frame.rename(columns={prediction_col: model}), on="id", how="left",
                                validate="one_to_one", sort=False)
    if len(aligned) != len(sample) or aligned.isna().any().any():
        raise ValueError("Alineacion test incompleta.")
    return aligned


def transform(values: np.ndarray, kind: str) -> np.ndarray:
    if kind == "probability": return values.astype(np.float64, copy=True)
    clipped = np.clip(values, 1e-6, 1-1e-6)
    return np.log(clipped/(1-clipped))


def make_pipeline(c: float) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(penalty="l2", C=c, solver="lbfgs", max_iter=5000, random_state=42)),
    ])


def choose_c(x: np.ndarray, y: np.ndarray) -> tuple[float, dict[float, float]]:
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=123)
    scores: dict[float, float] = {}
    for c in CS:
        fold_scores = []
        for train_idx, valid_idx in cv.split(x, y):
            pipe = make_pipeline(c)
            pipe.fit(x[train_idx], y[train_idx])
            fold_scores.append(roc_auc_score(y[valid_idx], pipe.predict_proba(x[valid_idx])[:, 1]))
        scores[c] = float(np.mean(fold_scores))
    # Prefer stronger regularization when effectively tied.
    best_score = max(scores.values())
    eligible = [c for c, score in scores.items() if best_score-score <= 1e-7]
    return min(eligible), scores


def evaluate_variant(name: str, models: list[str], kind: str, source: pd.DataFrame,
                     splits: list[tuple[np.ndarray, np.ndarray]]) -> tuple[dict[str, object], list[dict[str, object]]]:
    y = source["y_true"].to_numpy(dtype=np.int8)
    x = transform(source[models].to_numpy(dtype=np.float64), kind)
    prediction = np.zeros(len(source), dtype=np.float64)
    chosen_cs, fold_scores, coefficients, inner_scores = [], [], [], []
    for fold, (train_idx, valid_idx) in enumerate(splits, 1):
        c, scores = choose_c(x[train_idx], y[train_idx])
        pipe = make_pipeline(c)
        pipe.fit(x[train_idx], y[train_idx])
        prediction[valid_idx] = pipe.predict_proba(x[valid_idx])[:, 1]
        fold_scores.append(float(roc_auc_score(y[valid_idx], prediction[valid_idx])))
        chosen_cs.append(c)
        coef = pipe.named_steps["model"].coef_[0]
        for model, value in zip(models, coef):
            coefficients.append({"variant": name, "fold": fold, "C": c,
                                 "base_model": model, "coefficient": float(value),
                                 "sign": "+" if value > 0 else "-" if value < 0 else "0"})
        inner_scores.append({"fold": fold, **{str(c0): score for c0, score in scores.items()}})
    result = {
        "name": name, "models": models, "kind": kind, "prediction": prediction,
        "auc": float(roc_auc_score(y, prediction)), "folds": fold_scores,
        "mean": float(np.mean(fold_scores)), "std": float(np.std(fold_scores)),
        "chosen_cs": chosen_cs, "delta_exp016": float(roc_auc_score(y, prediction)-EXP016_AUC),
        "delta_exp021": float(roc_auc_score(y, prediction)-EXP021_AUC),
        "inner_scores": inner_scores,
    }
    return result, coefficients


def coefficient_summary(coefficients: pd.DataFrame, variant: str) -> pd.DataFrame:
    part = coefficients.loc[coefficients["variant"].eq(variant)]
    return part.groupby("base_model", sort=False).agg(
        coefficient_mean=("coefficient", "mean"), coefficient_std=("coefficient", "std"),
        signs=("sign", lambda values: ",".join(values)),
    ).reset_index()


def normalized_rank(values: pd.Series) -> np.ndarray:
    ranks = values.rank(method="average").to_numpy(dtype=np.float64)
    return (ranks-1)/(len(ranks)-1)


def regional(frame: pd.DataFrame, best_prediction: np.ndarray, exp021: np.ndarray) -> dict[str, dict[str, float | int]]:
    raw = pd.read_csv(DATA / "train.csv")
    if not raw["id"].equals(frame["id"]) or not raw["addicted_label"].equals(frame["y_true"]):
        raise ValueError("train no coincide con OOF para analisis regional.")
    engineered = add_threshold_features(raw.drop(columns=["id", "addicted_label"]))
    output = {}
    y = frame["y_true"].to_numpy()
    for region in ("clear_positive_zone", "clear_negative_zone", "ambiguous_zone"):
        mask = engineered[region].eq(1).to_numpy()
        output[region] = {
            "rows": int(mask.sum()),
            "best_meta_auc": float(roc_auc_score(y[mask], best_prediction[mask])),
            "exp016_auc": float(roc_auc_score(y[mask], frame.loc[mask, "exp016"])),
            "exp021_auc": float(roc_auc_score(y[mask], exp021[mask])),
        }
    return output


def update_log(best: dict[str, object]) -> None:
    columns = ["experiment_id", "datetime", "model", "features", "cv_strategy",
               "cv_roc_auc", "kaggle_score", "notes"]
    log = pd.read_csv(LOG_PATH, dtype=str, keep_default_na=False)
    if log.columns.tolist() != columns: raise ValueError("Esquema inesperado de experiment_log.csv")
    log = log.loc[~log["experiment_id"].eq("EXP-023")].copy()
    row = pd.DataFrame([{
        "experiment_id": "EXP-023", "datetime": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": "Nested_Logistic_Stack", "features": "OOF_predictions_base_models",
        "cv_strategy": "Nested_StratifiedKFold", "cv_roc_auc": f"{best['auc']:.8f}",
        "kaggle_score": "", "notes": f"{best['name']}; base_models={','.join(best['models'])}",
    }])
    pd.concat([log, row], ignore_index=True).to_csv(LOG_PATH, index=False)


def main() -> None:
    start = perf_counter()
    oof = load_oof()
    y = oof["y_true"].to_numpy(dtype=np.int8)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(cv.split(np.zeros(len(oof)), y))
    fold_id = np.zeros(len(oof), dtype=np.int8)
    for fold, (_, valid_idx) in enumerate(splits, 1): fold_id[valid_idx] = fold

    pearson = oof[MODELS].corr(method="pearson")
    spearman = oof[MODELS].corr(method="spearman")
    results, coef_rows = [], []
    for name, (models, kind) in VARIANTS.items():
        result, coefficients = evaluate_variant(name, models, kind, oof, splits)
        results.append(result); coef_rows.extend(coefficients)
        print(f"{name}: AUC={result['auc']:.10f}; folds={result['folds']}; C={result['chosen_cs']}", flush=True)
    best = max(results, key=lambda result: result["auc"])
    coefficients = pd.DataFrame(coef_rows)
    best_coef = coefficient_summary(coefficients, best["name"])

    exp020 = pd.read_csv(EXP020_OOF)
    if not exp020["id"].equals(oof["id"]) or not exp020["y_true"].equals(oof["y_true"]):
        raise ValueError("EXP-020 no coincide para reconstruir EXP-021.")
    exp021 = .7125*normalized_rank(oof["exp016"]) + .2875*normalized_rank(exp020["oof_prediction"])
    fixed_control = .7125*oof["exp016"].to_numpy() + .2875*oof["exp022"].to_numpy()
    control_auc = float(roc_auc_score(y, fixed_control))
    exp021_reconstructed_auc = float(roc_auc_score(y, exp021))

    best_logloss = float(log_loss(y, best["prediction"]))
    best_brier = float(brier_score_loss(y, best["prediction"]))
    regions = regional(oof, best["prediction"], exp021)
    c_range = max(best["chosen_cs"])/min(best["chosen_cs"])
    sign_stability = all(group["sign"].value_counts().iloc[0] >= 4
                         for _, group in coefficients.loc[coefficients["variant"].eq(best["name"])].groupby("base_model"))
    stable = c_range <= 100 and sign_stability
    fold_exp021 = [float(roc_auc_score(y[valid], exp021[valid])) for _, valid in splits]
    improvements_vs_exp021 = np.asarray(best["folds"])-np.asarray(fold_exp021)
    improved_folds = int(sum(improvements_vs_exp021 > 0))
    if best["delta_exp021"] >= .00005:
        qualifies = True; decision = "Supera EXP-021 por >=+0.00005: generar submission."
    elif best["delta_exp021"] >= .00002 and improved_folds >= 4 and stable:
        qualifies = True; decision = "Mejora marginal, >=4 folds y coeficientes estables: generar submission."
    else:
        qualifies = False; decision = "No alcanza el criterio: no generar submission ni registrar EXP-023."

    for directory in (PREDICTIONS, SUBMISSIONS, REPORTS, METRICS): directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": oof["id"], "y_true": y, "prediction": best["prediction"], "fold": fold_id}).to_csv(OOF_OUT, index=False)
    coefficients.to_csv(COEF_OUT, index=False)

    if qualifies:
        sample = pd.read_csv(DATA / "sample_submission.csv")
        test = load_test(sample)
        test_predictions = np.zeros((len(test), 5), dtype=np.float64)
        models, kind = best["models"], best["kind"]
        x_oof = transform(oof[models].to_numpy(dtype=np.float64), kind)
        x_test = transform(test[models].to_numpy(dtype=np.float64), kind)
        for column, (train_idx, _) in enumerate(splits):
            pipe = make_pipeline(best["chosen_cs"][column])
            pipe.fit(x_oof[train_idx], y[train_idx])
            test_predictions[:, column] = pipe.predict_proba(x_test)[:, 1]
        test_mean = test_predictions.mean(axis=1, dtype=np.float64)
        test_frame = pd.DataFrame({"id": sample["id"], "prediction": test_mean})
        submission = test_frame.rename(columns={"prediction": "addicted_label"})
        if len(submission) != 296_302 or submission.isna().any().any() or not submission["id"].equals(sample["id"]) or not submission["addicted_label"].between(0, 1).all():
            raise ValueError("Submission invalida.")
        test_frame.to_csv(TEST_OUT, index=False); submission.to_csv(SUBMISSION_OUT, index=False)
        update_log(best)

    elapsed = perf_counter()-start
    def result_text(result: dict[str, object]) -> str:
        return (f"{result['name']}: auc={result['auc']:.10f}; folds={result['folds']}; "
                f"mean={result['mean']:.10f}; std={result['std']:.10f}; "
                f"delta_exp016={result['delta_exp016']:+.10f}; delta_exp021={result['delta_exp021']:+.10f}; "
                f"chosen_C={result['chosen_cs']}; inner_scores={result['inner_scores']}")
    lines = [
        "Nested logistic stacking; fold-safe; no leaderboard optimization",
        "rank_variant: omitted because train-relative percentile mapping to validation/test adds an ambiguous convention",
        "pearson:", pearson.to_string(), "spearman:", spearman.to_string(),
        *(result_text(result) for result in results),
        f"fixed_control_exp016_exp022_auc: {control_auc:.10f}",
        f"exp021_reconstructed_auc: {exp021_reconstructed_auc:.10f}",
        f"best_variant: {best['name']}", f"best_logloss: {best_logloss:.10f}",
        f"best_brier: {best_brier:.10f}", f"regional: {regions}",
        "best_coefficient_summary:", best_coef.to_string(index=False),
        f"C_range_ratio: {c_range}; sign_stable_4of5: {sign_stability}; stable: {stable}",
        f"fold_deltas_vs_reconstructed_exp021: {improvements_vs_exp021.tolist()}",
        f"decision: {decision}", f"submission_generated: {qualifies}",
        f"submission_path: {SUBMISSION_OUT if qualifies else 'none'}", f"total_seconds: {elapsed:.2f}",
        "problems: none",
    ]
    METRICS_OUT.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"Best={best['name']}; AUC={best['auc']:.10f}; deltas EXP016={best['delta_exp016']:+.10f}, EXP021={best['delta_exp021']:+.10f}")
    print(f"Coefficients:\n{best_coef.to_string(index=False)}")
    print(f"Regional={regions}; logloss={best_logloss:.10f}; brier={best_brier:.10f}")
    print(f"Decision={decision}; submission={SUBMISSION_OUT if qualifies else 'none'}")
    print(f"Time={elapsed:.2f}s; problems=none")


if __name__ == "__main__":
    main()
