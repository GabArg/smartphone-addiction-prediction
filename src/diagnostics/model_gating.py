"""Fold-safe nested gating diagnostic for EXP-016 and EXP-022; no training."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.project_paths import PROJECT_ROOT


ROOT = PROJECT_ROOT
DATA = ROOT / "data"
PRED = ROOT / "outputs" / "predictions"
REPORTS = ROOT / "outputs" / "reports"
METRICS = ROOT / "outputs" / "metrics"
OOF16 = PRED / "oof_exp016_xgboost_depth5_9000.csv"
OOF22 = PRED / "oof_exp022_catboost_thresholds_9000.csv"
OOF_OUT = PRED / "oof_exp023_nested_gating_diagnostic.csv"
FOLD_OUT = REPORTS / "exp023_nested_gate_by_fold.csv"
METRICS_OUT = METRICS / "exp023_nested_gating_diagnostic.txt"
QUANTILES = [0.10, 0.15, 0.20, 0.25, 0.30, 1/3, 0.35, 0.40, 0.45, 0.50]
CAT_WEIGHTS = [1.0, 0.75, 0.50, 0.25]
REFERENCE_FOLDS = np.array([0.96502029, 0.96566446, 0.96588369, 0.96645502, 0.96548740])


def auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if np.unique(y).size == 2 else float("nan")


def load() -> pd.DataFrame:
    train = pd.read_csv(DATA / "train.csv")
    if train["id"].duplicated().any():
        raise ValueError("IDs duplicados en train.")
    aligned = train.copy()
    for name, path in (("exp016_prediction", OOF16), ("exp022_prediction", OOF22)):
        frame = pd.read_csv(path)
        if frame.columns.tolist() != ["id", "y_true", "oof_prediction"]:
            raise ValueError(f"Esquema OOF inesperado: {path.name}")
        if frame.isna().any().any() or frame["id"].duplicated().any():
            raise ValueError(f"OOF con NaN o IDs duplicados: {path.name}")
        aligned = aligned.merge(
            frame.rename(columns={"y_true": f"y_{name}", "oof_prediction": name}),
            on="id", how="inner", validate="one_to_one", sort=False,
        )
        if len(aligned) != len(train) or not aligned["addicted_label"].equals(aligned[f"y_{name}"]):
            raise ValueError(f"IDs o y_true no coinciden: {path.name}")
        aligned.drop(columns=f"y_{name}", inplace=True)
    if aligned[["exp016_prediction", "exp022_prediction"]].isna().any().any():
        raise ValueError("NaN tras alinear OOF.")
    return aligned


def local_metrics(frame: pd.DataFrame, mask: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    y = frame["addicted_label"].to_numpy()[mask]
    return {
        "rows": int(mask.sum()), "target_rate": float(y.mean()) if len(y) else float("nan"),
        "auc": auc(y, prediction[mask]) if len(y) else float("nan"),
    }


def best_weight(frame: pd.DataFrame, mask: np.ndarray) -> tuple[float, float]:
    y = frame["addicted_label"].to_numpy()[mask]
    p16 = frame["exp016_prediction"].to_numpy()[mask]
    p22 = frame["exp022_prediction"].to_numpy()[mask]
    candidates = []
    for weight in CAT_WEIGHTS:
        score = auc(y, weight * p22 + (1.0 - weight) * p16)
        candidates.append((score, weight))
    # Prefer less CatBoost when local AUC ties.
    score, weight = max(candidates, key=lambda item: (item[0], -item[1]))
    return weight, score


def apply_gate(frame: pd.DataFrame, mask: np.ndarray, weight: float) -> np.ndarray:
    p16 = frame["exp016_prediction"].to_numpy(dtype=np.float64)
    p22 = frame["exp022_prediction"].to_numpy(dtype=np.float64)
    output = p16.copy()
    output[mask] = weight * p22[mask] + (1.0 - weight) * p16[mask]
    return output


def evaluate_prediction(y: np.ndarray, p: np.ndarray, fold: np.ndarray) -> dict[str, object]:
    folds = [auc(y[fold == value], p[fold == value]) for value in range(1, 6)]
    return {
        "auc": auc(y, p), "folds": folds, "std": float(np.std(folds)),
        "logloss": float(log_loss(y, p)), "brier": float(brier_score_loss(y, p)),
    }


def main() -> None:
    start = perf_counter()
    frame = load()
    y = frame["addicted_label"].to_numpy()
    p16 = frame["exp016_prediction"].to_numpy(dtype=np.float64)
    p22 = frame["exp022_prediction"].to_numpy(dtype=np.float64)
    negative = (
        frame["daily_screen_time_hours"].notna()
        & frame["social_media_hours"].notna()
        & frame["daily_screen_time_hours"].le(6)
        & frame["social_media_hours"].le(4)
    ).to_numpy()
    structural = (
        frame["daily_screen_time_hours"].notna()
        & frame["social_media_hours"].notna()
        & frame["daily_screen_time_hours"].gt(5)
        & frame["daily_screen_time_hours"].le(6)
        & frame["social_media_hours"].le(1)
    ).to_numpy()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_id = np.zeros(len(frame), dtype=np.int8)
    splits = list(cv.split(np.zeros(len(frame)), y))
    for fold, (_, valid_idx) in enumerate(splits, 1): fold_id[valid_idx] = fold

    quantile_oof = p16.copy()
    structural_oof = p16.copy()
    quantile_flag = np.zeros(len(frame), dtype=np.int8)
    structural_flag = np.zeros(len(frame), dtype=np.int8)
    rows: list[dict[str, object]] = []
    quantile_selections: list[dict[str, object]] = []
    structural_selections: list[dict[str, object]] = []

    for fold, (train_idx, valid_idx) in enumerate(splits, 1):
        train_mask_base = np.zeros(len(frame), dtype=bool); train_mask_base[train_idx] = True
        valid_mask_base = np.zeros(len(frame), dtype=bool); valid_mask_base[valid_idx] = True
        candidates = []
        negative_train_values = p16[train_mask_base & negative]
        for quantile in QUANTILES:
            threshold = float(np.quantile(negative_train_values, quantile))
            gate_train = train_mask_base & negative & (p16 <= threshold)
            if gate_train.sum() < 5_000 or np.unique(y[gate_train]).size < 2: continue
            auc16_train = auc(y[gate_train], p16[gate_train])
            auc22_train = auc(y[gate_train], p22[gate_train])
            hard_gain = auc22_train - auc16_train
            if hard_gain <= 0: continue
            weight, blend_auc = best_weight(frame, gate_train)
            coverage = gate_train.sum() / int((train_mask_base & negative).sum())
            utility = hard_gain * np.sqrt(coverage)
            candidates.append({
                "quantile": quantile, "threshold": threshold, "gate_train": gate_train,
                "auc16_train": auc16_train, "auc22_train": auc22_train,
                "hard_gain": hard_gain, "weight": weight, "blend_auc": blend_auc,
                "utility": utility, "coverage": coverage,
            })
        selected = max(candidates, key=lambda item: (item["utility"], -item["quantile"])) if candidates else None
        if selected is None:
            gate_valid = np.zeros(len(frame), dtype=bool); q_weight = 0.0
            q_row = {"quantile": np.nan, "threshold": np.nan, "gate_train": np.zeros(len(frame), bool),
                     "auc16_train": np.nan, "auc22_train": np.nan, "weight": 0.0, "coverage": 0.0}
        else:
            q_row = selected; q_weight = float(selected["weight"])
            gate_valid = valid_mask_base & negative & (p16 <= selected["threshold"])
            quantile_oof[gate_valid] = q_weight*p22[gate_valid] + (1-q_weight)*p16[gate_valid]
            quantile_flag[gate_valid] = 1
        gate_train = q_row["gate_train"]
        q_valid_y = y[gate_valid]
        q_valid_blend = q_weight*p22[gate_valid] + (1-q_weight)*p16[gate_valid]
        q_record = {
            "fold": fold, "gate_type": "nested_quantile", "selected_quantile": q_row["quantile"],
            "numeric_threshold": q_row["threshold"], "train_gate_rows": int(gate_train.sum()),
            "valid_gate_rows": int(gate_valid.sum()), "train_gate_target_rate": float(y[gate_train].mean()) if gate_train.any() else np.nan,
            "train_auc_exp016": q_row["auc16_train"], "train_auc_exp022": q_row["auc22_train"],
            "selected_catboost_weight": q_weight,
            "valid_gate_target_rate": float(q_valid_y.mean()) if len(q_valid_y) else np.nan,
            "valid_auc_exp016": auc(q_valid_y, p16[gate_valid]) if len(q_valid_y) else np.nan,
            "valid_auc_exp022": auc(q_valid_y, p22[gate_valid]) if len(q_valid_y) else np.nan,
            "valid_auc_gated": auc(q_valid_y, q_valid_blend) if len(q_valid_y) else np.nan,
            "train_gate_coverage": q_row["coverage"],
        }
        rows.append(q_record); quantile_selections.append(q_record)

        gate_train_s = train_mask_base & structural
        gate_valid_s = valid_mask_base & structural
        s_weight, _ = best_weight(frame, gate_train_s)
        structural_oof[gate_valid_s] = s_weight*p22[gate_valid_s] + (1-s_weight)*p16[gate_valid_s]
        structural_flag[gate_valid_s] = 1
        s_record = {
            "fold": fold, "gate_type": "fixed_structural", "selected_quantile": np.nan,
            "numeric_threshold": np.nan, "train_gate_rows": int(gate_train_s.sum()),
            "valid_gate_rows": int(gate_valid_s.sum()), "train_gate_target_rate": float(y[gate_train_s].mean()),
            "train_auc_exp016": auc(y[gate_train_s], p16[gate_train_s]),
            "train_auc_exp022": auc(y[gate_train_s], p22[gate_train_s]),
            "selected_catboost_weight": s_weight, "valid_gate_target_rate": float(y[gate_valid_s].mean()),
            "valid_auc_exp016": auc(y[gate_valid_s], p16[gate_valid_s]),
            "valid_auc_exp022": auc(y[gate_valid_s], p22[gate_valid_s]),
            "valid_auc_gated": auc(y[gate_valid_s], structural_oof[gate_valid_s]),
            "train_gate_coverage": gate_train_s.sum()/len(train_idx),
        }
        rows.append(s_record); structural_selections.append(s_record)

    baseline = evaluate_prediction(y, p16, fold_id)
    quantile_eval = evaluate_prediction(y, quantile_oof, fold_id)
    structural_eval = evaluate_prediction(y, structural_oof, fold_id)

    q_values = [r["selected_quantile"] for r in quantile_selections if np.isfinite(r["selected_quantile"])]
    q_thresholds = [r["numeric_threshold"] for r in quantile_selections if np.isfinite(r["numeric_threshold"])]
    q_weights = [r["selected_catboost_weight"] for r in quantile_selections]
    quantile_stable = (
        len(q_values) == 5 and max(q_values)-min(q_values) <= .15
        and np.std(q_thresholds) <= .02 and len(set(q_weights)) <= 2
    )
    structural_train_positive = all(r["train_auc_exp022"] > r["train_auc_exp016"] for r in structural_selections)
    structural_valid_positive = sum(r["valid_auc_exp022"] > r["valid_auc_exp016"] for r in structural_selections)
    structural_stable = structural_train_positive and structural_valid_positive >= 4

    combined_eval = None
    if quantile_stable and structural_stable:
        combined = quantile_oof.copy()
        for fold in range(1, 6):
            record = structural_selections[fold-1]
            mask = (fold_id == fold) & structural & (quantile_flag == 0)
            weight = record["selected_catboost_weight"]
            combined[mask] = weight*p22[mask] + (1-weight)*p16[mask]
        combined_eval = evaluate_prediction(y, combined, fold_id)

    candidates = [("nested_quantile", quantile_eval), ("fixed_structural", structural_eval)]
    if combined_eval is not None: candidates.append(("combined_nonoverlap", combined_eval))
    best_name, best_eval = max(candidates, key=lambda item: item[1]["auc"])
    delta = best_eval["auc"] - baseline["auc"]
    fold_deltas = np.asarray(best_eval["folds"])-np.asarray(baseline["folds"])
    improved_folds = int(sum(fold_deltas > 0))
    if delta >= .00005:
        recommendation = "Recomendar crear EXP-023 real con submission."
    elif delta >= .00002 and improved_folds >= 4 and (best_name != "nested_quantile" or quantile_stable):
        recommendation = "Mejora marginal, estable y >=4/5 folds: recomendar EXP-023 con cautela."
    elif delta >= .00002:
        recommendation = "Mejora marginal pero no cumple estabilidad/folds: no recomendar EXP-023."
    elif delta >= 0:
        recommendation = "Mejora <+0.00002: no justifica gating ni EXP-023."
    else:
        recommendation = "El gating empeora: descartar la idea."

    PRED.mkdir(parents=True, exist_ok=True); REPORTS.mkdir(parents=True, exist_ok=True); METRICS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "id": frame["id"], "y_true": y, "exp016_prediction": p16,
        "exp022_prediction": p22, "gated_prediction": quantile_oof,
        "fold": fold_id, "gate_flag": quantile_flag,
    }).to_csv(OOF_OUT, index=False)
    fold_report = pd.DataFrame(rows)
    fold_report.to_csv(FOLD_OUT, index=False)
    elapsed = perf_counter()-start
    lines = [
        "Nested gating diagnostic; OOF-only; no training; no submission; no experiment log update",
        f"baseline: {baseline}", f"nested_quantile: {quantile_eval}",
        f"fixed_structural: {structural_eval}", f"combined_nonoverlap: {combined_eval}",
        f"quantile_stable: {quantile_stable}; quantiles={q_values}; thresholds={q_thresholds}; weights={q_weights}",
        f"structural_stable: {structural_stable}; train_positive={structural_train_positive}; valid_positive_folds={structural_valid_positive}",
        "fold_records:", fold_report.to_string(index=False), f"best_candidate: {best_name}",
        f"best_delta_vs_exp016: {delta:+.10f}", f"best_fold_deltas: {fold_deltas.tolist()}",
        f"recommendation: {recommendation}", f"total_seconds: {elapsed:.2f}", "problems: none",
    ]
    METRICS_OUT.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("Fold records:\n"+fold_report.to_string(index=False))
    print(f"Baseline={baseline}")
    print(f"Quantile={quantile_eval}; stable={quantile_stable}")
    print(f"Structural={structural_eval}; stable={structural_stable}")
    print(f"Combined={combined_eval}")
    print(f"Best={best_name}; delta={delta:+.10f}; recommendation={recommendation}")
    print(f"Time={elapsed:.2f}s; problems=none")


if __name__ == "__main__":
    main()
