"""Diagnose weak OOF bands for EXP-016 without training or producing submissions."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.project_paths import PROJECT_ROOT
from src.train_xgboost_exp012_threshold_features import add_threshold_features


ROOT = PROJECT_ROOT
DATA = ROOT / "data"
PRED = ROOT / "outputs" / "predictions"
REPORT = ROOT / "outputs" / "reports" / "weak_bands_results.csv"
METRICS = ROOT / "outputs" / "metrics" / "weak_bands_diagnostic.txt"
MODELS = {
    "EXP-016": PRED / "oof_exp016_xgboost_depth5_9000.csv",
    "EXP-022": PRED / "oof_exp022_catboost_thresholds_9000.csv",
    "EXP-019": PRED / "oof_exp019_lightgbm_thresholds.csv",
    "EXP-003": PRED / "oof_exp003_catboost.csv",
    "EXP-006": PRED / "oof_exp006_lightgbm_features.csv",
    "EXP-008": PRED / "oof_exp008_xgboost.csv",
}
SECONDARY_FEATURES = [
    "weekend_screen_time", "notifications_per_day", "app_opens_per_day",
    "work_study_hours", "gaming_hours", "sleep_hours", "age",
]
MIN_ROWS = 5_000


def safe_auc(y: pd.Series, prediction: pd.Series) -> float:
    valid = y.notna() & prediction.notna()
    return float(roc_auc_score(y[valid], prediction[valid])) if y[valid].nunique() == 2 else float("nan")


def band_metrics(frame: pd.DataFrame, mask: pd.Series, source: str, definition: str,
                 global_auc: float, global_logloss: float) -> dict[str, object]:
    part = frame.loc[mask]
    y, p = part["addicted_label"], part["EXP-016"]
    target_rate = float(y.mean())
    pred_mean = float(p.mean())
    auc = safe_auc(y, p)
    loss = float(log_loss(y, p, labels=[0, 1]))
    gap = abs(target_rate - pred_mean)
    weak_reasons = []
    if np.isfinite(auc) and auc < global_auc - 0.05:
        weak_reasons.append("low_auc")
    if loss > global_logloss + 0.10:
        weak_reasons.append("high_logloss")
    if gap > 0.05:
        weak_reasons.append("high_calibration_gap")
    if abs(target_rate - 0.5) <= 0.10:
        weak_reasons.append("mixed_classes")
    weak = len(part) >= MIN_ROWS and bool(weak_reasons)
    severity = (
        max(0.0, global_auc - auc) if np.isfinite(auc) else 0.0
    ) + max(0.0, loss - global_logloss) + gap + max(0.0, 0.10 - abs(target_rate - 0.5))
    return {
        "source": source, "definition": definition, "rows": len(part),
        "target_rate": target_rate, "prediction_mean": pred_mean, "auc_exp016": auc,
        "log_loss": loss, "calibration_gap": gap, "weak_candidate": weak,
        "weak_reasons": "+".join(weak_reasons), "severity": severity,
    }


def add_partition(results: list[dict[str, object]], frame: pd.DataFrame, labels: pd.Series,
                  source: str, global_auc: float, global_logloss: float) -> None:
    for label in labels.dropna().unique():
        mask = labels.eq(label)
        results.append(band_metrics(frame, mask, source, str(label), global_auc, global_logloss))


def load_and_align() -> pd.DataFrame:
    train = pd.read_csv(DATA / "train.csv")
    if train["id"].duplicated().any():
        raise ValueError("IDs duplicados en train.")
    aligned = train.copy()
    for name, path in MODELS.items():
        oof = pd.read_csv(path)
        if oof.columns.tolist() != ["id", "y_true", "oof_prediction"]:
            raise ValueError(f"Esquema OOF inesperado: {name}")
        if oof.isna().any().any() or oof["id"].duplicated().any():
            raise ValueError(f"NaN o IDs duplicados en {name}")
        aligned = aligned.merge(
            oof.rename(columns={"y_true": f"y_{name}", "oof_prediction": name}),
            on="id", how="inner", validate="one_to_one", sort=False,
        )
        if len(aligned) != len(train) or not aligned["addicted_label"].equals(aligned[f"y_{name}"]):
            raise ValueError(f"Filas, IDs o y_true no coinciden para {name}")
        aligned.drop(columns=f"y_{name}", inplace=True)
    if aligned[list(MODELS)].isna().any().any():
        raise ValueError("Predicciones OOF con NaN tras alinear.")
    return aligned


def feature_diagnostics(frame: pd.DataFrame, mask: pd.Series) -> list[dict[str, object]]:
    output = []
    part = frame.loc[mask]
    for feature in SECONDARY_FEATURES:
        row: dict[str, object] = {"feature": feature}
        for target in (0, 1):
            values = part.loc[part["addicted_label"].eq(target), feature]
            row[f"mean_y{target}"] = float(values.mean())
            row[f"median_y{target}"] = float(values.median())
        valid = part[feature].notna()
        raw_auc = safe_auc(part.loc[valid, "addicted_label"], part.loc[valid, feature])
        row["oriented_auc"] = max(raw_auc, 1.0 - raw_auc) if np.isfinite(raw_auc) else float("nan")
        output.append(row)
    return sorted(output, key=lambda r: r["oriented_auc"], reverse=True)


def main() -> None:
    start = perf_counter()
    frame = load_and_align()
    y, p = frame["addicted_label"], frame["EXP-016"]
    global_auc = float(roc_auc_score(y, p))
    global_loss = float(log_loss(y, p))
    global_brier = float(brier_score_loss(y, p))
    target_rate = float(y.mean())

    threshold = add_threshold_features(frame.drop(columns=[*MODELS.keys(), "addicted_label"]))
    for region in ("clear_positive_zone", "clear_negative_zone", "ambiguous_zone"):
        frame[region] = threshold[region].to_numpy()

    results: list[dict[str, object]] = []
    quantile_edges = [0, .05, .10, .20, .30, .40, .50, .60, .70, .80, .90, .95, 1]
    qlabels = [f"q{int(a*100):02d}-{int(b*100):02d}" for a, b in zip(quantile_edges[:-1], quantile_edges[1:])]
    quantile_bins = pd.qcut(p, q=quantile_edges, labels=qlabels, duplicates="drop")
    add_partition(results, frame, quantile_bins, "oof_quantile", global_auc, global_loss)

    fixed_bins = pd.cut(p, bins=np.linspace(0, 1, 11), include_lowest=True, right=True)
    add_partition(results, frame, fixed_bins, "oof_fixed_probability", global_auc, global_loss)

    for region in ("clear_positive_zone", "clear_negative_zone", "ambiguous_zone"):
        region_mask = frame[region].eq(1)
        tertiles = pd.Series(pd.NA, index=frame.index, dtype="object")
        tertiles.loc[region_mask] = pd.qcut(
            p.loc[region_mask], q=3, labels=["low", "middle", "high"], duplicates="drop"
        ).astype("object")
        for label in ("low", "middle", "high"):
            mask = region_mask & tertiles.eq(label)
            results.append(band_metrics(frame, mask, "region_prediction_tertile",
                                        f"{region} & prediction_tertile={label}", global_auc, global_loss))

    screen_edges = [-np.inf, 3, 4, 5, 6, 6.5, 7, 7.5, 8, 9, np.inf]
    screen_labels = ["screen<3", "3<=screen<4", "4<=screen<5", "5<=screen<6",
                     "6<=screen<6.5", "6.5<=screen<7", "7<=screen<7.5",
                     "7.5<=screen<8", "8<=screen<9", "screen>=9"]
    add_partition(results, frame, pd.cut(frame["daily_screen_time_hours"], screen_edges,
                  labels=screen_labels, right=False), "screen_band", global_auc, global_loss)

    social_edges = [-np.inf, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, np.inf]
    social_labels = ["social<1", "1<=social<1.5", "1.5<=social<2", "2<=social<2.5",
                     "2.5<=social<3", "3<=social<3.5", "3.5<=social<4",
                     "4<=social<4.5", "4.5<=social<5", "social>=5"]
    add_partition(results, frame, pd.cut(frame["social_media_hours"], social_edges,
                  labels=social_labels, right=False), "social_band", global_auc, global_loss)

    weekend = pd.qcut(frame["weekend_screen_time"], q=10, duplicates="drop")
    add_partition(results, frame, weekend, "weekend_decile", global_auc, global_loss)

    grid_screen = pd.cut(frame["daily_screen_time_hours"], [-np.inf, 4, 5, 6, 7, 8, np.inf],
                         labels=["screen<=4", "4<screen<=5", "5<screen<=6", "6<screen<=7",
                                 "7<screen<=8", "screen>8"], right=True)
    grid_social = pd.cut(frame["social_media_hours"], [-np.inf, 1, 2, 3, 4, np.inf],
                         labels=["social<=1", "1<social<=2", "2<social<=3", "3<social<=4",
                                 "social>4"], right=True)
    for s in grid_screen.dropna().unique():
        for m in grid_social.dropna().unique():
            mask = grid_screen.eq(s) & grid_social.eq(m)
            if not mask.any():
                continue
            results.append(band_metrics(frame, mask, "screen_social_grid", f"{s} & {m}",
                                        global_auc, global_loss))

    result = pd.DataFrame(results)
    weak_indices = result.index[result["weak_candidate"]].tolist()
    model_columns = []
    for name in MODELS:
        column = f"auc_{name.lower().replace('-', '')}"
        model_columns.append(column)
        result[column] = np.nan
    weak_details: dict[int, list[dict[str, object]]] = {}
    for idx in weak_indices:
        row = result.loc[idx]
        # Reconstruct each mask from a stable lookup saved during generation.
        # Definitions are unique within their source, so regenerate via matching metrics masks below.
        source, definition = row["source"], row["definition"]
        if source == "oof_quantile": mask = quantile_bins.astype(str).eq(definition)
        elif source == "oof_fixed_probability": mask = fixed_bins.astype(str).eq(definition)
        elif source == "screen_band": mask = pd.cut(frame["daily_screen_time_hours"], screen_edges, labels=screen_labels, right=False).astype(str).eq(definition)
        elif source == "social_band": mask = pd.cut(frame["social_media_hours"], social_edges, labels=social_labels, right=False).astype(str).eq(definition)
        elif source == "weekend_decile": mask = weekend.astype(str).eq(definition)
        elif source == "screen_social_grid":
            left, right = definition.split(" & ")
            mask = grid_screen.astype(str).eq(left) & grid_social.astype(str).eq(right)
        else:
            region, tertile_text = definition.split(" & prediction_tertile=")
            region_mask = frame[region].eq(1)
            temp = pd.Series(pd.NA, index=frame.index, dtype="object")
            temp.loc[region_mask] = pd.qcut(p.loc[region_mask], 3, labels=["low", "middle", "high"], duplicates="drop").astype("object")
            mask = region_mask & temp.eq(tertile_text)
        aucs = {name: safe_auc(y.loc[mask], frame.loc[mask, name]) for name in MODELS}
        for name, column in zip(MODELS, model_columns): result.loc[idx, column] = aucs[name]
        best_model = max(aucs, key=lambda name: aucs[name] if np.isfinite(aucs[name]) else -np.inf)
        result.loc[idx, "best_model"] = best_model
        result.loc[idx, "best_model_auc"] = aucs[best_model]
        result.loc[idx, "best_alt_delta_vs_exp016"] = aucs[best_model] - aucs["EXP-016"]
        weak_details[idx] = feature_diagnostics(frame, mask)

    result.sort_values(["weak_candidate", "severity", "rows"], ascending=[False, False, False], inplace=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(REPORT, index=False)

    weak = result.loc[result["weak_candidate"]].copy()
    structural = weak.loc[~weak["source"].isin(["oof_quantile", "oof_fixed_probability"])]
    minority_rows = structural["rows"] * np.minimum(structural["target_rate"], 1.0 - structural["target_rate"])
    recommendations = structural.loc[
        (structural["best_alt_delta_vs_exp016"].fillna(-np.inf) > 0) & (minority_rows >= 500)
    ].sort_values(
        ["best_alt_delta_vs_exp016", "rows"], ascending=[False, False]
    )
    chosen = []
    used_sources = set()
    for idx, row in recommendations.iterrows():
        if row["source"] in used_sources: continue
        chosen.append((idx, row)); used_sources.add(row["source"])
        if len(chosen) == 3: break

    def table_text(data: pd.DataFrame, count: int = 10) -> str:
        cols = ["source", "definition", "rows", "target_rate", "prediction_mean", "auc_exp016",
                "log_loss", "calibration_gap", "weak_reasons", "best_model", "best_model_auc",
                "best_alt_delta_vs_exp016"]
        return data.head(count)[cols].to_string(index=False)

    lines = [
        "Weak bands diagnostic; no training; no submissions; no leaderboard optimization",
        f"global_rows: {len(frame)}", f"global_auc_exp016: {global_auc:.10f}",
        f"global_log_loss: {global_loss:.10f}", f"global_brier_score: {global_brier:.10f}",
        f"global_target_rate: {target_rate:.10f}",
        "weak_definition: rows>=5000 and one of AUC<global-0.05, logloss>global+0.10, gap>0.05, target_rate in [0.40,0.60]",
        "worst_oof_prediction_bands:", table_text(weak.loc[weak["source"].isin(["oof_quantile", "oof_fixed_probability"])]),
        "worst_screen_social_bands:", table_text(weak.loc[weak["source"].isin(["screen_band", "social_band", "screen_social_grid"])]),
        "top_structural_weak_bands:", table_text(structural),
        "recommended_bands:",
    ]
    if chosen:
        for rank, (idx, row) in enumerate(chosen, 1):
            lines.append(
                f"H{rank}: {row['source']} | {row['definition']} | rows={int(row['rows'])} | "
                f"EXP-016={row['auc_exp016']:.8f} | best={row['best_model']} "
                f"{row['best_model_auc']:.8f} | delta={row['best_alt_delta_vs_exp016']:+.8f}"
            )
            lines.append("feature_diagnostics: " + str(weak_details[idx]))
    else:
        lines.append("No structural band has a positive alternative-model delta; no hypothesis recommended.")
    elapsed = perf_counter() - start
    lines.extend([f"total_seconds: {elapsed:.2f}", "problems: none"])
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Global: AUC={global_auc:.8f}; logloss={global_loss:.8f}; brier={global_brier:.8f}; target={target_rate:.8f}")
    print("Worst OOF bands:\n" + table_text(weak.loc[weak["source"].isin(["oof_quantile", "oof_fixed_probability"])]))
    print("Worst screen/social:\n" + table_text(weak.loc[weak["source"].isin(["screen_band", "social_band", "screen_social_grid"])]))
    print("Top structural:\n" + table_text(structural))
    print("Recommendations:")
    for rank, (_, row) in enumerate(chosen, 1):
        print(f"H{rank}: {row['source']} | {row['definition']} | rows={int(row['rows'])} | EXP016={row['auc_exp016']:.8f} | {row['best_model']}={row['best_model_auc']:.8f} | delta={row['best_alt_delta_vs_exp016']:+.8f}")
    print(f"Time={elapsed:.2f}s; report={REPORT}; problems=none")


if __name__ == "__main__":
    main()
