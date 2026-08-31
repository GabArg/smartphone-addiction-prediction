"""Statistical threshold diagnostic; does not train models or create submissions."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.project_paths import PROJECT_ROOT


TRAIN_PATH = PROJECT_ROOT / "data" / "train.csv"
METRICS_PATH = PROJECT_ROOT / "outputs" / "metrics" / "threshold_diagnostic.txt"
GRID_PATH = PROJECT_ROOT / "outputs" / "reports" / "threshold_grid_results.csv"

TARGET = "addicted_label"
SCREEN = "daily_screen_time_hours"
SOCIAL = "social_media_hours"
CURRENT_RULE = (6.0, 8.0, 4.0)

SCREEN_THRESHOLDS = np.arange(5.0, 9.0001, 0.25)
SOCIAL_THRESHOLDS = np.arange(3.0, 5.0001, 0.25)
LOWER_THRESHOLDS = np.arange(5.0, 7.0001, 0.25)
UPPER_THRESHOLDS = np.arange(7.0, 9.0001, 0.25)


class Reporter:
    def __init__(self) -> None:
        self.buffer = StringIO()

    def write(self, value: object = "") -> None:
        text = str(value)
        print(text)
        self.buffer.write(text + "\n")

    def section(self, title: str) -> None:
        self.write("\n" + "=" * 100)
        self.write(title)
        self.write("=" * 100)


def rule_metrics(
    screen: np.ndarray,
    social: np.ndarray,
    target: np.ndarray,
    lower: float,
    upper: float,
    social_threshold: float,
) -> dict[str, float | int]:
    positive = (screen > upper) | (social > social_threshold)
    negative = (screen <= lower) & (social <= social_threshold)
    ambiguous = ~(positive | negative)
    positive_count = int(positive.sum())
    negative_count = int(negative.sum())
    ambiguous_count = int(ambiguous.sum())
    total = len(target)
    false_positives = int(((target == 0) & positive).sum())
    false_negatives = int(((target == 1) & negative).sum())
    assigned = positive_count + negative_count
    positive_purity = float(target[positive].mean()) if positive_count else np.nan
    negative_purity = float((target[negative] == 0).mean()) if negative_count else np.nan
    positive_coverage = positive_count / total
    negative_coverage = negative_count / total
    return {
        "lower_screen": lower,
        "upper_screen": upper,
        "social_threshold": social_threshold,
        "valid_rows": total,
        "positive_count": positive_count,
        "positive_coverage": positive_coverage,
        "positive_purity": positive_purity,
        "negative_count": negative_count,
        "negative_coverage": negative_coverage,
        "negative_purity": negative_purity,
        "ambiguous_count": ambiguous_count,
        "ambiguous_coverage": ambiguous_count / total,
        "ambiguous_target_rate": float(target[ambiguous].mean()) if ambiguous_count else np.nan,
        "contradictory_false_positives": false_positives,
        "contradictory_false_negatives": false_negatives,
        "contradictory_count": false_positives + false_negatives,
        "assigned_coverage": assigned / total,
        "contradiction_rate": (false_positives + false_negatives) / assigned if assigned else np.nan,
        "rule_score": positive_coverage * positive_purity + negative_coverage * negative_purity,
    }


def individual_diagnostics(frame: pd.DataFrame, column: str, thresholds: np.ndarray) -> pd.DataFrame:
    valid = frame[[column, TARGET]].dropna()
    values = valid[column].to_numpy()
    target = valid[TARGET].to_numpy(dtype=np.int8)
    rows = []
    for threshold in thresholds:
        above = values > threshold
        below = ~above
        rows.append(
            {
                "variable": column,
                "threshold": threshold,
                "valid_rows": len(valid),
                "gt_count": int(above.sum()),
                "gt_coverage": float(above.mean()),
                "gt_positive_purity": float(target[above].mean()) if above.any() else np.nan,
                "le_count": int(below.sum()),
                "le_coverage": float(below.mean()),
                "le_negative_purity": float((target[below] == 0).mean()) if below.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def positive_rule_grid(screen: np.ndarray, social: np.ndarray, target: np.ndarray) -> pd.DataFrame:
    rows = []
    total_positives = int((target == 1).sum())
    total_negatives = int((target == 0).sum())
    for upper in UPPER_THRESHOLDS:
        for social_threshold in SOCIAL_THRESHOLDS:
            zone = (screen > upper) | (social > social_threshold)
            count = int(zone.sum())
            true_positive = int(((target == 1) & zone).sum())
            false_positive = int(((target == 0) & zone).sum())
            rows.append(
                {
                    "upper_screen": upper,
                    "social_threshold": social_threshold,
                    "coverage": count / len(target),
                    "positive_precision": true_positive / count if count else np.nan,
                    "positive_recall": true_positive / total_positives,
                    "false_positive_rate": false_positive / total_negatives,
                    "count": count,
                    "target_mean": float(target[zone].mean()) if count else np.nan,
                }
            )
    return pd.DataFrame(rows)


def negative_rule_grid(screen: np.ndarray, social: np.ndarray, target: np.ndarray) -> pd.DataFrame:
    rows = []
    total_positives = int((target == 1).sum())
    total_negatives = int((target == 0).sum())
    for lower in LOWER_THRESHOLDS:
        for social_threshold in SOCIAL_THRESHOLDS:
            zone = (screen <= lower) & (social <= social_threshold)
            count = int(zone.sum())
            true_negative = int(((target == 0) & zone).sum())
            false_negative = int(((target == 1) & zone).sum())
            rows.append(
                {
                    "lower_screen": lower,
                    "social_threshold": social_threshold,
                    "coverage": count / len(target),
                    "negative_precision": true_negative / count if count else np.nan,
                    "negative_recall": true_negative / total_negatives,
                    "false_negative_rate": false_negative / total_positives,
                    "count": count,
                    "target_mean": float(target[zone].mean()) if count else np.nan,
                }
            )
    return pd.DataFrame(rows)


def format_rules(frame: pd.DataFrame, rows: int = 20) -> str:
    columns = [
        "lower_screen", "upper_screen", "social_threshold", "assigned_coverage",
        "positive_purity", "negative_purity", "ambiguous_coverage",
        "contradiction_rate", "rule_score",
    ]
    return frame[columns].head(rows).to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )


def fold_diagnostics(
    valid_frame: pd.DataFrame, selected: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_rows = []
    dummy = np.zeros(len(valid_frame))
    y = valid_frame[TARGET].to_numpy(dtype=np.int8)
    for fold, (_, validation_indices) in enumerate(cv.split(dummy, y), start=1):
        fold_frame = valid_frame.iloc[validation_indices]
        screen = fold_frame[SCREEN].to_numpy()
        social = fold_frame[SOCIAL].to_numpy()
        target = fold_frame[TARGET].to_numpy(dtype=np.int8)
        for _, rule in selected.iterrows():
            metrics = rule_metrics(
                screen, social, target,
                float(rule["lower_screen"]),
                float(rule["upper_screen"]),
                float(rule["social_threshold"]),
            )
            metrics["fold"] = fold
            fold_rows.append(metrics)
    folds = pd.DataFrame(fold_rows)
    metric_columns = [
        "assigned_coverage", "positive_purity", "negative_purity",
        "ambiguous_coverage", "contradiction_rate", "rule_score",
    ]
    summary_rows = []
    group_columns = ["lower_screen", "upper_screen", "social_threshold"]
    for keys, group in folds.groupby(group_columns, sort=False):
        row = dict(zip(group_columns, keys))
        for metric in metric_columns:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=0)
        row["folds_better_contradiction_than_current"] = np.nan
        summary_rows.append(row)
    return folds, pd.DataFrame(summary_rows)


def main() -> None:
    start = perf_counter()
    reporter = Reporter()
    train = pd.read_csv(TRAIN_PATH, usecols=[TARGET, SCREEN, SOCIAL])
    valid = train[[TARGET, SCREEN, SOCIAL]].dropna().reset_index(drop=True)
    screen = valid[SCREEN].to_numpy()
    social = valid[SOCIAL].to_numpy()
    target = valid[TARGET].to_numpy(dtype=np.int8)

    reporter.write("THRESHOLD DIAGNOSTIC - PREDICTING SMARTPHONE ADDICTION")
    reporter.write(f"Train total: {len(train):,}; filas válidas screen+social+target: {len(valid):,}")
    reporter.write("No se imputaron missing y no se entrenó ningún modelo.")

    screen_individual = individual_diagnostics(train, SCREEN, SCREEN_THRESHOLDS)
    social_individual = individual_diagnostics(train, SOCIAL, SOCIAL_THRESHOLDS)
    reporter.section("1. THRESHOLDS INDIVIDUALES - SCREEN")
    reporter.write(screen_individual.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    reporter.section("2. THRESHOLDS INDIVIDUALES - SOCIAL")
    reporter.write(social_individual.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    positive_grid = positive_rule_grid(screen, social, target)
    negative_grid = negative_rule_grid(screen, social, target)
    reporter.section("3. MEJORES REGLAS POSITIVAS POR PRECISIÓN (coverage >= 0.10)")
    reporter.write(
        positive_grid.loc[positive_grid["coverage"] >= 0.10]
        .sort_values(["positive_precision", "coverage"], ascending=[False, False])
        .head(20).to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )
    reporter.section("4. MEJORES REGLAS NEGATIVAS POR PRECISIÓN (coverage >= 0.10)")
    reporter.write(
        negative_grid.loc[negative_grid["coverage"] >= 0.10]
        .sort_values(["negative_precision", "coverage"], ascending=[False, False])
        .head(20).to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )

    rows = []
    for lower in LOWER_THRESHOLDS:
        for upper in UPPER_THRESHOLDS:
            if lower >= upper:
                continue
            for social_threshold in SOCIAL_THRESHOLDS:
                rows.append(rule_metrics(screen, social, target, lower, upper, social_threshold))
    grid = pd.DataFrame(rows)
    nontrivial = grid.loc[grid["assigned_coverage"] >= 0.30].copy()
    nontrivial["contradiction_rank"] = nontrivial["contradiction_rate"].rank(method="min")
    nontrivial["score_rank"] = nontrivial["rule_score"].rank(method="min", ascending=False)
    nontrivial["coverage_rank"] = nontrivial["assigned_coverage"].rank(method="min", ascending=False)
    nontrivial["aggregate_rank"] = (
        nontrivial["contradiction_rank"]
        + nontrivial["score_rank"]
        + nontrivial["coverage_rank"]
    )
    current = grid.loc[
        grid["lower_screen"].eq(CURRENT_RULE[0])
        & grid["upper_screen"].eq(CURRENT_RULE[1])
        & grid["social_threshold"].eq(CURRENT_RULE[2])
    ].iloc[0]

    reporter.section("5. TOP 20 - MENOR CONTRADICTION RATE (coverage asignada >= 30%)")
    reporter.write(format_rules(nontrivial.sort_values(["contradiction_rate", "assigned_coverage"], ascending=[True, False])))
    reporter.section("6. TOP 20 - MAYOR RULE SCORE (coverage asignada >= 30%)")
    reporter.write(format_rules(nontrivial.sort_values(["rule_score", "contradiction_rate"], ascending=[False, True])))
    high_purity_floor = min(float(current["positive_purity"]), float(current["negative_purity"]))
    high_purity = nontrivial.loc[
        (nontrivial["positive_purity"] >= high_purity_floor)
        & (nontrivial["negative_purity"] >= high_purity_floor)
    ]
    reporter.section(
        f"7. TOP 20 - MAYOR COBERTURA CON AMBAS PUREZAS >= {high_purity_floor:.4f}"
    )
    reporter.write(format_rules(high_purity.sort_values(["assigned_coverage", "contradiction_rate"], ascending=[False, True])))

    top10 = nontrivial.sort_values(
        ["aggregate_rank", "contradiction_rate", "rule_score"],
        ascending=[True, True, False],
    ).head(10).copy()
    current_frame = pd.DataFrame([current])
    stability_rules = pd.concat([top10, current_frame], ignore_index=True).drop_duplicates(
        ["lower_screen", "upper_screen", "social_threshold"]
    )
    folds, stability = fold_diagnostics(valid, stability_rules)
    current_folds = folds.loc[
        folds["lower_screen"].eq(CURRENT_RULE[0])
        & folds["upper_screen"].eq(CURRENT_RULE[1])
        & folds["social_threshold"].eq(CURRENT_RULE[2])
    ].sort_values("fold")
    for index, row in stability.iterrows():
        candidate_folds = folds.loc[
            folds["lower_screen"].eq(row["lower_screen"])
            & folds["upper_screen"].eq(row["upper_screen"])
            & folds["social_threshold"].eq(row["social_threshold"])
        ].sort_values("fold")
        stability.loc[index, "folds_better_contradiction_than_current"] = int(
            (candidate_folds["contradiction_rate"].to_numpy()
             < current_folds["contradiction_rate"].to_numpy()).sum()
        )

    reporter.section("8. REGLA ACTUAL 6 / 8 / 4")
    reporter.write(format_rules(current_frame, rows=1))
    reporter.write("Métricas por fold:")
    reporter.write(
        current_folds[["fold", "assigned_coverage", "positive_purity", "negative_purity", "contradiction_rate", "rule_score"]]
        .to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )
    reporter.section("9. TOP 10 ALTERNATIVAS POR RANKING AGREGADO")
    reporter.write(format_rules(top10, rows=10))
    reporter.section("10. ESTABILIDAD DE TOP 10 + REGLA ACTUAL")
    reporter.write(stability.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    current_assigned = float(current["assigned_coverage"])
    eligible = stability.loc[
        ~(
            stability["lower_screen"].eq(CURRENT_RULE[0])
            & stability["upper_screen"].eq(CURRENT_RULE[1])
            & stability["social_threshold"].eq(CURRENT_RULE[2])
        )
        & (stability["assigned_coverage_mean"] >= current_assigned - 0.02)
        & (stability["contradiction_rate_mean"] < current_folds["contradiction_rate"].mean())
        & (stability["folds_better_contradiction_than_current"] >= 4)
    ].sort_values(["contradiction_rate_mean", "rule_score_mean"], ascending=[True, False])

    if eligible.empty:
        recommendation = (
            "A) Mantener 6 / 8 / 4: ninguna alternativa evaluada reduce contradiction rate "
            "de forma consistente en >=4 folds conservando cobertura dentro de 2 pp."
        )
    else:
        best = eligible.iloc[0]
        close = eligible.loc[
            eligible["contradiction_rate_mean"] <= best["contradiction_rate_mean"] + 0.0005
        ].head(2)
        variants = ", ".join(
            f"{row.lower_screen:g}/{row.upper_screen:g}/{row.social_threshold:g}"
            for row in close.itertuples()
        )
        recommendation = (
            f"B/C) Probar sólo estas variantes estables en un futuro XGBoost: {variants}. "
            "Reducen contradiction rate en >=4 folds y mantienen cobertura comparable; "
            "no ampliar a una gran familia de thresholds."
        )
    reporter.section("11. RECOMENDACIÓN")
    reporter.write(recommendation)

    GRID_PATH.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(GRID_PATH, index=False)
    elapsed = perf_counter() - start
    reporter.write(f"\nTiempo total: {elapsed:.2f} s")
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(reporter.buffer.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
