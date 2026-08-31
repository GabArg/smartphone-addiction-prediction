"""Diagnostico estadistico de contradicciones dentro de clear_negative_zone."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.project_paths import PROJECT_ROOT


ROOT = PROJECT_ROOT
TRAIN_PATH = ROOT / "data" / "train.csv"
METRICS_PATH = ROOT / "outputs" / "metrics" / "negative_zone_diagnostic.txt"
RULES_PATH = ROOT / "outputs" / "reports" / "negative_zone_rule_candidates.csv"
TARGET = "addicted_label"
SCREEN = "daily_screen_time_hours"
SOCIAL = "social_media_hours"
NUMERIC_FEATURES = [
    "age", SCREEN, SOCIAL, "gaming_hours", "work_study_hours", "sleep_hours",
    "notifications_per_day", "app_opens_per_day", "weekend_screen_time",
]
CATEGORICAL_FEATURES = ["gender", "stress_level", "academic_work_impact"]
RULE_FEATURES = [
    "age", "gaming_hours", "work_study_hours", "sleep_hours",
    "notifications_per_day", "app_opens_per_day", "weekend_screen_time",
]


def describe_numeric(zone: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in NUMERIC_FEATURES:
        for target_value in (0, 1):
            values = zone.loc[zone[TARGET] == target_value, feature]
            valid = values.dropna()
            rows.append({
                "feature": feature, "target": target_value,
                "count": int(valid.size), "mean": float(valid.mean()),
                "median": float(valid.median()), "std": float(valid.std()),
                "q10": float(valid.quantile(.10)), "q25": float(valid.quantile(.25)),
                "q50": float(valid.quantile(.50)), "q75": float(valid.quantile(.75)),
                "q90": float(valid.quantile(.90)), "missing_rate": float(values.isna().mean()),
            })
    return pd.DataFrame(rows)


def numeric_power(zone: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in NUMERIC_FEATURES:
        valid = zone[[feature, TARGET]].dropna()
        if valid[TARGET].nunique() < 2 or valid[feature].nunique() < 2:
            auc = np.nan
        else:
            auc = float(roc_auc_score(valid[TARGET], valid[feature]))
        rows.append({
            "feature": feature, "raw_auc": auc,
            "oriented_auc": max(auc, 1 - auc) if np.isfinite(auc) else np.nan,
            "positive_direction": ">" if auc >= .5 else "<=",
            "valid_rows": len(valid),
        })
    return pd.DataFrame(rows).sort_values("oriented_auc", ascending=False)


def categorical_analysis(zone: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    distributions: list[dict[str, object]] = []
    separations: list[dict[str, object]] = []
    for feature in CATEGORICAL_FEATURES:
        values = zone[feature].fillna("__MISSING__")
        table = pd.DataFrame({feature: values, TARGET: zone[TARGET]}).groupby(feature)[TARGET].agg(
            ["size", "mean"]
        ).reset_index()
        table["proportion"] = table["size"] / len(zone)
        for row in table.itertuples(index=False):
            distributions.append({
                "feature": feature, "category": row[0], "count": int(row[1]),
                "proportion": float(row[3]), "target_rate": float(row[2]),
            })
        separations.append({
            "feature": feature, "min_target_rate": float(table["mean"].min()),
            "max_target_rate": float(table["mean"].max()),
            "target_rate_range": float(table["mean"].max() - table["mean"].min()),
        })
    return pd.DataFrame(distributions), pd.DataFrame(separations).sort_values(
        "target_rate_range", ascending=False
    )


def evaluate_mask(zone: pd.DataFrame, mask: pd.Series, base_rate: float) -> dict[str, object]:
    selected = zone.loc[mask.fillna(False)]
    target_rate = float(selected[TARGET].mean()) if len(selected) else np.nan
    return {
        "rows": len(selected), "coverage": len(selected) / len(zone),
        "target_rate": target_rate, "positive_rate": target_rate,
        "negative_rate": 1 - target_rate if np.isfinite(target_rate) else np.nan,
        "precision_positive": target_rate,
        "precision_negative": 1 - target_rate if np.isfinite(target_rate) else np.nan,
        "positive_lift": target_rate / base_rate if base_rate > 0 and np.isfinite(target_rate) else np.nan,
        "negative_lift": ((1 - target_rate) / (1 - base_rate)
                          if base_rate < 1 and np.isfinite(target_rate) else np.nan),
    }


def univariate_rules(zone: pd.DataFrame, base_rate: float) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for feature in RULE_FEATURES:
        thresholds = sorted(set(float(zone[feature].quantile(q)) for q in np.arange(.1, 1, .1)))
        for threshold in thresholds:
            for operator in (">", "<="):
                mask = zone[feature].gt(threshold) if operator == ">" else zone[feature].le(threshold)
                output.append({
                    "rule_type": "univariate", "rule": f"{feature} {operator} {threshold:.8g}",
                    "feature_a": feature, "operator_a": operator, "threshold_a": threshold,
                    "feature_b": "", "operator_b": "", "threshold_b": np.nan,
                    **evaluate_mask(zone, mask, base_rate),
                })
    return output


def pair_rules(zone: pd.DataFrame, base_rate: float,
               top_features: list[str]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    thresholds = {
        feature: [float(zone[feature].quantile(q)) for q in (.25, .50, .75)]
        for feature in top_features
    }
    operator_pairs = [(">", ">"), (">", "<="), ("<=", ">")]
    for feature_a, feature_b in combinations(top_features, 2):
        for threshold_a in sorted(set(thresholds[feature_a])):
            for threshold_b in sorted(set(thresholds[feature_b])):
                for operator_a, operator_b in operator_pairs:
                    mask_a = (zone[feature_a].gt(threshold_a) if operator_a == ">"
                              else zone[feature_a].le(threshold_a))
                    mask_b = (zone[feature_b].gt(threshold_b) if operator_b == ">"
                              else zone[feature_b].le(threshold_b))
                    output.append({
                        "rule_type": "pair",
                        "rule": (f"{feature_a} {operator_a} {threshold_a:.8g} AND "
                                 f"{feature_b} {operator_b} {threshold_b:.8g}"),
                        "feature_a": feature_a, "operator_a": operator_a,
                        "threshold_a": threshold_a, "feature_b": feature_b,
                        "operator_b": operator_b, "threshold_b": threshold_b,
                        **evaluate_mask(zone, mask_a & mask_b, base_rate),
                    })
    return output


def rule_mask(frame: pd.DataFrame, row: pd.Series) -> pd.Series:
    first = (frame[row.feature_a].gt(row.threshold_a) if row.operator_a == ">"
             else frame[row.feature_a].le(row.threshold_a))
    if row.rule_type == "univariate":
        return first
    second = (frame[row.feature_b].gt(row.threshold_b) if row.operator_b == ">"
              else frame[row.feature_b].le(row.threshold_b))
    return first & second


def fold_stability(train: pd.DataFrame, zone_mask: pd.Series,
                   top_rules: pd.DataFrame) -> pd.DataFrame:
    splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(train, train[TARGET]))
    rows: list[dict[str, object]] = []
    for rank, rule in top_rules.reset_index(drop=True).iterrows():
        for fold, (_, valid_idx) in enumerate(splits, 1):
            fold_frame = train.iloc[valid_idx]
            fold_zone = fold_frame.loc[zone_mask.iloc[valid_idx].to_numpy()].copy()
            base = float(fold_zone[TARGET].mean())
            metrics = evaluate_mask(fold_zone, rule_mask(fold_zone, rule), base)
            rows.append({
                "rule_rank": rank + 1, "rule": rule.rule, "fold": fold,
                "coverage": metrics["coverage"], "target_rate": metrics["target_rate"],
                "positive_lift": metrics["positive_lift"], "rows": metrics["rows"],
            })
    return pd.DataFrame(rows)


def main() -> None:
    start = perf_counter()
    train = pd.read_csv(TRAIN_PATH)
    required = {TARGET, SCREEN, SOCIAL, *NUMERIC_FEATURES, *CATEGORICAL_FEATURES}
    missing = required - set(train.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {sorted(missing)}")
    known = train[SCREEN].notna() & train[SOCIAL].notna()
    zone_mask = known & train[SCREEN].le(6) & train[SOCIAL].le(4)
    zone = train.loc[zone_mask].copy()
    if zone.empty or zone[TARGET].nunique() != 2:
        raise ValueError("clear_negative_zone vacia o con una sola clase.")
    valid_train_rows = int(known.sum())
    base_rate = float(zone[TARGET].mean())

    descriptive = describe_numeric(zone)
    power = numeric_power(zone)
    category_distribution, category_separation = categorical_analysis(zone)
    top_four = power.head(4)["feature"].tolist()

    rules = pd.DataFrame([
        *univariate_rules(zone, base_rate),
        *pair_rules(zone, base_rate, top_four),
    ])
    eligible = rules.loc[(rules["coverage"] >= .05) & (rules["rows"] >= 1000)].copy()
    eligible = eligible.sort_values(
        ["target_rate", "coverage", "rows"], ascending=[False, False, False]
    ).reset_index(drop=True)
    top10 = eligible.head(10).copy()
    stability = fold_stability(train, zone_mask, top10)
    stability_summary = stability.groupby(["rule_rank", "rule"], sort=False).agg(
        coverage_mean=("coverage", "mean"), coverage_std=("coverage", "std"),
        target_rate_mean=("target_rate", "mean"), target_rate_std=("target_rate", "std"),
        lift_mean=("positive_lift", "mean"), lift_std=("positive_lift", "std"),
        min_rows=("rows", "min"),
    ).reset_index()

    # Recommend only stable, interpretable hypotheses with nontrivial lift.
    recommendation_pool = stability_summary.merge(
        top10[["rule", "rule_type", "coverage", "target_rate", "positive_lift"]], on="rule"
    )
    recommendation_pool = recommendation_pool.loc[
        (recommendation_pool["lift_mean"] >= 1.10)
        & (recommendation_pool["coverage_mean"] >= .05)
        & (recommendation_pool["target_rate_std"] <= .02)
    ].sort_values(["lift_mean", "coverage_mean"], ascending=False)
    # Keep at most one representative per feature pair so the final hypotheses
    # are genuinely distinct rather than three nested cutoffs of the same rule.
    recommendation_pool["interaction_key"] = recommendation_pool.apply(
        lambda row: tuple(sorted([row["rule"].split(" ")[0],
                                  row["rule"].split(" AND ")[-1].split(" ")[0]])),
        axis=1,
    )
    hypotheses = recommendation_pool.drop_duplicates("interaction_key").head(3)

    RULES_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    rules_out = eligible.copy()
    rules_out.to_csv(RULES_PATH, index=False)
    elapsed = perf_counter() - start

    class_distribution = zone[TARGET].value_counts().sort_index()
    class_means = descriptive.pivot(index="feature", columns="target", values="mean")
    class_medians = descriptive.pivot(index="feature", columns="target", values="median")
    mean_differences = pd.DataFrame({
        "mean_target0": class_means[0], "mean_target1": class_means[1],
        "mean_difference_1_minus_0": class_means[1] - class_means[0],
        "median_target0": class_medians[0], "median_target1": class_medians[1],
        "median_difference_1_minus_0": class_medians[1] - class_medians[0],
    }).reindex(power["feature"])
    hypotheses_text = (
        "\n".join(
            f"H{i}: {row.rule}; target_rate={row.target_rate:.6f} vs base={base_rate:.6f}; "
            f"coverage={row.coverage:.4f}; fold_lift={row.lift_mean:.4f}±{row.lift_std:.4f}"
            for i, row in enumerate(hypotheses.itertuples(index=False), 1)
        ) if not hypotheses.empty else "No hay hipotesis con señal suficientemente estable."
    )
    report = [
        "Clear negative zone diagnostic",
        "definition: daily_screen_time_hours <= 6 AND social_media_hours <= 4; both values known",
        f"train_rows: {len(train)}", f"valid_zone_determination_rows: {valid_train_rows}",
        f"negative_zone_rows: {len(zone)}",
        f"negative_zone_proportion_of_valid_train: {len(zone)/valid_train_rows:.8f}",
        f"target_0_count: {int(class_distribution.get(0, 0))}",
        f"target_1_count: {int(class_distribution.get(1, 0))}",
        f"target_0_rate: {1-base_rate:.8f}", f"target_1_rate: {base_rate:.8f}",
        "numeric_descriptive_by_class:", descriptive.to_string(index=False),
        "numeric_discriminative_power:", power.to_string(index=False),
        "important_mean_median_differences:", mean_differences.to_string(),
        "categorical_distribution_and_target_rate:", category_distribution.to_string(index=False),
        "categorical_separation:", category_separation.to_string(index=False),
        f"top_four_numeric_for_pairs: {top_four}",
        "top10_rules:", top10.to_string(index=False),
        "top10_fold_details:", stability.to_string(index=False),
        "top10_fold_summary:", stability_summary.to_string(index=False),
        "recommended_hypotheses:", hypotheses_text,
        f"total_seconds: {elapsed:.2f}", "problems: none",
    ]
    METRICS_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Zone rows={len(zone)} proportion_valid={len(zone)/valid_train_rows:.8f}", flush=True)
    print(f"Target rate 1={base_rate:.8f}; rate 0={1-base_rate:.8f}", flush=True)
    print("Power:\n" + power.to_string(index=False), flush=True)
    print("Differences:\n" + mean_differences.to_string(), flush=True)
    print("Top 10:\n" + top10.to_string(index=False), flush=True)
    print("Stability:\n" + stability_summary.to_string(index=False), flush=True)
    print("Hypotheses:\n" + hypotheses_text, flush=True)
    print(f"Time={elapsed:.2f}s", flush=True)


if __name__ == "__main__":
    main()
