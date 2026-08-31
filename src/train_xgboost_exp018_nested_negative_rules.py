"""EXP-018: nested outer-fold discovery of clear-negative-zone rules."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from train_xgboost_exp008 import MODEL_PARAMS, ordinal_encode_categories, validate_submission
from train_xgboost_exp012_threshold_features import NEW_FEATURES, add_threshold_features


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
PREDICTIONS = OUTPUTS / "predictions"
SUBMISSIONS = OUTPUTS / "submissions"
METRICS = OUTPUTS / "metrics"
REPORTS = OUTPUTS / "reports"
ID = "id"
TARGET = "addicted_label"
SCREEN = "daily_screen_time_hours"
SOCIAL = "social_media_hours"
EXPERIMENT = "EXP-018"

OOF_PATH = PREDICTIONS / "oof_exp018_xgboost_nested_negative.csv"
TEST_PATH = PREDICTIONS / "test_exp018_xgboost_nested_negative.csv"
SUBMISSION_PATH = SUBMISSIONS / "submission_exp018_xgboost_nested_negative.csv"
METRICS_PATH = METRICS / "exp018_xgboost_nested_negative_metrics.txt"
RULES_PATH = REPORTS / "exp018_nested_rules_by_fold.csv"
LOG_PATH = METRICS / "experiment_log.csv"
EXP016_OOF_PATH = PREDICTIONS / "oof_exp016_xgboost_depth5_9000.csv"

MODEL = dict(MODEL_PARAMS)
MODEL.update({"max_depth": 5, "n_estimators": 9000})
EARLY_STOPPING = 300
EXP016_FOLDS = np.array([0.96502029, 0.96566446, 0.96588369, 0.96645502, 0.96548740])
NEW_RULE_FEATURES = [
    "neg_rule_social_weekend", "neg_rule_social_work", "neg_rule_social_screen",
    "negative_rule_count", "negative_rule_any",
]
FAMILIES = {
    "A_social_weekend": {
        "secondary": "weekend_screen_time", "operator": ">",
        "social_quantiles": [.40, .45, .50, .55, .60, .65, .70],
        "secondary_quantiles": [.50, .60, .70, .75, .80],
        "feature": "neg_rule_social_weekend",
    },
    "B_social_work": {
        "secondary": "work_study_hours", "operator": "<=",
        "social_quantiles": [.40, .45, .50, .55, .60, .65, .70],
        "secondary_quantiles": [.25, .35, .50, .65, .75],
        "feature": "neg_rule_social_work",
    },
    "C_social_screen": {
        "secondary": SCREEN, "operator": ">",
        "social_quantiles": [.40, .45, .50, .55, .60, .65, .70],
        "secondary_quantiles": [.50, .60, .70, .75, .80],
        "feature": "neg_rule_social_screen",
    },
}


def exact_negative_zone(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    known = frame[SCREEN].notna() & frame[SOCIAL].notna()
    zone = known & frame[SCREEN].le(6) & frame[SOCIAL].le(4)
    return known, zone


def discover_rules(outer_train: pd.DataFrame, y: pd.Series,
                   fold: int) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    _, zone_mask = exact_negative_zone(outer_train)
    zone = outer_train.loc[zone_mask]
    zone_y = y.loc[zone.index]
    if len(zone) < 1000 or zone_y.nunique() != 2:
        raise ValueError(f"Fold {fold}: zona negativa insuficiente.")
    base_rate = float(zone_y.mean())
    chosen: dict[str, dict[str, object]] = {}
    report_rows: list[dict[str, object]] = []
    for family, definition in FAMILIES.items():
        social_thresholds = sorted(set(
            float(zone[SOCIAL].quantile(q)) for q in definition["social_quantiles"]
        ))
        secondary = str(definition["secondary"])
        secondary_thresholds = sorted(set(
            float(zone[secondary].quantile(q)) for q in definition["secondary_quantiles"]
        ))
        candidates: list[dict[str, object]] = []
        for social_threshold in social_thresholds:
            for secondary_threshold in secondary_thresholds:
                mask = zone[SOCIAL].gt(social_threshold)
                if definition["operator"] == ">":
                    mask &= zone[secondary].gt(secondary_threshold)
                else:
                    mask &= zone[secondary].le(secondary_threshold)
                selected_y = zone_y.loc[mask.fillna(False)]
                coverage = len(selected_y) / len(zone)
                if coverage < .05 or len(selected_y) < 1000:
                    continue
                target_rate = float(selected_y.mean())
                utility = (target_rate - base_rate) * np.sqrt(coverage)
                candidates.append({
                    "fold": fold, "rule_family": family,
                    "social_threshold": social_threshold,
                    "secondary_threshold": secondary_threshold,
                    "secondary_feature": secondary, "secondary_operator": definition["operator"],
                    "coverage": coverage, "base_target_rate": base_rate,
                    "rule_target_rate": target_rate,
                    "lift": target_rate / base_rate, "rule_utility": utility,
                    "rows": len(selected_y), "negative_zone_rows": len(zone),
                    "feature_name": definition["feature"],
                })
        if not candidates:
            raise ValueError(f"Fold {fold}: no hay regla valida para {family}.")
        best = max(candidates, key=lambda row: (row["rule_utility"], row["coverage"]))
        chosen[family] = best
        report_rows.append(best)
    return chosen, report_rows


def add_nested_features(frame: pd.DataFrame,
                        rules: dict[str, dict[str, object]]) -> pd.DataFrame:
    engineered = frame.copy()
    zone_known, in_zone = exact_negative_zone(engineered)
    rule_columns: list[str] = []
    for family, rule in rules.items():
        column = str(rule["feature_name"])
        rule_columns.append(column)
        secondary = str(rule["secondary_feature"])
        result = pd.Series(np.nan, index=engineered.index, dtype=np.float64)
        result.loc[zone_known & ~in_zone] = 0.0
        evaluable = in_zone & engineered[SOCIAL].notna() & engineered[secondary].notna()
        condition = engineered[SOCIAL].gt(float(rule["social_threshold"]))
        if rule["secondary_operator"] == ">":
            condition &= engineered[secondary].gt(float(rule["secondary_threshold"]))
        else:
            condition &= engineered[secondary].le(float(rule["secondary_threshold"]))
        result.loc[evaluable] = condition.loc[evaluable].astype(np.float64)
        engineered[column] = result

    rule_values = engineered[rule_columns]
    engineered["negative_rule_count"] = rule_values.sum(axis=1, min_count=len(rule_columns))
    any_result = pd.Series(np.nan, index=engineered.index, dtype=np.float64)
    any_true = rule_values.eq(1).any(axis=1)
    all_known = rule_values.notna().all(axis=1)
    any_result.loc[any_true] = 1.0
    any_result.loc[~any_true & all_known] = 0.0
    engineered["negative_rule_any"] = any_result
    return engineered


def update_log(mean_auc: float) -> None:
    columns = [
        "experiment_id", "datetime", "model", "features", "cv_strategy",
        "cv_roc_auc", "kaggle_score", "notes",
    ]
    log = pd.read_csv(LOG_PATH, dtype=str, keep_default_na=False)
    if log.columns.tolist() != columns or (log["experiment_id"] == "EXP-017").sum() != 1:
        raise ValueError("Estado inesperado de experiment_log.csv.")
    log.loc[log["experiment_id"] == "EXP-017", "kaggle_score"] = "0.96725"
    log = log.loc[log["experiment_id"] != EXPERIMENT].copy()
    row = pd.DataFrame([{
        "experiment_id": EXPERIMENT,
        "datetime": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": "XGBoost",
        "features": "exp016_plus_nested_negative_zone_rules",
        "cv_strategy": "Nested_rules_with_StratifiedKFold_5",
        "cv_roc_auc": f"{mean_auc:.6f}", "kaggle_score": "",
        "notes": "EXP-016 plus 5 outer-fold-discovered clear-negative-zone rule features",
    }])
    pd.concat([log, row], ignore_index=True).to_csv(LOG_PATH, index=False)


def main() -> None:
    total_start = perf_counter()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    originals = [c for c in train.columns if c not in {ID, TARGET}]
    if test.columns.tolist() != [ID, *originals]:
        raise ValueError("Esquema train/test original no coincide.")
    base_raw = add_threshold_features(train[originals])
    base_test_raw = add_threshold_features(test[originals])
    if base_raw.columns.tolist() != base_test_raw.columns.tolist():
        raise ValueError("Esquema EXP-016 train/test no coincide.")
    if len(base_raw.columns) != len(originals) + len(NEW_FEATURES):
        raise ValueError("Features base distintas de EXP-016.")
    categoricals = [c for c in originals if not pd.api.types.is_numeric_dtype(base_raw[c])]
    numeric = [c for c in base_raw if c not in categoricals]
    X_base, X_test_base, mappings = ordinal_encode_categories(base_raw, base_test_raw, categoricals)
    if not X_base[numeric].equals(base_raw[numeric]):
        raise ValueError("Preprocessing numerico de train fue modificado.")
    if not X_test_base[numeric].equals(base_test_raw[numeric]):
        raise ValueError("Preprocessing numerico de test fue modificado.")
    y = train[TARGET]
    splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_base, y))

    print(f"{EXPERIMENT}: nested rules, rows={len(train)}, base_features={X_base.shape[1]}", flush=True)
    print(f"Model={MODEL}; early_stopping={EARLY_STOPPING}; mappings={mappings}", flush=True)
    oof = np.zeros(len(train), dtype=np.float64)
    test_prediction = np.zeros(len(test), dtype=np.float64)
    scores: list[float] = []
    best_iterations: list[int] = []
    fold_seconds: list[float] = []
    rule_report: list[dict[str, object]] = []
    gains: list[dict[str, float]] = []

    for fold, (train_idx, valid_idx) in enumerate(splits, 1):
        fold_start = perf_counter()
        raw_outer_train = train.iloc[train_idx][originals].copy()
        rules, rows = discover_rules(raw_outer_train, y.iloc[train_idx], fold)
        rule_report.extend(rows)

        train_nested = add_nested_features(X_base.iloc[train_idx], rules)
        valid_nested = add_nested_features(X_base.iloc[valid_idx], rules)
        test_nested = add_nested_features(X_test_base, rules)
        expected = [*X_base.columns, *NEW_RULE_FEATURES]
        if train_nested.columns.tolist() != expected or valid_nested.columns.tolist() != expected:
            raise ValueError(f"Fold {fold}: esquema nested incorrecto.")
        if test_nested.columns.tolist() != expected:
            raise ValueError(f"Fold {fold}: esquema nested test incorrecto.")
        for frame in (train_nested, valid_nested, test_nested):
            if np.isinf(frame[NEW_RULE_FEATURES].to_numpy(dtype=float, na_value=np.nan)).any():
                raise ValueError(f"Fold {fold}: infinitos en features nested.")

        model = XGBClassifier(**MODEL, early_stopping_rounds=EARLY_STOPPING)
        model.fit(
            train_nested, y.iloc[train_idx],
            eval_set=[(valid_nested, y.iloc[valid_idx])], verbose=False,
        )
        best = int(model.best_iteration)
        iteration_range = (0, best + 1)
        valid_prediction = model.predict_proba(valid_nested, iteration_range=iteration_range)[:, 1]
        score = float(roc_auc_score(y.iloc[valid_idx], valid_prediction))
        oof[valid_idx] = valid_prediction.astype(np.float64)
        test_prediction += model.predict_proba(
            test_nested, iteration_range=iteration_range
        )[:, 1].astype(np.float64) / 5.0
        raw_gain = model.get_booster().get_score(importance_type="gain")
        total_gain = float(sum(raw_gain.values()))
        gains.append({feature: raw_gain.get(feature, 0.0) / total_gain for feature in expected})
        elapsed = perf_counter() - fold_start
        scores.append(score)
        best_iterations.append(best)
        fold_seconds.append(elapsed)
        print(f"Fold {fold} rules={rows}", flush=True)
        print(f"Fold {fold} AUC={score:.8f} best={best} time={elapsed:.2f}s", flush=True)

    total_seconds = perf_counter() - total_start
    mean_auc = float(np.mean(scores))
    std_auc = float(np.std(scores))
    overall_auc = float(roc_auc_score(y, oof))
    exp016 = pd.read_csv(EXP016_OOF_PATH)
    if not exp016[ID].equals(train[ID]) or not exp016["y_true"].equals(y):
        raise ValueError("OOF EXP-016 no coincide con train.")
    exp016_global = float(roc_auc_score(y, exp016["oof_prediction"]))
    global_delta = overall_auc - exp016_global
    fold_deltas = np.asarray(scores) - EXP016_FOLDS
    exp016_exact_folds = [
        float(roc_auc_score(y.iloc[v], exp016["oof_prediction"].iloc[v])) for _, v in splits
    ]
    std_delta = std_auc - float(np.std(exp016_exact_folds))

    _, negative_zone = exact_negative_zone(train)
    region_results: dict[str, dict[str, float | int]] = {}
    for region, mask in {"clear_negative_zone": negative_zone, "outside_clear_negative_zone": ~negative_zone}.items():
        region_y = y.loc[mask]
        old_pred = exp016.loc[mask, "oof_prediction"]
        new_pred = pd.Series(oof, index=train.index).loc[mask]
        region_results[region] = {
            "rows": int(mask.sum()),
            "exp016_auc": float(roc_auc_score(region_y, old_pred)),
            "exp018_auc": float(roc_auc_score(region_y, new_pred)),
            "auc_delta": float(roc_auc_score(region_y, new_pred) - roc_auc_score(region_y, old_pred)),
            "exp016_logloss": float(log_loss(region_y, old_pred, labels=[0, 1])),
            "exp018_logloss": float(log_loss(region_y, new_pred, labels=[0, 1])),
        }

    mean_gain = pd.DataFrame(gains).mean().sort_values(ascending=False)
    new_feature_gain = {feature: float(mean_gain.get(feature, 0.0)) for feature in NEW_RULE_FEATURES}
    improved_folds = int(sum(fold_deltas > 0))
    negative_improved = region_results["clear_negative_zone"]["auc_delta"] > 0
    if global_delta >= .00010:
        decision = "Mejora >= +0.00010: recomendar submission."
    elif global_delta >= .00003 and improved_folds >= 4 and negative_improved:
        decision = "Mejora marginal, >=4 folds y mejora zona negativa: recomendar submission."
    elif global_delta >= 0:
        decision = "Empate practico: no recomendar submission."
    else:
        decision = "Empeora: descartar reglas nested."

    rule_frame = pd.DataFrame(rule_report)
    threshold_stability = rule_frame.groupby("rule_family").agg(
        social_threshold_mean=("social_threshold", "mean"),
        social_threshold_std=("social_threshold", "std"),
        secondary_threshold_mean=("secondary_threshold", "mean"),
        secondary_threshold_std=("secondary_threshold", "std"),
        lift_mean=("lift", "mean"), lift_std=("lift", "std"),
        coverage_mean=("coverage", "mean"), coverage_std=("coverage", "std"),
    )
    stable_rules = bool(
        (threshold_stability["lift_std"] < .05).all()
        and (threshold_stability["coverage_std"] < .03).all()
    )
    stability_text = (
        "Thresholds vary reasonably with stable lift/coverage (robust hypothesis)."
        if stable_rules else "Threshold/lift behavior is unstable across folds."
    )

    test_prediction = np.clip(test_prediction, 0, 1)
    oof_frame = pd.DataFrame({ID: train[ID], "y_true": y, "oof_prediction": oof})
    test_frame = pd.DataFrame({ID: sample[ID], "prediction": test_prediction})
    submission = test_frame.rename(columns={"prediction": TARGET})
    if oof_frame.isna().any().any() or not oof_frame["oof_prediction"].between(0, 1).all():
        raise ValueError("OOF EXP-018 invalida.")
    validate_submission(submission, test, sample)
    for directory in (PREDICTIONS, SUBMISSIONS, METRICS, REPORTS):
        directory.mkdir(parents=True, exist_ok=True)
    oof_frame.to_csv(OOF_PATH, index=False)
    test_frame.to_csv(TEST_PATH, index=False)
    submission.to_csv(SUBMISSION_PATH, index=False)
    rule_frame.to_csv(RULES_PATH, index=False)
    validate_submission(pd.read_csv(SUBMISSION_PATH), test, sample)

    lines = [
        f"experiment_id: {EXPERIMENT}",
        f"parameters: {MODEL}; early_stopping_rounds={EARLY_STOPPING}",
        f"base_features_reused: {NEW_FEATURES}", f"new_rule_features: {NEW_RULE_FEATURES}",
        "nested_rule_selection: thresholds and utility selected only on each outer_train",
        "rules_by_fold:", rule_frame.to_string(index=False),
        "threshold_stability:", threshold_stability.to_string(),
        f"stability_assessment: {stability_text}",
        *(f"fold_{i}_roc_auc: {value:.8f}" for i, value in enumerate(scores, 1)),
        f"mean_roc_auc: {mean_auc:.8f}", f"std_roc_auc: {std_auc:.8f}",
        f"overall_oof_roc_auc: {overall_auc:.8f}",
        f"exp016_oof_auc: {exp016_global:.8f}", f"global_delta: {global_delta:+.8f}",
        f"fold_deltas: {fold_deltas.tolist()}", f"std_delta: {std_delta:+.8f}",
        *(f"fold_{i}_best_iteration_zero_based: {value}" for i, value in enumerate(best_iterations, 1)),
        "region_results:", str(region_results),
        f"new_feature_mean_normalized_gain: {new_feature_gain}",
        *(f"fold_{i}_seconds: {value:.2f}" for i, value in enumerate(fold_seconds, 1)),
        f"total_seconds: {total_seconds:.2f}", f"decision: {decision}",
        "submission_validations: OK", "problems: none",
    ]
    METRICS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    update_log(mean_auc)

    print(f"Rules:\n{rule_frame.to_string(index=False)}", flush=True)
    print(f"Stability:\n{threshold_stability.to_string()}\n{stability_text}", flush=True)
    print(f"Scores={scores}; mean={mean_auc:.8f}; std={std_auc:.8f}; overall={overall_auc:.8f}", flush=True)
    print(f"Deltas={fold_deltas.tolist()}; global={global_delta:+.8f}; std_delta={std_delta:+.8f}", flush=True)
    print(f"Regions={region_results}", flush=True)
    print(f"New feature gain={new_feature_gain}", flush=True)
    print(f"Best iterations={best_iterations}; total={total_seconds:.2f}s", flush=True)
    print(f"Decision={decision}", flush=True)
    print(f"Submission={SUBMISSION_PATH}; validations=OK", flush=True)


if __name__ == "__main__":
    main()
