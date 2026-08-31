"""EXP-024: fold-safe nearest-neighbor and rounded-signature diagnostics."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.project_paths import PROJECT_ROOT
from src.train_xgboost_exp012_threshold_features import add_threshold_features


ROOT = PROJECT_ROOT
DATA = ROOT / "data"
OUT = ROOT / "outputs"
PRED = OUT / "predictions"
REPORT = OUT / "reports"
METRICS = OUT / "metrics"
OOF16 = PRED / "oof_exp016_xgboost_depth5_9000.csv"
OOF22 = PRED / "oof_exp022_catboost_thresholds_9000.csv"
METRICS_OUT = METRICS / "exp024_nearest_neighbors_diagnostic.txt"
CONFIG_OUT = REPORT / "exp024_knn_configs.csv"
SIGNATURE_OUT = REPORT / "exp024_signature_results.csv"
DISTANCE_OUT = REPORT / "exp024_nearest_distance_bands.csv"
OOF_OUT = PRED / "oof_exp024_best_nested_blend.csv"

SPACES = {
    "SPACE-A": ["age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
                "notifications_per_day", "app_opens_per_day", "sleep_hours",
                "work_study_hours", "weekend_screen_time"],
    "SPACE-B": ["daily_screen_time_hours", "social_media_hours", "weekend_screen_time",
                "work_study_hours", "gaming_hours"],
    "SPACE-C": ["daily_screen_time_hours", "social_media_hours", "weekend_screen_time"],
}
METRICS_KNN = ["euclidean", "manhattan"]
KS = [5, 10, 20, 50, 100, 200]
WEIGHTED_KS = [10, 20, 50]
BLEND_WEIGHTS = [0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20]
BATCH = 5_000


def load() -> pd.DataFrame:
    train = pd.read_csv(DATA / "train.csv")
    for feature in sorted(set(sum(SPACES.values(), []))):
        if feature not in train: raise ValueError(f"Feature ausente: {feature}")
    aligned = train.copy()
    for name, path in (("exp016", OOF16), ("exp022", OOF22)):
        frame = pd.read_csv(path)
        if frame.columns.tolist() != ["id", "y_true", "oof_prediction"]:
            raise ValueError(f"Esquema OOF inesperado: {name}")
        if frame.isna().any().any() or frame["id"].duplicated().any():
            raise ValueError(f"OOF invalida: {name}")
        aligned = aligned.merge(frame.rename(columns={"y_true": f"y_{name}", "oof_prediction": name}),
                                on="id", how="inner", validate="one_to_one", sort=False)
        if len(aligned) != len(train) or not aligned["addicted_label"].equals(aligned[f"y_{name}"]):
            raise ValueError(f"IDs/y_true no coinciden: {name}")
        aligned.drop(columns=f"y_{name}", inplace=True)
    return aligned


def safe_auc(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(roc_auc_score(y, prediction)) if np.unique(y).size == 2 else float("nan")


def compute_knn(frame: pd.DataFrame, splits: list[tuple[np.ndarray, np.ndarray]],
                fold_id: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    y = frame["addicted_label"].to_numpy(dtype=np.float64)
    signals: dict[str, np.ndarray] = {}
    nearest: dict[str, np.ndarray] = {}
    for space, columns in SPACES.items():
        for metric in METRICS_KNN:
            prefix = f"{space}|{metric}"
            for k in KS:
                signals[f"{prefix}|mean|k={k}"] = np.zeros(len(frame), dtype=np.float64)
                signals[f"{prefix}|distance_mean|k={k}"] = np.zeros(len(frame), dtype=np.float64)
            for k in WEIGHTED_KS:
                signals[f"{prefix}|weighted|k={k}"] = np.zeros(len(frame), dtype=np.float64)
            nearest[prefix] = np.zeros(len(frame), dtype=np.float64)
            for fold, (train_idx, valid_idx) in enumerate(splits, 1):
                imputer = SimpleImputer(strategy="median")
                scaler = StandardScaler()
                x_train = scaler.fit_transform(imputer.fit_transform(frame.iloc[train_idx][columns]))
                x_valid = scaler.transform(imputer.transform(frame.iloc[valid_idx][columns]))
                nn = NearestNeighbors(n_neighbors=max(KS), metric=metric, algorithm="kd_tree", n_jobs=-1)
                nn.fit(x_train)
                y_train = y[train_idx]
                for start in range(0, len(valid_idx), BATCH):
                    stop = min(start+BATCH, len(valid_idx))
                    distances, indices = nn.kneighbors(x_valid[start:stop], return_distance=True)
                    targets = y_train[indices]
                    positions = valid_idx[start:stop]
                    nearest[prefix][positions] = distances[:, 0]
                    cumulative_targets = np.cumsum(targets, axis=1)
                    cumulative_distances = np.cumsum(distances, axis=1)
                    inv = 1.0/(distances+1e-6)
                    cumulative_weight = np.cumsum(inv, axis=1)
                    cumulative_weighted_target = np.cumsum(inv*targets, axis=1)
                    for k in KS:
                        signals[f"{prefix}|mean|k={k}"][positions] = cumulative_targets[:, k-1]/k
                        signals[f"{prefix}|distance_mean|k={k}"][positions] = -cumulative_distances[:, k-1]/k
                    for k in WEIGHTED_KS:
                        signals[f"{prefix}|weighted|k={k}"][positions] = (
                            cumulative_weighted_target[:, k-1]/cumulative_weight[:, k-1]
                        )
                print(f"{prefix} fold {fold}/5 complete", flush=True)
            signals[f"{prefix}|distance_min|k=1"] = -nearest[prefix].copy()
    return signals, nearest


def config_metrics(frame: pd.DataFrame, signals: dict[str, np.ndarray], fold_id: np.ndarray) -> pd.DataFrame:
    y = frame["addicted_label"].to_numpy()
    p16 = frame["exp016"].to_numpy()
    residual = y-p16
    rows = []
    for name, signal in signals.items():
        space, metric, signal_type, kval = name.split("|")
        folds = [safe_auc(y[fold_id == fold], signal[fold_id == fold]) for fold in range(1, 6)]
        pearson = float(pd.Series(signal).corr(pd.Series(p16), method="pearson"))
        spearman = float(pd.Series(signal).corr(pd.Series(p16), method="spearman"))
        residual_pearson = float(pd.Series(signal).corr(pd.Series(residual), method="pearson"))
        residual_spearman = float(pd.Series(signal).corr(pd.Series(residual), method="spearman"))
        rows.append({
            "config": name, "space": space, "metric": metric, "signal_type": signal_type,
            "k": int(kval.split("=")[1]), "oof_auc": safe_auc(y, signal),
            "fold_mean": float(np.mean(folds)), "fold_std": float(np.std(folds)),
            **{f"fold_{i}_auc": score for i, score in enumerate(folds, 1)},
            "pearson_exp016": pearson, "spearman_exp016": spearman,
            "pearson_residual": residual_pearson, "spearman_residual": residual_spearman,
        })
    return pd.DataFrame(rows).sort_values("oof_auc", ascending=False)


def candidate_pool(configs: pd.DataFrame, subset: np.ndarray | None = None) -> list[str]:
    eligible = configs.loc[configs["signal_type"].isin(["mean", "weighted"])].copy()
    if subset is not None:
        eligible = eligible.assign(selection_auc=subset)
    else:
        eligible = eligible.assign(selection_auc=eligible["oof_auc"])
    top_auc = eligible.nlargest(3, "selection_auc")["config"].tolist()
    remaining = eligible.loc[~eligible["config"].isin(top_auc)].copy()
    diverse = remaining.assign(residual_strength=remaining["pearson_residual"].abs()).nlargest(
        2, "residual_strength"
    )["config"].tolist()
    return top_auc+diverse


def nested_blend(frame: pd.DataFrame, signals: dict[str, np.ndarray], configs: pd.DataFrame,
                 splits: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, list[dict[str, object]]]:
    y = frame["addicted_label"].to_numpy(); p16 = frame["exp016"].to_numpy()
    output = p16.copy(); records = []
    eligible_names = configs.loc[configs["signal_type"].isin(["mean", "weighted"]), "config"].tolist()
    for fold, (train_idx, valid_idx) in enumerate(splits, 1):
        train_aucs = np.array([safe_auc(y[train_idx], signals[name][train_idx]) for name in eligible_names])
        temp = configs.set_index("config").loc[eligible_names].copy()
        # Diversity/residual correlations are recalculated using outer-train only.
        residual = y[train_idx]-p16[train_idx]
        temp["selection_auc"] = train_aucs
        temp["pearson_residual"] = [
            pd.Series(signals[name][train_idx]).corr(pd.Series(residual), method="pearson")
            for name in eligible_names
        ]
        pool = temp.reset_index().pipe(candidate_pool, subset=temp["selection_auc"].to_numpy())
        candidates = []
        baseline_train = safe_auc(y[train_idx], p16[train_idx])
        for name in pool:
            for weight in BLEND_WEIGHTS:
                blend = (1-weight)*p16[train_idx]+weight*signals[name][train_idx]
                candidates.append((safe_auc(y[train_idx], blend), -weight, name, weight))
        best_auc, _, name, weight = max(candidates)
        output[valid_idx] = (1-weight)*p16[valid_idx]+weight*signals[name][valid_idx]
        records.append({"fold": fold, "config": name, "weight": weight,
                        "train_auc": best_auc, "train_delta": best_auc-baseline_train,
                        "valid_auc": safe_auc(y[valid_idx], output[valid_idx]),
                        "valid_delta": safe_auc(y[valid_idx], output[valid_idx])-safe_auc(y[valid_idx], p16[valid_idx])})
    return output, records


def signature_values(frame: pd.DataFrame, signature: str) -> pd.DataFrame:
    if signature == "A_1decimal_5vars":
        cols = ["daily_screen_time_hours", "social_media_hours", "weekend_screen_time", "work_study_hours", "gaming_hours"]
        return frame[cols].round(1)
    if signature == "B_halfunit_5vars":
        cols = ["daily_screen_time_hours", "social_media_hours", "weekend_screen_time", "work_study_hours", "gaming_hours"]
        return (frame[cols]/0.5).round()*0.5
    out = pd.DataFrame(index=frame.index)
    out["daily_screen_time_hours"] = frame["daily_screen_time_hours"].round(0)
    out["social_media_hours"] = (frame["social_media_hours"]/0.5).round()*0.5
    out["weekend_screen_time"] = frame["weekend_screen_time"].round(0)
    return out


def signature_diagnostic(frame: pd.DataFrame, splits: list[tuple[np.ndarray, np.ndarray]],
                         fold_id: np.ndarray) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    y = frame["addicted_label"].to_numpy(); rows=[]; signals={}
    for signature in ["A_1decimal_5vars", "B_halfunit_5vars", "C_mixed_3vars"]:
        values = signature_values(frame, signature)
        signal = np.zeros(len(frame), dtype=np.float64); covered = np.zeros(len(frame), dtype=bool)
        counts_valid = {minimum: 0 for minimum in [5, 10, 20, 50]}
        signature_purity = {minimum: [] for minimum in [5, 10, 20, 50]}
        signature_low = {minimum: [] for minimum in [5, 10, 20, 50]}
        signature_high = {minimum: [] for minimum in [5, 10, 20, 50]}
        signature_counts = {minimum: 0 for minimum in [5, 10, 20, 50]}
        for fold, (train_idx, valid_idx) in enumerate(splits, 1):
            train_values = values.iloc[train_idx].copy(); train_values["target"] = y[train_idx]
            keys = list(values.columns)
            stats = train_values.groupby(keys, dropna=False)["target"].agg(["count", "mean"]).reset_index()
            mapped = values.iloc[valid_idx].merge(stats, on=keys, how="left", sort=False)
            fallback = float(y[train_idx].mean())
            signal[valid_idx] = mapped["mean"].fillna(fallback).to_numpy()
            covered[valid_idx] = mapped["count"].notna().to_numpy()
            for minimum in [5, 10, 20, 50]:
                eligible = stats.loc[stats["count"] >= minimum]
                signature_counts[minimum] += len(eligible)
                signature_purity[minimum].extend(np.maximum(eligible["mean"], 1-eligible["mean"]).tolist())
                signature_low[minimum].extend((eligible["mean"] < .05).tolist())
                signature_high[minimum].extend((eligible["mean"] > .95).tolist())
                counts_valid[minimum] += int(mapped["count"].ge(minimum).sum())
        signals[signature] = signal
        for minimum in [5, 10, 20, 50]:
            mask = np.zeros(len(frame), dtype=bool)
            # Aggregate coverage count is sufficient for report; signal uses all known signatures.
            rows.append({"signature": signature, "min_train_count": minimum,
                         "signature_count_sum_folds": signature_counts[minimum],
                         "valid_rows_covered": counts_valid[minimum],
                         "coverage": counts_valid[minimum]/len(frame),
                         "mean_purity": float(np.mean(signature_purity[minimum])) if signature_purity[minimum] else np.nan,
                         "pct_rate_below_005": float(np.mean(signature_low[minimum])) if signature_low[minimum] else np.nan,
                         "pct_rate_above_095": float(np.mean(signature_high[minimum])) if signature_high[minimum] else np.nan,
                         "global_signal_auc": safe_auc(y, signal), "known_signature_coverage": float(covered.mean())})
    return signals, pd.DataFrame(rows)


def nested_signature(frame: pd.DataFrame, signals: dict[str, np.ndarray],
                     splits: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, list[dict[str, object]]]:
    y=frame["addicted_label"].to_numpy(); p16=frame["exp016"].to_numpy(); output=p16.copy(); records=[]
    for fold,(train_idx,valid_idx) in enumerate(splits,1):
        base=safe_auc(y[train_idx],p16[train_idx]); candidates=[]
        for name,signal in signals.items():
            for weight in BLEND_WEIGHTS:
                score=safe_auc(y[train_idx],(1-weight)*p16[train_idx]+weight*signal[train_idx])
                candidates.append((score,-weight,name,weight))
        score,_,name,weight=max(candidates)
        output[valid_idx]=(1-weight)*p16[valid_idx]+weight*signals[name][valid_idx]
        records.append({"fold":fold,"signature":name,"weight":weight,"train_delta":score-base,
                        "valid_delta":safe_auc(y[valid_idx],output[valid_idx])-safe_auc(y[valid_idx],p16[valid_idx])})
    return output,records


def evaluate(y: np.ndarray, prediction: np.ndarray, fold_id: np.ndarray) -> dict[str, object]:
    folds=[safe_auc(y[fold_id==fold],prediction[fold_id==fold]) for fold in range(1,6)]
    return {"auc":safe_auc(y,prediction),"folds":folds,"mean":float(np.mean(folds)),"std":float(np.std(folds))}


def main() -> None:
    start=perf_counter(); frame=load(); y=frame["addicted_label"].to_numpy(); p16=frame["exp016"].to_numpy()
    cv=StratifiedKFold(5,shuffle=True,random_state=42); splits=list(cv.split(np.zeros(len(frame)),y))
    fold_id=np.zeros(len(frame),dtype=np.int8)
    for fold,(_,valid) in enumerate(splits,1): fold_id[valid]=fold
    signals,nearest=compute_knn(frame,splits,fold_id)
    configs=config_metrics(frame,signals,fold_id); configs.to_csv(CONFIG_OUT,index=False)
    global_top5=candidate_pool(configs)
    knn_nested,knn_records=nested_blend(frame,signals,configs,splits); knn_eval=evaluate(y,knn_nested,fold_id)
    signature_signals,signature_results=signature_diagnostic(frame,splits,fold_id)
    signature_results.to_csv(SIGNATURE_OUT,index=False)
    signature_nested,signature_records=nested_signature(frame,signature_signals,splits)
    signature_eval=evaluate(y,signature_nested,fold_id)
    if knn_eval["auc"]>=signature_eval["auc"]:
        best_type,best_prediction,best_eval="KNN",knn_nested,knn_eval
    else: best_type,best_prediction,best_eval="SIGNATURE",signature_nested,signature_eval

    best_standalone=configs.loc[configs["signal_type"].isin(["mean","weighted"])].iloc[0]
    best_signal=signals[best_standalone["config"]]
    nearest_key="|".join(best_standalone["config"].split("|")[:2])
    distance=nearest[nearest_key]
    distance_bins=pd.qcut(distance,10,duplicates="drop")
    distance_rows=[]
    for band in distance_bins.unique().sort_values():
        mask=distance_bins==band
        distance_rows.append({"distance_band":str(band),"rows":int(mask.sum()),"target_rate":float(y[mask].mean()),
                              "auc_exp016":safe_auc(y[mask],p16[mask]),"auc_best_knn":safe_auc(y[mask],best_signal[mask]),
                              "logloss_exp016":float(log_loss(y[mask],p16[mask])),"distance_mean":float(distance[mask].mean())})
    distance_report=pd.DataFrame(distance_rows); distance_report.to_csv(DISTANCE_OUT,index=False)

    engineered=add_threshold_features(frame.drop(columns=["id","addicted_label","exp016","exp022"]))
    regional={}
    for region in ["clear_positive_zone","clear_negative_zone","ambiguous_zone"]:
        mask=engineered[region].eq(1).to_numpy()
        regional[region]={"rows":int(mask.sum()),"exp016_auc":safe_auc(y[mask],p16[mask]),
                          "best_local_signal_auc":safe_auc(y[mask],best_signal[mask]),
                          "best_nested_auc":safe_auc(y[mask],best_prediction[mask])}
    delta=best_eval["auc"]-safe_auc(y,p16); fold_base=evaluate(y,p16,fold_id)["folds"]
    fold_delta=np.asarray(best_eval["folds"])-np.asarray(fold_base); improved=int(sum(fold_delta>0))
    if delta>=.00005: recommendation="Senal real: recomendar EXP-025."
    elif delta>=.00002 and improved>=4: recommendation="Senal marginal y >=4/5 folds: considerar EXP-025."
    elif delta>=.00002: recommendation="Marginal pero inestable: no recomendar EXP-025."
    elif delta>=0: recommendation="Mejora <+0.00002: no justifica rama KNN/EXP-025."
    else: recommendation="Empeora: descartar rama KNN."
    pd.DataFrame({"id":frame["id"],"y_true":y,"exp016_prediction":p16,
                  "nested_prediction":best_prediction,"fold":fold_id,"method":best_type}).to_csv(OOF_OUT,index=False)
    elapsed=perf_counter()-start
    lines=["EXP-024 nearest-neighbor/signature diagnostic; no training boosters; no submission",
           "SPACE-D omitted: one-hot dimensionality risks brute-force memory/time at 691k rows",
           f"best_standalone: {best_standalone.to_dict()}",f"global_top5: {global_top5}",
           f"knn_nested_records: {knn_records}",f"knn_nested_eval: {knn_eval}",
           f"signature_nested_records: {signature_records}",f"signature_nested_eval: {signature_eval}",
           f"signature_results:\n{signature_results.to_string(index=False)}",
           f"nearest_distance_bands:\n{distance_report.to_string(index=False)}",f"regional: {regional}",
           f"best_type: {best_type}; best_eval: {best_eval}; delta={delta:+.10f}; fold_delta={fold_delta.tolist()}",
           f"recommendation: {recommendation}",f"total_seconds: {elapsed:.2f}","problems: SPACE-D omitted as allowed"]
    METRICS_OUT.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"Best standalone={best_standalone.to_dict()}")
    print(f"Top5={global_top5}")
    print(f"KNN nested={knn_eval}; records={knn_records}")
    print(f"Signature nested={signature_eval}; records={signature_records}")
    print(f"Distance bands:\n{distance_report.to_string(index=False)}")
    print(f"Signatures:\n{signature_results.to_string(index=False)}")
    print(f"Regional={regional}")
    print(f"Best={best_type}; delta={delta:+.10f}; recommendation={recommendation}; time={elapsed:.2f}s")


if __name__=="__main__": main()
