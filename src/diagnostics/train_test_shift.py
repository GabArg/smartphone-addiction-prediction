"""EXP-025: adversarial train-vs-test validation and test-likeness diagnosis."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.project_paths import PROJECT_ROOT


ROOT = PROJECT_ROOT
DATA = ROOT / "data"
OUT = ROOT / "outputs"
REPORTS = OUT / "reports"
METRICS = OUT / "metrics"
PREDICTIONS = OUT / "predictions"
METRICS_OUT = METRICS / "exp025_adversarial_validation.txt"
SHIFT_OUT = REPORTS / "exp025_feature_shift.csv"
MISSING_OUT = REPORTS / "exp025_missing_shift.csv"
DIST_OUT = REPORTS / "exp025_distribution_distances.csv"
QUINTILE_OUT = REPORTS / "exp025_adversarial_quintiles.csv"
OOF_OUT = PREDICTIONS / "oof_exp025_adversarial_train.csv"
EXP016_PATH = PREDICTIONS / "oof_exp016_xgboost_depth5_9000.csv"

PARAMS = {
    "objective": "binary", "n_estimators": 3000, "learning_rate": .03,
    "num_leaves": 31, "max_depth": -1, "min_child_samples": 20,
    "subsample": .9, "colsample_bytree": .9, "reg_alpha": 0.0,
    "reg_lambda": 0.0, "random_state": 42, "n_jobs": -1, "verbosity": -1,
}


def aligned_categories(frame: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in categorical:
        values = output[column].astype("string").fillna("__MISSING__")
        output[column] = pd.Categorical(values, categories=pd.unique(values))
    return output


def lgbm_cv(x: pd.DataFrame, y: np.ndarray, categorical: list[str]) -> tuple[np.ndarray, list[float], list[int], pd.DataFrame]:
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    oof = np.zeros(len(x), dtype=np.float64); scores=[]; best=[]; gains=[]; splits=[]
    for fold, (train_idx, valid_idx) in enumerate(cv.split(x, y), 1):
        model = LGBMClassifier(**PARAMS)
        model.fit(x.iloc[train_idx], y[train_idx], categorical_feature=categorical,
                  eval_set=[(x.iloc[valid_idx], y[valid_idx])], eval_metric="auc",
                  callbacks=[early_stopping(200, verbose=False), log_evaluation(0)])
        prediction = model.predict_proba(x.iloc[valid_idx], num_iteration=model.best_iteration_)[:, 1]
        oof[valid_idx] = prediction
        scores.append(float(roc_auc_score(y[valid_idx], prediction))); best.append(int(model.best_iteration_))
        booster = model.booster_
        gains.append(pd.DataFrame({"feature": x.columns,
                                   "gain": booster.feature_importance("gain"),
                                   "split": booster.feature_importance("split")}))
        print(f"LightGBM fold {fold}: AUC={scores[-1]:.8f}; best={best[-1]}", flush=True)
    importance = pd.concat(gains).groupby("feature", as_index=False)[["gain", "split"]].mean()
    importance.sort_values("gain", ascending=False, inplace=True)
    return oof, scores, best, importance


def logistic_cv(x: np.ndarray, y: np.ndarray, scale: bool = False) -> tuple[np.ndarray, list[float]]:
    cv = StratifiedKFold(5, shuffle=True, random_state=42); oof=np.zeros(len(y)); scores=[]
    for train_idx, valid_idx in cv.split(x, y):
        steps=[]
        if scale: steps.append(("scale", StandardScaler()))
        steps.append(("model", LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000)))
        model=Pipeline(steps); model.fit(x[train_idx], y[train_idx])
        oof[valid_idx]=model.predict_proba(x[valid_idx])[:,1]
        scores.append(float(roc_auc_score(y[valid_idx],oof[valid_idx])))
    return oof,scores


def univariate(train: pd.DataFrame, test: pd.DataFrame, features: list[str], categorical: list[str]) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    shifts=[]; missing=[]; distances=[]
    for column in features:
        tr,te=train[column],test[column]
        trm,tem=float(tr.isna().mean()),float(te.isna().mean())
        missing.append({"feature":column,"train_missing_rate":trm,"test_missing_rate":tem,
                        "delta_missing_pp":100*(tem-trm),"abs_delta_missing_pp":100*abs(tem-trm)})
        if column in categorical:
            a=tr.astype("string").fillna("__MISSING__").value_counts(normalize=True)
            b=te.astype("string").fillna("__MISSING__").value_counts(normalize=True)
            cats=a.index.union(b.index); diff=(a.reindex(cats,fill_value=0)-b.reindex(cats,fill_value=0)).abs()
            maxdiff=float(diff.max()); tv=float(.5*diff.sum())
            shifts.append({"feature":column,"type":"categorical","shift_score":maxdiff,
                           "train_mean":np.nan,"test_mean":np.nan,"train_median":np.nan,"test_median":np.nan,
                           "train_std":np.nan,"test_std":np.nan,"standardized_mean_difference":np.nan,
                           "max_category_proportion_difference":maxdiff})
            distances.append({"feature":column,"type":"categorical","ks_statistic":np.nan,
                              "wasserstein_distance":np.nan,"total_variation_distance":tv})
        else:
            trv=tr.dropna().astype(float); tev=te.dropna().astype(float)
            pooled=np.sqrt((trv.var()+tev.var())/2); smd=float((tev.mean()-trv.mean())/pooled) if pooled else 0.0
            shifts.append({"feature":column,"type":"numeric","shift_score":abs(smd),
                           "train_mean":trv.mean(),"test_mean":tev.mean(),"train_median":trv.median(),
                           "test_median":tev.median(),"train_std":trv.std(),"test_std":tev.std(),
                           "standardized_mean_difference":smd,"max_category_proportion_difference":np.nan})
            distances.append({"feature":column,"type":"numeric",
                              "ks_statistic":float(ks_2samp(trv,tev).statistic),
                              "wasserstein_distance":float(wasserstein_distance(trv,tev)),
                              "total_variation_distance":np.nan})
    return (pd.DataFrame(shifts).sort_values("shift_score",ascending=False),
            pd.DataFrame(missing).sort_values("abs_delta_missing_pp",ascending=False),
            pd.DataFrame(distances).assign(max_distance=lambda d:d[["ks_statistic","total_variation_distance"]].max(axis=1)).sort_values("max_distance",ascending=False))


def q1_q5_differences(frame: pd.DataFrame, features: list[str], categorical: list[str]) -> pd.DataFrame:
    q1=frame.loc[frame["adversarial_quintile"].eq("Q1")]; q5=frame.loc[frame["adversarial_quintile"].eq("Q5")]; rows=[]
    for column in features:
        if column in categorical:
            a=q1[column].astype("string").fillna("__MISSING__").value_counts(normalize=True)
            b=q5[column].astype("string").fillna("__MISSING__").value_counts(normalize=True)
            cats=a.index.union(b.index); distance=float(.5*(a.reindex(cats,fill_value=0)-b.reindex(cats,fill_value=0)).abs().sum())
            rows.append({"feature":column,"type":"categorical","difference":distance,"q1_value":np.nan,"q5_value":np.nan})
        else:
            a=q1[column];b=q5[column];pooled=np.sqrt((a.var()+b.var())/2)
            diff=float((b.mean()-a.mean())/pooled) if pooled else 0.0
            rows.append({"feature":column,"type":"numeric","difference":abs(diff),"signed_difference":diff,
                         "q1_value":a.mean(),"q5_value":b.mean()})
    return pd.DataFrame(rows).sort_values("difference",ascending=False).head(15)


def main() -> None:
    start=perf_counter(); train=pd.read_csv(DATA/"train.csv");test=pd.read_csv(DATA/"test.csv")
    features=[c for c in test.columns if c!="id"]
    if [c for c in train.columns if c not in {"id","addicted_label"}]!=features: raise ValueError("Esquema train/test inesperado")
    categorical=[c for c in features if not pd.api.types.is_numeric_dtype(train[c])]
    numerical=[c for c in features if c not in categorical]
    shifts,missing,distances=univariate(train,test,features,categorical)
    combined=pd.concat([train[features],test[features]],ignore_index=True)
    y_adv=np.r_[np.zeros(len(train),dtype=np.int8),np.ones(len(test),dtype=np.int8)]
    x=aligned_categories(combined,categorical)
    adv_oof,adv_folds,best_iterations,importance=lgbm_cv(x,y_adv,categorical)
    adv_auc=float(roc_auc_score(y_adv,adv_oof))

    missing_x=combined[features].isna().astype(np.float64).to_numpy()
    missing_oof,missing_folds=logistic_cv(missing_x,y_adv)
    missing_auc=float(roc_auc_score(y_adv,missing_oof))

    complete_mask=combined[features].notna().all(axis=1).to_numpy()
    complete=aligned_categories(combined.loc[complete_mask,features].reset_index(drop=True),categorical)
    complete_y=y_adv[complete_mask]
    values_oof,values_folds,values_best,_=lgbm_cv(complete,complete_y,categorical)
    values_auc=float(roc_auc_score(complete_y,values_oof))

    id_values=pd.concat([train["id"],test["id"]],ignore_index=True).to_numpy(dtype=np.float64).reshape(-1,1)
    id_oof,id_folds=logistic_cv(id_values,y_adv,scale=True);id_auc=float(roc_auc_score(y_adv,id_oof))
    id_direct=float(roc_auc_score(y_adv,id_values[:,0]));id_direct=max(id_direct,1-id_direct)

    train_score=adv_oof[:len(train)]
    exp16=pd.read_csv(EXP016_PATH)
    if not exp16["id"].equals(train["id"]) or not exp16["y_true"].equals(train["addicted_label"]): raise ValueError("EXP-016 no alinea")
    analysis=train.copy();analysis["adversarial_score"]=train_score
    analysis["adversarial_quintile"]=pd.qcut(train_score,5,labels=["Q1","Q2","Q3","Q4","Q5"])
    analysis["exp016_prediction"]=exp16["oof_prediction"]
    quintiles=[]
    for q in ["Q1","Q2","Q3","Q4","Q5"]:
        part=analysis.loc[analysis["adversarial_quintile"].eq(q)];yt=part["addicted_label"];pred=part["exp016_prediction"]
        quintiles.append({"quintile":q,"rows":len(part),"score_min":part["adversarial_score"].min(),
                          "score_max":part["adversarial_score"].max(),"score_mean":part["adversarial_score"].mean(),
                          "target_rate":yt.mean(),"exp016_auc":roc_auc_score(yt,pred),
                          "exp016_logloss":log_loss(yt,pred),"exp016_brier":brier_score_loss(yt,pred)})
    quintile_df=pd.DataFrame(quintiles)
    differences=q1_q5_differences(analysis,features,categorical)
    error_abs=np.abs(train["addicted_label"].to_numpy()-exp16["oof_prediction"].to_numpy())
    individual_loss=-(train["addicted_label"].to_numpy()*np.log(np.clip(exp16["oof_prediction"],1e-15,1))+
                      (1-train["addicted_label"].to_numpy())*np.log(np.clip(1-exp16["oof_prediction"],1e-15,1)))
    correlations={
        "pearson_error_abs":float(pd.Series(train_score).corr(pd.Series(error_abs),method="pearson")),
        "spearman_error_abs":float(pd.Series(train_score).corr(pd.Series(error_abs),method="spearman")),
        "pearson_logloss":float(pd.Series(train_score).corr(pd.Series(individual_loss),method="pearson")),
        "spearman_logloss":float(pd.Series(train_score).corr(pd.Series(individual_loss),method="spearman")),
    }
    REPORTS.mkdir(parents=True,exist_ok=True);METRICS.mkdir(parents=True,exist_ok=True);PREDICTIONS.mkdir(parents=True,exist_ok=True)
    shifts.to_csv(SHIFT_OUT,index=False);missing.to_csv(MISSING_OUT,index=False);distances.to_csv(DIST_OUT,index=False);quintile_df.to_csv(QUINTILE_OUT,index=False)
    pd.DataFrame({"id":train["id"],"is_test":0,"adversarial_prediction":train_score}).to_csv(OOF_OUT,index=False)
    q_auc_change=abs(quintile_df.loc[quintile_df.quintile.eq("Q5"),"exp016_auc"].iloc[0]-quintile_df.loc[quintile_df.quintile.eq("Q1"),"exp016_auc"].iloc[0])
    missing_explains=missing_auc>=adv_auc-.02
    justify=adv_auc>=.55 and not missing_explains and q_auc_change>=.0005
    if adv_auc<.52: interpretation="Shift minimo."
    elif adv_auc<.55: interpretation="Shift pequeno."
    elif adv_auc<.65: interpretation="Shift real/moderado."
    else: interpretation="Shift importante."
    recommendation="Justifica diagnostico EXP-026 basado en test-likeness." if justify else "No justifica CV weighting ni EXP-026 basado en test-likeness."
    elapsed=perf_counter()-start
    lines=["EXP-025 adversarial validation; no target-model training; no submission",f"train_rows: {len(train)}; test_rows: {len(test)}; is_test_rate: {y_adv.mean():.10f}",
           f"adversarial_folds: {adv_folds}; mean={np.mean(adv_folds):.10f}; std={np.std(adv_folds):.10f}; oof={adv_auc:.10f}; best_iterations={best_iterations}",
           f"missing_only_folds: {missing_folds}; oof={missing_auc:.10f}",
           f"values_only_complete_case_rows: {complete_mask.sum()}; folds={values_folds}; oof={values_auc:.10f}; best_iterations={values_best}",
           "ADV_NO_MISSING_SHIFT uses the same strict complete-case LightGBM as values-only; no explicit or implicit missing marker remains",
           f"id_train_minmax: {train.id.min()},{train.id.max()}; id_test_minmax: {test.id.min()},{test.id.max()}; id_logistic_oof={id_auc:.10f}; id_direct_oriented_auc={id_direct:.10f}",
           "top_importance_gain_split:\n"+importance.head(25).to_string(index=False),"top_feature_shifts:\n"+shifts.head(15).to_string(index=False),
           "top_missing_shifts:\n"+missing.head(15).to_string(index=False),"top_distribution_distances:\n"+distances.head(15).to_string(index=False),
           "adversarial_quintiles:\n"+quintile_df.to_string(index=False),f"score_error_correlations: {correlations}",
           "Q1_vs_Q5_feature_differences:\n"+differences.to_string(index=False),f"interpretation: {interpretation}; missing_explains={missing_explains}; q1_q5_auc_abs_change={q_auc_change:.10f}",
           f"recommendation: {recommendation}",f"total_seconds: {elapsed:.2f}","problems: none"]
    METRICS_OUT.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"Adversarial folds={adv_folds}; OOF={adv_auc:.8f}; std={np.std(adv_folds):.8f}")
    print(f"Missing-only={missing_auc:.8f}; values-only/no-missing={values_auc:.8f}; id-only={id_auc:.8f}; direct-id={id_direct:.8f}")
    print("Importance:\n"+importance.head(25).to_string(index=False));print("Missing:\n"+missing.head(15).to_string(index=False));print("Distances:\n"+distances.head(15).to_string(index=False))
    print("Quintiles:\n"+quintile_df.to_string(index=False));print(f"Correlations={correlations}");print("Q1/Q5:\n"+differences.to_string(index=False))
    print(f"Interpretation={interpretation}; recommendation={recommendation}; time={elapsed:.2f}s; problems=none")


if __name__=="__main__":main()
