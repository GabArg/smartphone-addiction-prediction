"""EXP-026: forensic comparison and external-prior diagnosis, without submissions."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.project_paths import PROJECT_ROOT


ROOT = PROJECT_ROOT
DATA = ROOT / "data"
ORIGINAL_PATH = DATA / "original" / "Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv"
OUT = ROOT / "outputs"; REPORTS = OUT / "reports"; METRICS = OUT / "metrics"; PRED = OUT / "predictions"
METRICS_OUT = METRICS / "exp026_original_forensics.txt"
DIST_OUT = REPORTS / "exp026_original_vs_synthetic_distributions.csv"
CORR_OUT = REPORTS / "exp026_correlation_shift.csv"
AMBIG_OUT = REPORTS / "exp026_original_ambiguous_analysis.csv"
PRIORS_OUT = REPORTS / "exp026_original_priors.csv"
NN_OUT = REPORTS / "exp026_distance_to_original.csv"
OOF_OUT = PRED / "oof_exp026_original_prior_blend.csv"
EXP016_PATH = PRED / "oof_exp016_xgboost_depth5_9000.csv"
TARGET = "addicted_label"
WEIGHTS = [.01, .02, .03, .05, .075, .10, .15, .20, .30]


def auc(y: np.ndarray | pd.Series, p: np.ndarray | pd.Series) -> float:
    return float(roc_auc_score(y, p)) if pd.Series(y).nunique() == 2 else float("nan")


def regions(frame: pd.DataFrame) -> dict[str, pd.Series]:
    screen, social = frame["daily_screen_time_hours"], frame["social_media_hours"]
    known = screen.notna() & social.notna()
    return {
        "clear_positive_zone": known & ((screen > 8) | (social > 4)),
        "clear_negative_zone": known & (screen <= 6) & (social <= 4),
        "ambiguous_zone": known & (screen > 6) & (screen <= 8) & (social <= 4),
    }


def region_table(original: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    ro, rs = regions(original), regions(synthetic); rows=[]
    for name in ro:
        a, b = original.loc[ro[name], TARGET], synthetic.loc[rs[name], TARGET]
        rows.append({"region": name, "original_rows": len(a), "original_target_rate": a.mean(),
                     "original_positives": int(a.sum()), "original_negatives": int((1-a).sum()),
                     "original_purity": max(a.mean(), 1-a.mean()),
                     "original_contradictions": int((1-a).sum()) if name=="clear_positive_zone" else int(a.sum()) if name=="clear_negative_zone" else np.nan,
                     "synthetic_rows": len(b), "synthetic_target_rate": b.mean(),
                     "synthetic_positives": int(b.sum()), "synthetic_negatives": int((1-b).sum()),
                     "synthetic_purity": max(b.mean(), 1-b.mean()),
                     "synthetic_contradictions": int((1-b).sum()) if name=="clear_positive_zone" else int(b.sum()) if name=="clear_negative_zone" else np.nan,
                     "target_rate_difference": b.mean()-a.mean()})
    return pd.DataFrame(rows)


def granularity(series: pd.Series) -> float:
    values=np.sort(series.dropna().astype(float).unique()); diffs=np.diff(values);diffs=diffs[diffs>1e-10]
    if not len(diffs): return np.nan
    rounded=np.round(diffs,8); return float(pd.Series(rounded).value_counts().index[0])


def distributions(original: pd.DataFrame, synthetic: pd.DataFrame, shared: list[str], categorical: list[str]) -> pd.DataFrame:
    rows=[]; qs=[.01,.05,.10,.25,.50,.75,.90,.95,.99]
    for feature in shared:
        if feature in categorical:
            a=original[feature].astype("string").fillna("__MISSING__").value_counts(normalize=True)
            b=synthetic[feature].astype("string").fillna("__MISSING__").value_counts(normalize=True)
            cats=a.index.union(b.index); tv=.5*(a.reindex(cats,fill_value=0)-b.reindex(cats,fill_value=0)).abs().sum()
            rows.append({"feature":feature,"type":"categorical","original_missing":original[feature].isna().mean(),
                         "synthetic_missing":synthetic[feature].isna().mean(),"tv_distance":tv,
                         "original_categories":"|".join(map(str,a.index)),"synthetic_categories":"|".join(map(str,b.index))})
        else:
            a=original[feature].dropna().astype(float);b=synthetic[feature].dropna().astype(float)
            row={"feature":feature,"type":"numeric","original_missing":original[feature].isna().mean(),
                 "synthetic_missing":synthetic[feature].isna().mean(),"original_mean":a.mean(),"synthetic_mean":b.mean(),
                 "original_median":a.median(),"synthetic_median":b.median(),"original_std":a.std(),"synthetic_std":b.std(),
                 "original_min":a.min(),"synthetic_min":b.min(),"original_max":a.max(),"synthetic_max":b.max(),
                 "ks_statistic":ks_2samp(a,b).statistic,"wasserstein":wasserstein_distance(a,b),
                 "original_unique":a.nunique(),"synthetic_unique":b.nunique(),
                 "original_granularity":granularity(a),"synthetic_granularity":granularity(b)}
            for q in qs: row[f"original_q{int(q*100):02d}"]=a.quantile(q);row[f"synthetic_q{int(q*100):02d}"]=b.quantile(q)
            rows.append(row)
    return pd.DataFrame(rows)


def correlation_shift(original: pd.DataFrame, synthetic: pd.DataFrame, numeric: list[str]) -> pd.DataFrame:
    o=original[numeric].dropna();s=synthetic[numeric].dropna();rows=[]
    for method in ["pearson","spearman"]:
        co=o.corr(method=method);cs=s.corr(method=method)
        for i,a in enumerate(numeric):
            for b in numeric[i+1:]: rows.append({"method":method,"feature_a":a,"feature_b":b,
                "original_correlation":co.loc[a,b],"synthetic_correlation":cs.loc[a,b],
                "absolute_difference":abs(co.loc[a,b]-cs.loc[a,b])})
    return pd.DataFrame(rows).sort_values("absolute_difference",ascending=False)


def preprocessing(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),numeric),
        ("cat",Pipeline([("impute",SimpleImputer(strategy="constant",fill_value="__MISSING__")),
                         ("onehot",OneHotEncoder(handle_unknown="ignore"))]),categorical)])


def ambiguous_models(original: pd.DataFrame, shared: list[str], numeric: list[str], categorical: list[str]) -> tuple[pd.DataFrame,dict[str,float],pd.DataFrame]:
    mask=regions(original)["ambiguous_zone"];data=original.loc[mask].reset_index(drop=True);y=data[TARGET].to_numpy();rows=[]
    for feature in numeric:
        valid=data[feature].notna();raw=auc(y[valid],data.loc[valid,feature]);rows.append({"analysis":"individual_numeric","feature":feature,
            "auc_direct":raw,"auc_oriented":max(raw,1-raw),"direction":"higher_positive" if raw>=.5 else "lower_positive"})
    for feature in categorical:
        rates=data.groupby(feature,dropna=False)[TARGET].agg(["count","mean"])
        rows.append({"analysis":"categorical_rates","feature":feature,"category_rates":rates.to_dict("index")})
    cv=StratifiedKFold(5,shuffle=True,random_state=42);log_oof=np.zeros(len(data));lgb_oof=np.zeros(len(data));coefs=[];imps=[]
    for fold,(tr,va) in enumerate(cv.split(data,y),1):
        pipe=Pipeline([("prep",preprocessing(numeric,categorical)),("model",LogisticRegression(C=1,solver="liblinear",max_iter=2000))])
        pipe.fit(data.iloc[tr][shared],y[tr]);log_oof[va]=pipe.predict_proba(data.iloc[va][shared])[:,1]
        names=pipe.named_steps["prep"].get_feature_names_out();coefs.append(pd.Series(pipe.named_steps["model"].coef_[0],index=names))
        combined=pd.concat([data.iloc[tr][shared],data.iloc[va][shared]],ignore_index=True)
        for c in categorical: combined[c]=pd.Categorical(combined[c].astype("string").fillna("__MISSING__"))
        xtr=combined.iloc[:len(tr)];xva=combined.iloc[len(tr):]
        model=LGBMClassifier(objective="binary",n_estimators=500,learning_rate=.03,num_leaves=15,random_state=42,n_jobs=-1,verbosity=-1)
        model.fit(xtr,y[tr],categorical_feature=categorical,eval_set=[(xva,y[va])],eval_metric="auc",
                  callbacks=[early_stopping(50,verbose=False),log_evaluation(0)])
        lgb_oof[va]=model.predict_proba(xva,num_iteration=model.best_iteration_)[:,1]
        imps.append(pd.Series(model.booster_.feature_importance("gain"),index=shared))
    metrics={"rows":len(data),"target_rate":y.mean(),"logistic_oof_auc":auc(y,log_oof),"lightgbm_oof_auc":auc(y,lgb_oof)}
    importance=pd.DataFrame({"logistic_abs_coef":pd.DataFrame(coefs).abs().mean(),"lightgbm_gain":pd.DataFrame(imps).mean()}).fillna(0).sort_values("lightgbm_gain",ascending=False)
    return pd.DataFrame(rows),metrics,importance


def build_priors(original: pd.DataFrame, synthetic: pd.DataFrame, shared: list[str], numeric: list[str], categorical: list[str]) -> dict[str,np.ndarray]:
    y=original[TARGET].to_numpy();prior_a=np.full(len(synthetic),y.mean());ro=regions(original);rs=regions(synthetic)
    for name in ro: prior_a[rs[name].to_numpy()]=original.loc[ro[name],TARGET].mean()
    logpipe=Pipeline([("prep",preprocessing(numeric,categorical)),("model",LogisticRegression(C=1,solver="liblinear",max_iter=3000))])
    logpipe.fit(original[shared],y);prior_b=logpipe.predict_proba(synthetic[shared])[:,1]
    combined=pd.concat([original[shared],synthetic[shared]],ignore_index=True)
    for c in categorical: combined[c]=pd.Categorical(combined[c].astype("string").fillna("__MISSING__"))
    lgb=LGBMClassifier(objective="binary",n_estimators=500,learning_rate=.03,num_leaves=15,random_state=42,n_jobs=-1,verbosity=-1)
    lgb.fit(combined.iloc[:len(original)],y,categorical_feature=categorical,callbacks=[log_evaluation(0)])
    prior_c=lgb.predict_proba(combined.iloc[len(original):])[:,1]
    return {"PRIOR-A":prior_a,"PRIOR-B":prior_b,"PRIOR-C":prior_c}


def nested_blend(y: np.ndarray,p16: np.ndarray,prior: np.ndarray,splits:list[tuple[np.ndarray,np.ndarray]]) -> tuple[np.ndarray,list[dict[str,float]]]:
    output=p16.copy();records=[]
    for fold,(tr,va) in enumerate(splits,1):
        candidates=[]
        for w in WEIGHTS: candidates.append((auc(y[tr],(1-w)*p16[tr]+w*prior[tr]),-w,w))
        score,_,w=max(candidates);output[va]=(1-w)*p16[va]+w*prior[va]
        records.append({"fold":fold,"weight":w,"train_auc":score,"valid_auc":auc(y[va],output[va]),"valid_delta":auc(y[va],output[va])-auc(y[va],p16[va])})
    return output,records


def matching(original:pd.DataFrame,synthetic:pd.DataFrame,shared:list[str],numeric:list[str],categorical:list[str]) -> pd.DataFrame:
    rows=[]
    variants={"exact":None,"round_1_decimal":"decimal","natural_granularity":"natural"}
    for name,kind in variants.items():
        o=original[shared].copy();s=synthetic[shared].copy()
        if kind=="decimal":
            o[numeric]=o[numeric].round(1);s[numeric]=s[numeric].round(1)
        elif kind=="natural":
            for c in numeric:
                step=granularity(original[c]);o[c]=(o[c]/step).round()*step;s[c]=(s[c]/step).round()*step
        stats=pd.concat([o,original[[TARGET]]],axis=1).groupby(shared,dropna=False)[TARGET].agg(["count","mean"]).reset_index()
        mapped=s.merge(stats,on=shared,how="left",sort=False);mask=mapped["count"].notna().to_numpy()
        agreement=((synthetic.loc[mask,TARGET].to_numpy()==(mapped.loc[mask,"mean"].to_numpy()>=.5))).mean() if mask.any() else np.nan
        rows.append({"match_type":name,"synthetic_matches":int(mask.sum()),"coverage":mask.mean(),"target_agreement":agreement,"contradictions":int(mask.sum()*(1-agreement)) if mask.any() else 0})
    return pd.DataFrame(rows)


def main()->None:
    start=perf_counter();original=pd.read_csv(ORIGINAL_PATH);train=pd.read_csv(DATA/"train.csv");test=pd.read_csv(DATA/"test.csv")
    synthetic_features=[c for c in train if c not in {"id",TARGET}];original_features=[c for c in original if c!=TARGET]
    shared=[c for c in synthetic_features if c in original_features];original_only=[c for c in original_features if c not in shared];synthetic_only=[c for c in train if c not in shared+[TARGET]]
    categorical=[c for c in shared if not pd.api.types.is_numeric_dtype(original[c])];numeric=[c for c in shared if c not in categorical]
    audit={"shape":original.shape,"columns":original.columns.tolist(),"dtypes":original.dtypes.astype(str).to_dict(),"duplicates":int(original.duplicated().sum()),
           "missing":original.isna().sum().to_dict(),"cardinality":original.nunique(dropna=False).to_dict(),"target_rate":original[TARGET].mean()}
    reg=region_table(original,train);dist=distributions(original,train,shared,categorical);corr=correlation_shift(original,train,numeric)
    ambig_analysis,ambig_metrics,ambig_importance=ambiguous_models(original,shared,numeric,categorical)
    exp16=pd.read_csv(EXP016_PATH)
    if not exp16.id.equals(train.id) or not exp16.y_true.equals(train[TARGET]):raise ValueError("EXP016 no alinea")
    y=train[TARGET].to_numpy();p16=exp16.oof_prediction.to_numpy();residual=y-p16
    cv=StratifiedKFold(5,shuffle=True,random_state=42);splits=list(cv.split(np.zeros(len(train)),y));priors=build_priors(original,train,shared,numeric,categorical)
    prior_rows=[];nested_predictions={};nested_records={}
    for name,prior in priors.items():
        nested,records=nested_blend(y,p16,prior,splits);nested_predictions[name]=nested;nested_records[name]=records
        folds=[auc(y[va],nested[va]) for _,va in splits]
        prior_rows.append({"prior":name,"synthetic_auc":auc(y,prior),"synthetic_logloss":log_loss(y,np.clip(prior,1e-8,1-1e-8)),
                           "pearson_exp016":pd.Series(prior).corr(pd.Series(p16),method="pearson"),"spearman_exp016":pd.Series(prior).corr(pd.Series(p16),method="spearman"),
                           "pearson_residual":pd.Series(prior).corr(pd.Series(residual),method="pearson"),"spearman_residual":pd.Series(prior).corr(pd.Series(residual),method="spearman"),
                           "nested_auc":auc(y,nested),"nested_std":np.std(folds),"nested_delta":auc(y,nested)-auc(y,p16),"nested_folds":folds,
                           "weights":[r["weight"] for r in records]})
    prior_df=pd.DataFrame(prior_rows).sort_values("nested_auc",ascending=False);best_name=prior_df.iloc[0].prior;best_nested=nested_predictions[best_name]
    regional=[];rt=regions(train)
    for region,mask_s in rt.items():
        mask=mask_s.to_numpy()
        for name,prior in priors.items():regional.append({"region":region,"prior":name,"rows":int(mask.sum()),"auc_exp016":auc(y[mask],p16[mask]),
            "auc_prior":auc(y[mask],prior[mask]),"auc_nested":auc(y[mask],nested_predictions[name][mask])})
    # Ambiguous 4x4 cells.
    cell_rows=[];amb=rt["ambiguous_zone"]
    sb=pd.cut(train.daily_screen_time_hours,[6,6.5,7,7.5,8],include_lowest=False);mb=pd.cut(train.social_media_hours,[-np.inf,1,2,3,4])
    for a in sb.dropna().unique():
        for b in mb.dropna().unique():
            mask=(amb & sb.eq(a) & mb.eq(b)).to_numpy()
            if mask.sum()>=1000:cell_rows.append({"screen_band":str(a),"social_band":str(b),"rows":int(mask.sum()),"auc_exp016":auc(y[mask],p16[mask]),
                **{f"auc_{n}":auc(y[mask],v[mask]) for n,v in priors.items()}})
    # Distance to original using shared numeric features only.
    imputer=SimpleImputer(strategy="median");scaler=StandardScaler();xo=scaler.fit_transform(imputer.fit_transform(original[numeric]));xs=scaler.transform(imputer.transform(train[numeric]))
    nn=NearestNeighbors(n_neighbors=1,algorithm="kd_tree").fit(xo);distance=np.zeros(len(train));neighbor=np.zeros(len(train),dtype=int)
    for i in range(0,len(train),50000):
        stop=min(i+50000,len(train));d,idx=nn.kneighbors(xs[i:stop]);distance[i:stop]=d.ravel();neighbor[i:stop]=idx.ravel()
    dbins=pd.qcut(distance,10,duplicates="drop");nn_rows=[]
    for band in dbins.unique().sort_values():
        mask=(dbins==band);nearest_label=original[TARGET].to_numpy()[neighbor[mask]]
        nn_rows.append({"distance_decile":str(band),"rows":int(mask.sum()),"distance_mean":distance[mask].mean(),"target_rate":y[mask].mean(),
                        "auc_exp016":auc(y[mask],p16[mask]),"auc_prior_c":auc(y[mask],priors["PRIOR-C"][mask]),"logloss_exp016":log_loss(y[mask],p16[mask]),
                        "nearest_original_label_agreement":(y[mask]==nearest_label).mean()})
    nn_df=pd.DataFrame(nn_rows);matches=matching(original,train,shared,numeric,categorical)
    best_delta=float(prior_df.iloc[0].nested_delta);best_records=nested_records[best_name];improved=sum(r["valid_delta"]>0 for r in best_records)
    ambig_big=max((r["auc_nested"]-r["auc_exp016"] for r in regional if r["region"]=="ambiguous_zone"),default=0)
    if best_delta>=.00005:recommendation="Senal fuerte: recomendar EXP-027."
    elif best_delta>=.00002 and improved>=4:recommendation="Senal marginal estable: recomendar EXP-027 con cautela."
    elif best_delta>=.00002:recommendation="Senal marginal inestable: no recomendar EXP-027."
    elif ambig_big>=.00010:recommendation="No ayuda globalmente, pero mejora ambiguous_zone; considerar diagnostico estructural predefinido antes de EXP-027."
    else:recommendation="No existe senal residual suficiente: no recomendar EXP-027."
    REPORTS.mkdir(parents=True,exist_ok=True);METRICS.mkdir(parents=True,exist_ok=True);PRED.mkdir(parents=True,exist_ok=True)
    dist.to_csv(DIST_OUT,index=False);corr.to_csv(CORR_OUT,index=False)
    pd.concat([ambig_analysis,pd.DataFrame([{"analysis":"metrics",**ambig_metrics,"importance":ambig_importance.to_dict("index")}])],ignore_index=True).to_csv(AMBIG_OUT,index=False)
    prior_df.to_csv(PRIORS_OUT,index=False);nn_df.to_csv(NN_OUT,index=False)
    fold_id=np.zeros(len(train),dtype=int);weights_by_row=np.zeros(len(train))
    for fold,(_,va) in enumerate(splits,1):fold_id[va]=fold;weights_by_row[va]=best_records[fold-1]["weight"]
    pd.DataFrame({"id":train.id,"y_true":y,"exp016_prediction":p16,"original_prior":priors[best_name],"blended_prediction":best_nested,"prior":best_name,"fold":fold_id,"selected_weight":weights_by_row}).to_csv(OOF_OUT,index=False)
    screen_social=corr[(corr.method=="pearson")&(((corr.feature_a=="daily_screen_time_hours")&(corr.feature_b=="social_media_hours"))|((corr.feature_b=="daily_screen_time_hours")&(corr.feature_a=="social_media_hours")))].iloc[0]
    elapsed=perf_counter()-start
    lines=["EXP-026 original forensics; no submission; no experiment-log update",f"audit: {audit}",f"shared: {shared}",f"original_only: {original_only}",f"synthetic_only: {synthetic_only}",
           "regions:\n"+reg.to_string(index=False),"top_distribution_shifts:\n"+dist.sort_values(["ks_statistic","tv_distance"],ascending=False,na_position="last").head(20).to_string(index=False),
           "top_correlation_shifts:\n"+corr.head(20).to_string(index=False),f"screen_social_correlation: {screen_social.to_dict()}",
           f"ambiguous_metrics: {ambig_metrics}","ambiguous_individual:\n"+ambig_analysis.to_string(index=False),"ambiguous_importance:\n"+ambig_importance.to_string(),
           "priors:\n"+prior_df.to_string(index=False),f"nested_records: {nested_records}","regional:\n"+pd.DataFrame(regional).to_string(index=False),
           "ambiguous_cells:\n"+pd.DataFrame(cell_rows).to_string(index=False),"distance_to_original:\n"+nn_df.to_string(index=False),"matches:\n"+matches.to_string(index=False),
           f"recommendation: {recommendation}",f"total_seconds: {elapsed:.2f}","problems: none"]
    METRICS_OUT.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"Audit={audit}\nshared={shared}\noriginal_only={original_only}\nsynthetic_only={synthetic_only}")
    print("Regions:\n"+reg.to_string(index=False));print("Correlation shifts:\n"+corr.head(20).to_string(index=False));print(f"Ambiguous={ambig_metrics}")
    print("Priors:\n"+prior_df.to_string(index=False));print("Regional:\n"+pd.DataFrame(regional).to_string(index=False));print("Distance:\n"+nn_df.to_string(index=False));print("Matches:\n"+matches.to_string(index=False))
    print(f"Recommendation={recommendation}; time={elapsed:.2f}s; problems=none")


if __name__=="__main__":main()
