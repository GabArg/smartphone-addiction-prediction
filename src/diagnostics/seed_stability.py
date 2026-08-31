"""EXP-027: stochastic seed-diversity diagnosis for the exact EXP-016 pipeline."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.project_paths import PROJECT_ROOT
from src.train_xgboost_exp008 import MODEL_PARAMS, ordinal_encode_categories
from src.train_xgboost_exp012_threshold_features import NEW_FEATURES, add_threshold_features


ROOT=PROJECT_ROOT;DATA=ROOT/"data";OUT=ROOT/"outputs"
PRED=OUT/"predictions";REPORTS=OUT/"reports";METRICS=OUT/"metrics";SUB=OUT/"submissions"
METRICS_OUT=METRICS/"exp027_seed_diversity.txt";CORR_OUT=REPORTS/"exp027_seed_correlations.csv"
FOLD_OUT=REPORTS/"exp027_seed_fold_metrics.csv";ENSEMBLE_OUT=REPORTS/"exp027_seed_ensemble_results.csv"
SUB_OUT=SUB/"submission_exp027_seed_ensemble.csv";LOG_PATH=METRICS/"experiment_log.csv"
SEEDS=[42,2026,777];REFERENCE=np.array([0.9650202929179754,0.9656644620983974,0.9658836874972772,0.966455024819542,0.9654874019318932])
EXP016=0.9657017303568276;EXP021=0.9658597975;EARLY=300
MODEL=dict(MODEL_PARAMS);MODEL.update({"max_depth":5,"n_estimators":9000})
EXISTING_16= PRED/"oof_exp016_xgboost_depth5_9000.csv"; EXISTING_16_TEST=PRED/"test_exp016_xgboost_depth5_9000.csv"
OOF22=PRED/"oof_exp022_catboost_thresholds_9000.csv";TEST22=PRED/"test_exp022_catboost_thresholds_9000.csv"


def rank(values:np.ndarray)->np.ndarray:
    r=pd.Series(values).rank(method="average").to_numpy(dtype=np.float64);return (r-1)/(len(r)-1)


def folds_auc(y:np.ndarray,p:np.ndarray,splits:list[tuple[np.ndarray,np.ndarray]])->list[float]:
    return [float(roc_auc_score(y[v],p[v])) for _,v in splits]


def train_fold(X:pd.DataFrame,y:pd.Series,Xtest:pd.DataFrame,seed:int,fold:int,tr:np.ndarray,va:np.ndarray,predict_test:bool):
    params=dict(MODEL);params["random_state"]=seed;start=perf_counter()
    model=XGBClassifier(**params,early_stopping_rounds=EARLY)
    model.fit(X.iloc[tr],y.iloc[tr],eval_set=[(X.iloc[va],y.iloc[va])],verbose=False)
    best=int(model.best_iteration);last=len(model.evals_result()["validation_0"]["auc"])-1
    pv=model.predict_proba(X.iloc[va],iteration_range=(0,best+1))[:,1].astype(np.float64)
    pt=model.predict_proba(Xtest,iteration_range=(0,best+1))[:,1].astype(np.float64) if predict_test else None
    return pv,pt,float(roc_auc_score(y.iloc[va],pv)),best,last,perf_counter()-start


def candidate(name,method,pred,y,indices,splits3):
    score=float(roc_auc_score(y[indices],pred[indices]));folds=[float(roc_auc_score(y[v],pred[v])) for _,v in splits3]
    return {"name":name,"method":method,"auc":score,"folds":folds,"delta":score-float(roc_auc_score(y[indices],predictions[42][indices])),
            "improved_folds":int(sum(np.asarray(folds)>np.asarray(seed3_folds[42]))),"prediction":pred}


def update_log(auc_value:float,notes:str):
    cols=["experiment_id","datetime","model","features","cv_strategy","cv_roc_auc","kaggle_score","notes"]
    log=pd.read_csv(LOG_PATH,dtype=str,keep_default_na=False)
    if log.columns.tolist()!=cols:raise ValueError("experiment_log.csv inesperado")
    log=log.loc[~log.experiment_id.eq("EXP-027")].copy()
    row=pd.DataFrame([{"experiment_id":"EXP-027","datetime":datetime.now().astimezone().isoformat(timespec="seconds"),
        "model":"XGBoost_Seed_Ensemble","features":"exp012_threshold_region_features","cv_strategy":"StratifiedKFold_5_seed_ensemble",
        "cv_roc_auc":f"{auc_value:.8f}","kaggle_score":"","notes":notes}])
    pd.concat([log,row],ignore_index=True).to_csv(LOG_PATH,index=False)


def main():
    global predictions,seed3_folds
    total_start=perf_counter();train=pd.read_csv(DATA/"train.csv");test=pd.read_csv(DATA/"test.csv");sample=pd.read_csv(DATA/"sample_submission.csv")
    originals=[c for c in train if c not in {"id","addicted_label"}];raw=add_threshold_features(train[originals]);rawt=add_threshold_features(test[originals])
    if raw.columns.tolist()!=rawt.columns.tolist() or len(raw.columns)!=len(originals)+len(NEW_FEATURES):raise ValueError("Features no coinciden con EXP-016")
    cats=[c for c in originals if not pd.api.types.is_numeric_dtype(raw[c])];numeric=[c for c in raw if c not in cats]
    X,Xtest,mappings=ordinal_encode_categories(raw,rawt,cats)
    if not X[numeric].equals(raw[numeric]) or not Xtest[numeric].equals(rawt[numeric]):raise ValueError("Preprocessing numerico alterado")
    yseries=train.addicted_label;y=yseries.to_numpy();splits=list(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y));splits3=splits[:3]
    diagnostic_idx=np.sort(np.concatenate([v for _,v in splits3]));predictions={s:np.full(len(train),np.nan) for s in SEEDS};test_sums={s:np.zeros(len(test)) for s in SEEDS}
    rows=[];seed3_folds={};best_iters={s:{} for s in SEEDS}
    for seed in SEEDS:
        for fold,(tr,va) in enumerate(splits3,1):
            pv,pt,score,best,last,seconds=train_fold(X,yseries,Xtest,seed,fold,tr,va,predict_test=(seed!=42))
            predictions[seed][va]=pv
            if pt is not None:test_sums[seed]+=pt/5
            best_iters[seed][fold]=best;rows.append({"stage":"diagnostic","seed":seed,"fold":fold,"auc":score,"best_iteration":best,"last_iteration":last,"seconds":seconds})
            pd.DataFrame(rows).to_csv(FOLD_OUT,index=False);print(f"seed={seed} fold={fold} AUC={score:.8f} best={best} time={seconds:.1f}s",flush=True)
        seed3_folds[seed]=[r["auc"] for r in rows if r["seed"]==seed and r["stage"]=="diagnostic"]
        if seed==42 and np.max(np.abs(np.asarray(seed3_folds[42])-REFERENCE[:3]))>2e-6:
            raise RuntimeError(f"Seed42 no reproduce EXP-016: {seed3_folds[42]}")

    correlations=[]
    for i,a in enumerate(SEEDS):
        for b in SEEDS[i+1:]:correlations.append({"stage":"3fold","seed_a":a,"seed_b":b,
            "pearson":pd.Series(predictions[a][diagnostic_idx]).corr(pd.Series(predictions[b][diagnostic_idx]),method="pearson"),
            "spearman":pd.Series(predictions[a][diagnostic_idx]).corr(pd.Series(predictions[b][diagnostic_idx]),method="spearman")})
    corr3=pd.DataFrame(correlations);corr3.to_csv(CORR_OUT,index=False)
    seed3_auc={s:float(roc_auc_score(y[diagnostic_idx],predictions[s][diagnostic_idx])) for s in SEEDS}
    blends=[]
    definitions=[("prob_42_2026",{42:.5,2026:.5}), ("prob_42_777",{42:.5,777:.5}),
                 ("prob_equal3",{42:1/3,2026:1/3,777:1/3}), ("prob_50_25_25",{42:.5,2026:.25,777:.25})]
    for name,weights in definitions:
        p=sum(w*predictions[s] for s,w in weights.items());blends.append(candidate(name,"probability",p,y,diagnostic_idx,splits3))
        pr=sum(w*rank(predictions[s][diagnostic_idx]) for s,w in weights.items());full=np.full(len(train),np.nan);full[diagnostic_idx]=pr
        blends.append(candidate(name.replace("prob","rank"),"rank",full,y,diagnostic_idx,splits3))
    best3=max(blends,key=lambda x:x["auc"])
    cond_a=any(abs(seed3_auc[s]-seed3_auc[42])<=.00010 and float(corr3[((corr3.seed_a==42)&(corr3.seed_b==s))|((corr3.seed_b==42)&(corr3.seed_a==s))].pearson.iloc[0])<=.9995 for s in [2026,777])
    cond_b=any(b["delta"]>=.00003 and b["improved_folds"]>=2 for b in blends)
    cond_c=any("equal3" in b["name"] and b["delta"]>=.00005 for b in blends);passed=cond_a or cond_b or cond_c
    promising=[42]
    if passed:
        for s in [2026,777]:
            pair=[b for b in blends if str(s) in b["name"]]
            c=float(corr3[((corr3.seed_a==42)&(corr3.seed_b==s))|((corr3.seed_b==42)&(corr3.seed_a==s))].pearson.iloc[0])
            if (abs(seed3_auc[s]-seed3_auc[42])<=.00010 and c<=.9995) or any(b["delta"]>=.00002 for b in pair):promising.append(s)
        if len(promising)==1:promising=SEEDS.copy()

    full_results=[];submission_generated=False;submission_path="none";best_seed_ensemble=None;best_cat=None
    if passed:
        existing=pd.read_csv(EXISTING_16);predictions[42]=existing.oof_prediction.to_numpy();test_sums[42]=pd.read_csv(EXISTING_16_TEST).prediction.to_numpy()
        best_iters[42].update({4:8198,5:7197})
        for seed in promising:
            if seed==42:continue
            for fold,(tr,va) in enumerate(splits[3:],4):
                pv,pt,score,best,last,seconds=train_fold(X,yseries,Xtest,seed,fold,tr,va,True);predictions[seed][va]=pv;test_sums[seed]+=pt/5
                best_iters[seed][fold]=best;rows.append({"stage":"full_completion","seed":seed,"fold":fold,"auc":score,"best_iteration":best,"last_iteration":last,"seconds":seconds})
                pd.DataFrame(rows).to_csv(FOLD_OUT,index=False);print(f"seed={seed} fold={fold} AUC={score:.8f} best={best} time={seconds:.1f}s",flush=True)
        for seed in promising:
            pd.DataFrame({"id":train.id,"y_true":y,"oof_prediction":predictions[seed]}).to_csv(PRED/f"oof_exp027_seed{seed}.csv",index=False)
            full_results.append({"candidate":f"seed{seed}","method":"individual","weights":str({seed:1}),"auc":roc_auc_score(y,predictions[seed]),
                                 "folds":folds_auc(y,predictions[seed],splits),"std":np.std(folds_auc(y,predictions[seed],splits))})
        # Seed-only ensembles.
        weight_sets=[]
        if len(promising)==2 and 42 in promising:
            other=[s for s in promising if s!=42][0];weight_sets=[{42:w,other:1-w} for w in [.5,.6,.7,.8,.9]]
        elif len(promising)==3:
            for a in range(0,11):
                for b in range(0,11-a):weight_sets.append({42:a/10,2026:b/10,777:(10-a-b)/10})
        weight_sets.append({s:1/len(promising) for s in promising})
        for weights in weight_sets:
            pp=sum(w*predictions[s] for s,w in weights.items());pr=sum(w*rank(predictions[s]) for s,w in weights.items())
            for method,p in [("probability",pp),("rank",pr)]:
                fs=folds_auc(y,p,splits);full_results.append({"candidate":"seed_ensemble","method":method,"weights":str(weights),"auc":roc_auc_score(y,p),"folds":fs,"std":np.std(fs)})
        best_seed_ensemble=max([r for r in full_results if r["candidate"]=="seed_ensemble"],key=lambda r:r["auc"])
        weights=eval(best_seed_ensemble["weights"]);seed_oof=(sum(w*predictions[s] for s,w in weights.items()) if best_seed_ensemble["method"]=="probability" else sum(w*rank(predictions[s]) for s,w in weights.items()))
        seed_test=(sum(w*test_sums[s] for s,w in weights.items()) if best_seed_ensemble["method"]=="probability" else sum(w*rank(test_sums[s]) for s,w in weights.items()))
        p22=pd.read_csv(OOF22).oof_prediction.to_numpy();t22=pd.read_csv(TEST22).prediction.to_numpy()
        cat_candidates=[]
        for cw in [.20,.25,.30,.35]:
            for method,px,pc,tx,tc in [("probability",seed_oof,p22,seed_test,t22),("rank",rank(seed_oof),rank(p22),rank(seed_test),rank(t22))]:
                p=(1-cw)*px+cw*pc;fs=folds_auc(y,p,splits);cat_candidates.append({"candidate":"bagged_xgb_catboost","method":method,"cat_weight":cw,"auc":roc_auc_score(y,p),"folds":fs,"std":np.std(fs),"prediction":p,"test":(1-cw)*tx+cw*tc})
        best_cat=max(cat_candidates,key=lambda r:r["auc"]);delta_cat=best_cat["auc"]-EXP021;improved_cat=sum(np.asarray(best_cat["folds"])>np.asarray(REFERENCE))
        if delta_cat>=.00002 and improved_cat>=4:
            sub=pd.DataFrame({"id":sample.id,"addicted_label":best_cat["test"]})
            if len(sub)!=296302 or sub.isna().any().any() or not sub.addicted_label.between(0,1).all():raise ValueError("Submission invalida")
            sub.to_csv(SUB_OUT,index=False);submission_generated=True;submission_path=str(SUB_OUT)
            update_log(best_cat["auc"],f"{best_cat['method']}; seed_weights={weights}; catboost_weight={best_cat['cat_weight']}")
        pd.DataFrame([{k:v for k,v in r.items() if k not in {"prediction","test"}} for r in full_results+cat_candidates]).to_csv(ENSEMBLE_OUT,index=False)
        # Full correlations.
        full_corr=[]
        for i,a in enumerate(promising):
            for b in promising[i+1:]:full_corr.append({"stage":"5fold","seed_a":a,"seed_b":b,"pearson":pd.Series(predictions[a]).corr(pd.Series(predictions[b])),"spearman":pd.Series(predictions[a]).corr(pd.Series(predictions[b]),method="spearman")})
        pd.concat([corr3,pd.DataFrame(full_corr)],ignore_index=True).to_csv(CORR_OUT,index=False)

    REPORTS.mkdir(parents=True,exist_ok=True);METRICS.mkdir(parents=True,exist_ok=True);PRED.mkdir(parents=True,exist_ok=True);SUB.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(FOLD_OUT,index=False)
    blend_table=pd.DataFrame([{k:v for k,v in b.items() if k!="prediction"} for b in blends])
    elapsed=perf_counter()-total_start
    lines=["EXP-027 seed diversity; exact EXP-016 pipeline",f"model: {MODEL}; early={EARLY}; features={NEW_FEATURES}; mappings={mappings}",
           f"seed3_folds: {seed3_folds}; seed3_combined_auc: {seed3_auc}","3fold_correlations:\n"+corr3.to_string(index=False),"3fold_blends:\n"+blend_table.to_string(index=False),
           f"filter: cond_a={cond_a}; cond_b={cond_b}; cond_c={cond_c}; passed={passed}; promising={promising}",f"best_iterations: {best_iters}",
           f"full_results: {full_results}",f"best_seed_ensemble: {best_seed_ensemble}",f"best_bagged_catboost: {best_cat}",
           f"submission_generated: {submission_generated}; path={submission_path}",f"total_seconds: {elapsed:.2f}","problems: none"]
    METRICS_OUT.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"3fold AUC={seed3_auc}; correlations=\n{corr3.to_string(index=False)}\nblends=\n{blend_table.to_string(index=False)}")
    print(f"Passed={passed}; promising={promising}; full={full_results}; best_seed={best_seed_ensemble}; best_cat={best_cat}")
    print(f"Time={elapsed:.2f}s; submission={submission_generated}; path={submission_path}; problems=none")


if __name__=="__main__":main()
