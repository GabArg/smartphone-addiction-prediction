"""EXP-033: exact/discretized value target-encoding residual-signal diagnosis."""

from __future__ import annotations

import heapq
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.project_paths import PROJECT_ROOT


ROOT=PROJECT_ROOT;DATA=ROOT/"data";OUT=ROOT/"outputs"
PRED=OUT/"predictions";REPORTS=OUT/"reports";METRICS=OUT/"metrics"
NUM_CONT=["daily_screen_time_hours","social_media_hours","weekend_screen_time","work_study_hours","gaming_hours","sleep_hours"]
NUM_INT=["notifications_per_day","app_opens_per_day","age"]
CATS=["gender","stress_level","academic_work_impact"]
FEATURES=NUM_CONT+NUM_INT+CATS;SMOOTH=[1,5,10,20,50,100,200]
GRANS=["exact","round_0.1","round_0.25","round_0.5","round_1.0"]
WEIGHTS=[.005,.01,.02,.03,.05,.075,.10,.15,.20];MULTI_WEIGHTS=[.005,.01,.02,.03,.05,.075,.10]
EXP027=.965919188052602;EXP028=.9659307051390922;MISS_NUM=-1e20;MISS_CAT="__MISSING__"
PAIR_SPECS=[("daily_screen_time_hours","social_media_hours"),("daily_screen_time_hours","weekend_screen_time"),
 ("social_media_hours","weekend_screen_time"),("daily_screen_time_hours","work_study_hours"),("social_media_hours","gaming_hours")]


def load_oof(path,train):
    d=pd.read_csv(path)
    if not d.id.equals(train.id) or not d.y_true.equals(train.addicted_label) or d.oof_prediction.isna().any():raise ValueError(f"OOF invalida {path.name}")
    return d.oof_prediction.to_numpy(np.float64)

def key_single(s,gran):
    if s.name in CATS:return s.fillna(MISS_CAT).astype(str).to_numpy()
    x=s.to_numpy(np.float64)
    if gran=="round_0.1":x=np.round(x,1)
    elif gran=="round_0.25":x=np.round(x*4)/4
    elif gran=="round_0.5":x=np.round(x*2)/2
    elif gran=="round_1.0":x=np.round(x)
    return np.where(np.isnan(x),MISS_NUM,x)

def key_pair(df,a,b,gran):
    ka=key_single(df[a],gran);kb=key_single(df[b],gran)
    return pd.util.hash_pandas_object(pd.DataFrame({"a":ka,"b":kb}),index=False).to_numpy(np.uint64)

def rep_specs():
    out=[]
    for f in FEATURES:
        for g in (GRANS if f in NUM_CONT else ["exact"]):out.append((f,g))
    return out

def map_stats(ktrain,ytrain,kval,global_mean,no_missing=False,missing_val=None):
    st=pd.DataFrame({"k":ktrain,"y":ytrain}).groupby("k",sort=False).y.agg(["count","sum"])
    kv=pd.Series(kval);cnt=kv.map(st["count"]).fillna(0).to_numpy(np.float64,copy=True);sm=kv.map(st["sum"]).fillna(0).to_numpy(np.float64,copy=True)
    unseen=cnt==0
    if no_missing and missing_val is not None:
        missing=np.asarray(kval)==missing_val;cnt[missing]=0;sm[missing]=0;unseen[missing]=True
    return cnt,sm,unseen

def te_from(cnt,sm,gm,s):return (sm+s*gm)/(cnt+s)
def rank(x):
    r=pd.Series(x).rank(method="average").to_numpy(np.float64);return (r-1)/(len(r)-1)
def folds_auc(y,p,splits):return [float(roc_auc_score(y[v],p[v])) for _,v in splits]

def cardinality_audit(train,test):
    rows=[]
    for f in FEATURES:
        tr=key_single(train[f],"exact");te=key_single(test[f],"exact");vc=pd.Series(tr).value_counts(dropna=False);freq=pd.Series(tr).map(vc).to_numpy()
        seen=pd.Series(te).isin(vc.index).to_numpy();testfreq=pd.Series(te).map(vc).fillna(0).to_numpy()
        row={"feature":f,"unique_values":len(vc),"cardinality_ratio":len(vc)/len(train),"median_frequency":vc.median(),"mean_frequency":vc.mean(),"max_frequency":vc.max(),
             "test_seen_rate":seen.mean(),"test_unseen_rate":1-seen.mean(),"mean_train_frequency_for_test_values":testfreq[seen].mean() if seen.any() else 0}
        for n in [2,5,10,20,50,100]:row[f"rows_value_freq_ge_{n}"]=np.mean(freq>=n)
        rows.append(row)
    return pd.DataFrame(rows)

def global_screen(train,y,fold_ids,splits):
    results=[]
    for feature,gran in rep_specs():
        keys=key_single(train[feature],gran);cnt=np.zeros(len(y));sm=np.zeros(len(y));un=np.zeros(len(y),bool)
        for fold,(tr,va) in enumerate(splits):
            c,z,u=map_stats(keys[tr],y[tr],keys[va],y[tr].mean());cnt[va]=c;sm[va]=z;un[va]=u
        countscore=np.log1p(cnt);cauc=roc_auc_score(y,countscore);cfs=folds_auc(y,countscore,splits)
        results.append({"encoding":"count","feature":feature,"granularity":gran,"smoothing":np.nan,"auc":max(cauc,1-cauc),"orientation":1 if cauc>=.5 else -1,
                        "fold_std":np.std(cfs),"coverage":1-un.mean(),"unseen_rate":un.mean()})
        for s in SMOOTH:
            # Fold means differ minimally; fill per fold to stay exact.
            te=np.zeros(len(y))
            for fold,(tr,va) in enumerate(splits):te[va]=te_from(cnt[va],sm[va],y[tr].mean(),s)
            fs=folds_auc(y,te,splits);results.append({"encoding":"TE","feature":feature,"granularity":gran,"smoothing":s,"auc":roc_auc_score(y,te),
                "orientation":1,"fold_std":np.std(fs),"coverage":1-un.mean(),"unseen_rate":un.mean()})
    return pd.DataFrame(results)

def make_oof_config(train,y,splits,feature,gran,smoothing,no_missing=False):
    keys=key_single(train[feature],gran);out=np.zeros(len(y));un=np.zeros(len(y),bool);counts=np.zeros(len(y))
    missing_val=MISS_CAT if feature in CATS else MISS_NUM
    for tr,va in splits:
        gm=y[tr].mean();c,z,u=map_stats(keys[tr],y[tr],keys[va],gm,no_missing,missing_val);out[va]=te_from(c,z,gm,smoothing);un[va]=u;counts[va]=c
    return out,un,counts

def stability(train,y,top):
    rows=[]
    fold_ids=np.empty(len(y),int);splits=list(StratifiedKFold(5,shuffle=True,random_state=42).split(train,y))
    for f,(_,v) in enumerate(splits):fold_ids[v]=f
    for r in top.itertuples():
        k=key_single(train[r.feature],r.granularity);d=pd.DataFrame({"k":k,"fold":fold_ids,"y":y})
        rates=d.groupby(["k","fold"]).y.agg(["mean","count"]).reset_index();pv=rates.pivot(index="k",columns="fold",values="mean")
        counts=d.groupby("k").size();std=pv.std(axis=1,ddof=0);common=pv.notna().sum(axis=1)>=2
        ws=np.average(std[common].fillna(0),weights=counts.reindex(std[common].index)) if common.any() else np.nan
        rows.append({"feature":r.feature,"granularity":r.granularity,"smoothing":r.smoothing,"auc":r.auc,"fold_std":r.fold_std,"coverage":r.coverage,
                     "cardinality":pd.Series(k).nunique(),"weighted_std_target_rate_across_folds":ws})
    return pd.DataFrame(rows)

def inner_candidates(train,y,p27,outer_train,outer_folds,topn=10):
    heap=[];serial=0
    relmap=np.full(len(y),-1,dtype=np.int64);relmap[outer_train]=np.arange(len(outer_train))
    for feature,gran in rep_specs():
        keys=key_single(train[feature],gran)
        cnt=np.zeros(len(outer_train));sm=np.zeros(len(outer_train))
        for vf in outer_folds:
            mask=FOLD_IDS[outer_train]==vf;va=outer_train[mask];tr=outer_train[~mask]
            c,z,u=map_stats(keys[tr],y[tr],keys[va],y[tr].mean());rel=relmap[va];cnt[rel]=c;sm[rel]=z
        for s in SMOOTH:
            te=np.zeros(len(outer_train))
            for vf in outer_folds:
                mask=FOLD_IDS[outer_train]==vf;va=outer_train[mask];tr=outer_train[~mask];rel=relmap[va]
                te[rel]=te_from(cnt[rel],sm[rel],y[tr].mean(),s)
            auc=roc_auc_score(y[outer_train],te);item=(auc,serial,{"feature":feature,"granularity":gran,"smoothing":s},te.astype(np.float32));serial+=1
            if len(heap)<topn:heapq.heappush(heap,item)
            elif auc>heap[0][0]:heapq.heapreplace(heap,item)
    return sorted(heap,key=lambda x:x[0],reverse=True)

def full_map_single(train,y,outer_train,outer_valid,feature,granularity,smoothing,no_missing=False):
    k=key_single(train[feature],granularity);gm=y[outer_train].mean();mv=MISS_CAT if feature in CATS else MISS_NUM
    c,z,u=map_stats(k[outer_train],y[outer_train],k[outer_valid],gm,no_missing,mv);return te_from(c,z,gm,smoothing),u,c

def nested_individual(train,y,p27,splits):
    final=np.zeros(len(y));final_nomiss=np.zeros(len(y));choices=[];test_fold_te=[];top_by_outer=[]
    for fold,(otr,ova) in enumerate(splits):
        outer_folds=[x for x in range(5) if x!=fold];tops=inner_candidates(train,y,p27,otr,outer_folds,10);top_by_outer.append(tops)
        opts=[];rb=rank(p27[otr])
        for _,_,cfg,te in tops:
            rt=rank(te)
            for method in ["probability","rank"]:
                b,t=(p27[otr],te) if method=="probability" else (rb,rt)
                for w in WEIGHTS:
                    score=roc_auc_score(y[otr],(1-w)*b+w*t);opts.append((score,-w,method,w,cfg))
        best=max(opts,key=lambda x:(x[0],x[1]));cfg=best[4]
        tv,u,c=full_map_single(train,y,otr,ova,**cfg);tvn,_,_=full_map_single(train,y,otr,ova,**cfg,no_missing=True)
        bv=p27[ova];final[ova]=(1-best[3])*bv+best[3]*tv if best[2]=="probability" else (1-best[3])*rank(bv)+best[3]*rank(tv)
        final_nomiss[ova]=(1-best[3])*bv+best[3]*tvn if best[2]=="probability" else (1-best[3])*rank(bv)+best[3]*rank(tvn)
        choices.append({"fold":fold+1,**cfg,"method":best[2],"weight":best[3],"selection_auc":best[0],"valid_coverage":1-u.mean(),"valid_unseen":u.mean()})
    return final,final_nomiss,pd.DataFrame(choices),top_by_outer

def nested_multi(train,y,p27,splits,top_by_outer):
    final=np.zeros(len(y));choices=[]
    for fold,(otr,ova) in enumerate(splits):
        selected=[];seen=set()
        for auc,serial,cfg,te in top_by_outer[fold]:
            if cfg["feature"] not in seen:selected.append((cfg,te));seen.add(cfg["feature"])
            if len(selected)==5:break
        if not selected:final[ova]=p27[ova];continue
        prob=np.mean([x[1] for x in selected],axis=0);rmean=np.mean([rank(x[1]) for x in selected],axis=0);opts=[]
        for method,sig in [("probability",prob),("rank",rmean)]:
            base=p27[otr] if method=="probability" else rank(p27[otr])
            for w in MULTI_WEIGHTS:opts.append((roc_auc_score(y[otr],(1-w)*base+w*sig),-w,method,w))
        b=max(opts,key=lambda x:(x[0],x[1]));valid_tes=[full_map_single(train,y,otr,ova,**cfg)[0] for cfg,_ in selected]
        sigv=np.mean(valid_tes,axis=0) if b[2]=="probability" else np.mean([rank(x) for x in valid_tes],axis=0)
        basev=p27[ova] if b[2]=="probability" else rank(p27[ova]);final[ova]=(1-b[3])*basev+b[3]*sigv
        choices.append({"fold":fold+1,"features":"|".join(x[0]["feature"] for x in selected),"method":b[2],"weight":b[3],"selection_auc":b[0]})
    return final,pd.DataFrame(choices)

def pair_global(train,y,splits,p27):
    rows=[];saved={}
    for a,b in PAIR_SPECS:
        for gran in ["round_0.5","round_1.0"]:
            keys=key_pair(train,a,b,gran);cnt=np.zeros(len(y));sm=np.zeros(len(y));un=np.zeros(len(y),bool)
            for tr,va in splits:
                c,z,u=map_stats(keys[tr],y[tr],keys[va],y[tr].mean());cnt[va]=c;sm[va]=z;un[va]=u
            for s in [10,50,100]:
                te=np.zeros(len(y))
                for tr,va in splits:te[va]=te_from(cnt[va],sm[va],y[tr].mean(),s)
                rows.append({"feature_a":a,"feature_b":b,"granularity":gran,"smoothing":s,"auc":roc_auc_score(y,te),"coverage":1-un.mean(),"unseen":un.mean(),
                             "pearson_exp027":pd.Series(te).corr(pd.Series(p27)),"pearson_residual":pd.Series(te).corr(pd.Series(y-p27)),"spearman_residual":pd.Series(te).corr(pd.Series(y-p27),method="spearman")})
                saved[(a,b,gran,s)]=te
    return pd.DataFrame(rows).sort_values("auc",ascending=False),saved

def main():
    global FOLD_IDS
    start=perf_counter();[d.mkdir(parents=True,exist_ok=True) for d in [REPORTS,METRICS,PRED]]
    train=pd.read_csv(DATA/"train.csv");test=pd.read_csv(DATA/"test.csv");y=train.addicted_label.to_numpy();splits=list(StratifiedKFold(5,shuffle=True,random_state=42).split(train,y))
    FOLD_IDS=np.empty(len(y),int)
    for f,(_,v) in enumerate(splits):FOLD_IDS[v]=f
    p16=load_oof(PRED/"oof_exp016_xgboost_depth5_9000.csv",train);p22=load_oof(PRED/"oof_exp022_catboost_thresholds_9000.csv",train)
    s42=load_oof(PRED/"oof_exp027_seed42.csv",train);s2026=load_oof(PRED/"oof_exp027_seed2026.csv",train);s777=load_oof(PRED/"oof_exp027_seed777.csv",train)
    s31415=load_oof(PRED/"oof_exp028_seed31415.csv",train);s1234=load_oof(PRED/"oof_exp028_seed1234.csv",train)
    p27=.75*(.4*s42+.3*s2026+.3*s777)+.25*p22;p28=.75*((s42+s2026+s777+s31415+s1234)/5)+.25*p22
    checks={"EXP016":roc_auc_score(y,p16),"EXP027":roc_auc_score(y,p27),"EXP028":roc_auc_score(y,p28)}
    if max(abs(checks[k]-v) for k,v in {"EXP016":.9657017303568276,"EXP027":EXP027,"EXP028":EXP028}.items())>2e-7:raise ValueError(f"Control OOF fallo {checks}")
    card=cardinality_audit(train,test);card.to_csv(REPORTS/"exp033_value_cardinality.csv",index=False)
    results=global_screen(train,y,FOLD_IDS,splits);results.to_csv(REPORTS/"exp033_te_results.csv",index=False)
    top20=results[results.encoding.eq("TE")].nlargest(20,"auc").copy();oofs={}
    for r in top20.itertuples():oofs[(r.feature,r.granularity,int(r.smoothing))]=make_oof_config(train,y,splits,r.feature,r.granularity,int(r.smoothing))[0]
    stab=stability(train,y,top20.head(10));stab.to_csv(REPORTS/"exp033_te_stability.csv",index=False)
    corrrows=[]
    for r in top20.itertuples():
        te=oofs[(r.feature,r.granularity,int(r.smoothing))]
        for model,p in [("EXP-016",p16),("EXP-027",p27),("EXP-028",p28),("residual_027",y-p27)]:
            corrrows.append({"feature":r.feature,"granularity":r.granularity,"smoothing":r.smoothing,"model":model,"pearson":pd.Series(te).corr(pd.Series(p)),"spearman":pd.Series(te).corr(pd.Series(p),method="spearman")})
    corrs=pd.DataFrame(corrrows);corrs.to_csv(REPORTS/"exp033_te_correlations.csv",index=False)

    nested,nested_nm,choices,topouter=nested_individual(train,y,p27,splits);multi,mchoices=nested_multi(train,y,p27,splits,topouter)
    nested_auc=roc_auc_score(y,nested);multi_auc=roc_auc_score(y,multi)
    blendrows=[{"type":"individual_result","auc":nested_auc,"delta_exp027":nested_auc-EXP027,"delta_exp028":nested_auc-EXP028},
               {"type":"no_missing_result","auc":roc_auc_score(y,nested_nm),"delta_exp027":roc_auc_score(y,nested_nm)-EXP027},
               {"type":"multi_result","auc":multi_auc,"delta_exp027":multi_auc-EXP027,"delta_exp028":multi_auc-EXP028}]
    choices.insert(0,"type","individual_choice");mchoices.insert(0,"type","multi_choice");blends=pd.concat([pd.DataFrame(blendrows),choices,mchoices],ignore_index=True);blends.to_csv(REPORTS/"exp033_te_blends.csv",index=False)
    pd.DataFrame({"id":train.id,"y_true":y,"prediction":nested,"fold":FOLD_IDS+1}).to_csv(PRED/"oof_exp033_best_te_nested.csv",index=False)

    pairres,pairoofs=pair_global(train,y,splits,p27);pairres.to_csv(REPORTS/"exp033_pair_te_results.csv",index=False)
    bestcfg=top20.iloc[0];bestte=oofs[(bestcfg.feature,bestcfg.granularity,int(bestcfg.smoothing))]
    # Test transfer uses full-train mappings only for diagnostics.
    transfers=[]
    for r in top20.itertuples():
        kt=key_single(train[r.feature],r.granularity);ke=key_single(test[r.feature],r.granularity);gm=y.mean();c,z,u=map_stats(kt,y,ke,gm);tet=te_from(c,z,gm,int(r.smoothing));oo=oofs[(r.feature,r.granularity,int(r.smoothing))]
        transfers.append({"feature":r.feature,"granularity":r.granularity,"smoothing":r.smoothing,"test_coverage":1-u.mean(),"test_unseen":u.mean(),
          "train_oof_mean":oo.mean(),"train_oof_std":oo.std(),"test_mean":tet.mean(),"test_std":tet.std(),"mean_shift":tet.mean()-oo.mean(),"std_ratio":tet.std()/oo.std() if oo.std()>0 else np.nan,
          "test_q01":np.quantile(tet,.01),"test_q50":np.quantile(tet,.5),"test_q99":np.quantile(tet,.99)})
    transfer=pd.DataFrame(transfers);transfer.to_csv(REPORTS/"exp033_test_transferability.csv",index=False)

    # Missing and regional analysis for the honest nested result.
    region=[];screen=train.daily_screen_time_hours;social=train.social_media_hours;valid=screen.notna()&social.notna();cp=valid&((screen>8)|(social>4));cn=valid&screen.le(6)&social.le(4);amb=valid&~cp&~cn
    for label,mask in [("clear_positive",cp),("clear_negative",cn),("ambiguous",amb)]:
        region.append(("region",label,mask.sum(),roc_auc_score(y[mask],p27[mask]),roc_auc_score(y[mask],nested[mask])))
    for label,lo,hi in [("0.30-0.40",.3,.4),("0.40-0.50",.4,.5),("0.50-0.60",.5,.6),("0.60-0.70",.6,.7),("0.70-0.80",.7,.8)]:
        mask=(p16>=lo)&(p16<hi);region.append(("score_band",label,mask.sum(),roc_auc_score(y[mask],p27[mask]),roc_auc_score(y[mask],nested[mask])))
    regiondf=pd.DataFrame(region,columns=["analysis","segment","rows","auc_exp027","auc_nested_te"])
    best_feature=bestcfg.feature;miss=train[best_feature].isna().to_numpy();missing_info={"best_global_feature":best_feature,"missing_rate":miss.mean(),"nested_auc":nested_auc,
      "nested_no_missing_auc":roc_auc_score(y,nested_nm),"delta_no_missing_vs_regular":roc_auc_score(y,nested_nm)-nested_auc}
    success=(nested_auc-EXP027>=.00003 and sum(np.asarray(folds_auc(y,nested,splits))>np.asarray(folds_auc(y,p27,splits)))>=4)
    residual_clear=abs(corrs[corrs.model.eq("residual_027")].pearson).max()
    # Residual correlation alone is diagnostic; require honest nested uplift before
    # recommending that the same TE be added to another target model.
    justify=max(nested_auc,multi_auc)-EXP027>=.00002
    elapsed=perf_counter()-start
    lines=["EXP-033 exact/discretized TE diagnostic",f"checks={checks}","cardinality:\n"+card.to_string(index=False),"top20:\n"+top20.to_string(index=False),
      "stability:\n"+stab.to_string(index=False),"correlations:\n"+corrs.to_string(index=False),f"individual_nested_auc={nested_auc}; choices=\n{choices.to_string(index=False)}",
      f"multi_auc={multi_auc}; choices=\n{mchoices.to_string(index=False)}","pair_top:\n"+pairres.head(15).to_string(index=False),f"missing_control={missing_info}",
      "regions_bands:\n"+regiondf.to_string(index=False),"test_transfer:\n"+transfer.to_string(index=False),f"success_direct={success}; justify_exp034={justify}; residual_max_abs_pearson={residual_clear}",
      f"total_seconds={elapsed:.2f}","problems=none"]
    (METRICS/"exp033_exact_value_te_metrics.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"nested={nested_auc:.12f} d27={nested_auc-EXP027:.12f}; multi={multi_auc:.12f}; justify={justify}; seconds={elapsed:.2f}")

if __name__=="__main__":main()
