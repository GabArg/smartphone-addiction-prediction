"""EXP-040: fold-safe dual numeric/native-categorical LightGBM experiment."""
from __future__ import annotations

import gc
import json
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
from lightgbm import LGBMClassifier
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from train_lightgbm_exp004 import prepare_aligned_categories
from train_exp035_exact_value_logistic import load_oof, safe_ratio, stringify

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "outputs"
PRED, REPORTS, METRICS, SUB = (OUT / x for x in ["predictions", "reports", "metrics", "submissions"])
ID, TARGET = "id", "addicted_label"
ORIGINAL = ["age", "gender", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
            "notifications_per_day", "app_opens_per_day", "sleep_hours", "work_study_hours",
            "weekend_screen_time", "stress_level", "academic_work_impact"]
BASE_CATS = ["gender", "stress_level", "academic_work_impact"]
FREQ = {"weekend_freq": "weekend_screen_time", "screen_freq": "daily_screen_time_hours"}
RELATIONS = {
    "social_over_screen": ("social_media_hours", "daily_screen_time_hours", "ratio"),
    "gaming_over_screen": ("gaming_hours", "daily_screen_time_hours", "ratio"),
    "work_over_screen": ("work_study_hours", "daily_screen_time_hours", "ratio"),
    "work_over_social": ("work_study_hours", "social_media_hours", "ratio"),
    "gaming_over_social": ("gaming_hours", "social_media_hours", "ratio"),
    "screen_minus_social": ("daily_screen_time_hours", "social_media_hours", "difference"),
}
RAW_CATS = {
    "screen_cat": "daily_screen_time_hours", "social_cat": "social_media_hours",
    "weekend_cat": "weekend_screen_time", "work_cat": "work_study_hours",
    "gaming_cat": "gaming_hours", "sleep_cat": "sleep_hours",
    "notifications_cat": "notifications_per_day", "appopens_cat": "app_opens_per_day",
    "age_cat": "age",
}
REL_CATS = {f"{x}_cat": x for x in RELATIONS}
PARAMS = dict(objective="binary", metric="auc", learning_rate=.03, n_estimators=10000,
              num_leaves=15, max_depth=-1, min_child_samples=100, subsample=.9,
              colsample_bytree=.9, reg_alpha=0., reg_lambda=1., random_state=42,
              # Six physical cores avoids Windows/OpenMP oversubscription observed with -1.
              n_jobs=6, verbosity=-1, max_bin=4095)
BASELINE_FULL = .966683766179125
BASELINE_ENSEMBLE = .9674680607837304


def rss(): return psutil.Process().memory_info().rss / 1024**3


def exact_key(s: pd.Series) -> pd.Series:
    z = stringify(s)
    return z.where(s.notna(), "__MISSING__").astype(str)


def relation_values(raw: pd.DataFrame) -> dict[str, pd.Series]:
    out = {}
    for name, (a, b, kind) in RELATIONS.items():
        out[name] = safe_ratio(raw[a], raw[b]) if kind == "ratio" else raw[a] - raw[b]
    return out


def add_base_features(x, raw, train_idx, reference_raw):
    x = x.copy()
    for name, col in FREQ.items():
        keys = exact_key(reference_raw[col].iloc[train_idx])
        counts = keys.value_counts(dropna=False) / len(train_idx)
        x[name] = exact_key(raw[col]).map(counts).fillna(0).to_numpy(np.float32)
    for name, values in relation_values(raw).items():
        x[name] = values.to_numpy(np.float64)
    return x


def category_codes(train_s, other_s):
    missing = "__MISSING__"
    a = exact_key(train_s)
    b = exact_key(other_s)
    vals = sorted(v for v in a.unique() if v != missing)
    mapping = {v: i for i, v in enumerate(vals)}
    missing_code, unseen_code = len(vals), len(vals) + 1
    def encode(s):
        return s.map(mapping).where(s != missing, missing_code).fillna(unseen_code).astype(np.int32)
    return encode(a), encode(b), len(vals) + 2


def build_fold(X, Xt, raw, rawt, tr, va, raw_cats, rel_cats, need_test):
    a = add_base_features(X.iloc[tr], raw.iloc[tr], tr, raw)
    b = add_base_features(X.iloc[va], raw.iloc[va], tr, raw)
    e = add_base_features(Xt, rawt, tr, raw) if need_test else None
    cardinalities = {}
    train_rel, val_rel = relation_values(raw.iloc[tr]), relation_values(raw.iloc[va])
    test_rel = relation_values(rawt) if need_test else {}
    for name in raw_cats:
        col = RAW_CATS[name]
        ca, cb, card = category_codes(raw[col].iloc[tr], raw[col].iloc[va])
        a[name], b[name], cardinalities[name] = ca.to_numpy(), cb.to_numpy(), card
        if need_test:
            _, ce, _ = category_codes(raw[col].iloc[tr], rawt[col]); e[name] = ce.to_numpy()
    for name in rel_cats:
        rel = REL_CATS[name]
        decimals = 1 if rel == "screen_minus_social" else 2
        ta, vb = train_rel[rel].round(decimals), val_rel[rel].round(decimals)
        ca, cb, card = category_codes(ta, vb)
        a[name], b[name], cardinalities[name] = ca.to_numpy(), cb.to_numpy(), card
        if need_test:
            _, ce, _ = category_codes(ta, test_rel[rel].round(decimals)); e[name] = ce.to_numpy()
    return a, b, e, cardinalities


def train_cfg(ctx, label, raw_cats=(), rel_cats=(), fold_ids=(0,1,2), updates=None, need_test=False):
    X, Xt, raw, rawt, y, splits = ctx
    oof = np.full(len(y), np.nan); tests, rows, gains = [], [], []
    params = {**PARAMS, **(updates or {})}
    for f in fold_ids:
        tr, va = splits[f]; start = perf_counter()
        a, b, e, cards = build_fold(X, Xt, raw, rawt, tr, va, raw_cats, rel_cats, need_test)
        cat_features = BASE_CATS + list(raw_cats) + list(rel_cats)
        model = LGBMClassifier(**params)
        model.fit(a, y[tr], eval_set=[(b, y[va])], eval_metric="auc",
                  categorical_feature=cat_features,
                  callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(0)])
        best = int(model.best_iteration_); pred = model.predict_proba(b, num_iteration=best)[:,1]
        oof[va] = pred
        if need_test: tests.append(model.predict_proba(e, num_iteration=best)[:,1])
        gain = model.booster_.feature_importance(importance_type="gain")
        gains.append(pd.DataFrame({"feature": model.booster_.feature_name(), "gain": gain, "fold": f+1}))
        row = dict(label=label, fold=f+1, auc=roc_auc_score(y[va], pred), best_iteration=best,
                   seconds=perf_counter()-start, rss_gb=rss(), raw_cats="+".join(raw_cats),
                   rel_cats="+".join(rel_cats), max_bin=params["max_bin"],
                   cat_smooth=params.get("cat_smooth", 10),
                   min_data_per_group=params.get("min_data_per_group", 100),
                   cardinalities=json.dumps(cards, sort_keys=True))
        rows.append(row); print(f"{label} fold={f+1} auc={row['auc']:.8f} best={best} sec={row['seconds']:.1f} ram={row['rss_gb']:.2f}", flush=True)
        del a,b,e,model; gc.collect()
    return oof, np.asarray(tests), rows, pd.concat(gains, ignore_index=True)


def summary(label, oof, y, splits, fold_ids, rows, baseline=None):
    idx = np.concatenate([splits[f][1] for f in fold_ids]); fs = np.array([roc_auc_score(y[splits[f][1]],oof[splits[f][1]]) for f in fold_ids])
    d = dict(label=label, oof=roc_auc_score(y[idx],oof[idx]), delta=0., folds_improved=0,
             fold_aucs=json.dumps(fs.tolist()), std=fs.std(), best_iterations=json.dumps([x["best_iteration"] for x in rows]),
             seconds=sum(x["seconds"] for x in rows), peak_rss_gb=max(x["rss_gb"] for x in rows))
    if baseline is not None:
        d["delta"] = d["oof"]-baseline[0]; d["folds_improved"] = int(np.sum(fs > baseline[1]))
        d["fold_deltas"] = json.dumps((fs-baseline[1]).tolist())
    return d


def promising(s):
    ds = np.array(json.loads(s.get("fold_deltas", "[]")))
    return s["delta"] >= .00003 or (s["folds_improved"] == 3 and np.all(ds >= .000015))


def rank(p): return rankdata(p, method="average") / len(p)


def refs(train):
    r = {"EXP-016":load_oof(PRED/"oof_exp016_xgboost_depth5_9000.csv",train),
         "EXP-022":load_oof(PRED/"oof_exp022_catboost_thresholds_9000.csv",train),
         "EXP-037":load_oof(PRED/"oof_exp037_relational_logistic.csv",train),
         "EXP-039":load_oof(PRED/"oof_exp039_lgbm_highbin.csv",train)}
    seeds=[load_oof(PRED/f"oof_exp027_seed{x}.csv",train) for x in [42,2026,777]]
    r["EXP-027"] = .75*(.4*seeds[0]+.3*seeds[1]+.3*seeds[2])+.25*r["EXP-022"]
    return r


def nested_candidates(y, splits, candidates):
    out=np.zeros(len(y)); choices=[]
    for f,(tr,va) in enumerate(splits,1):
        scored=[(roc_auc_score(y[tr],p[tr]),name,p) for name,p in candidates]
        z=max(scored,key=lambda q:q[0]); out[va]=z[2][va]; choices.append((f,z[1],z[0]))
    return out, choices


def pairwise_rows(y, base16, models):
    pos=np.where(y==1)[0]; neg=np.where(y==0)[0]; rng=np.random.default_rng(29042)
    pi=rng.choice(pos,125287,replace=True); ni=rng.choice(neg,125287,replace=True)
    bad=base16[pi] <= base16[ni]
    # Matching correct-close sample exactly follows EXP-029's conceptual comparison.
    pi2=rng.choice(pos,125287,replace=True); ni2=rng.choice(neg,125287,replace=True)
    correct=base16[pi2] > base16[ni2]
    rows=[]
    for name,p in models.items():
        corrected=np.mean(p[pi[bad]]>p[ni[bad]]) if bad.any() else np.nan
        broken=np.mean(p[pi2[correct]]<=p[ni2[correct]]) if correct.any() else np.nan
        rows.append(dict(model=name, corrected_misordered=corrected, broken_correct=broken, net_pair_gain=corrected-broken,
                         misordered_pairs=int(bad.sum()), correct_pairs=int(correct.sum())))
    return rows


def main():
    total=perf_counter()
    for d in [PRED,REPORTS,METRICS,SUB]: d.mkdir(parents=True,exist_ok=True)
    train,test=pd.read_csv(DATA/"train.csv"),pd.read_csv(DATA/"test.csv"); y=train[TARGET].to_numpy()
    raw,rawt=train[ORIGINAL].copy(),test[ORIGINAL].copy(); X,Xt=prepare_aligned_categories(raw,rawt,BASE_CATS)
    splits=list(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y)); ctx=(X,Xt,raw,rawt,y,splits)
    progress=[]
    def run(label,rc=(),lc=(),folds=(0,1,2),updates=None,test_pred=False):
        o,t,rows,g=train_cfg(ctx,label,rc,lc,folds,updates,test_pred); progress.extend(rows)
        pd.DataFrame(progress).to_csv(REPORTS/"exp040_all_fold_progress.csv",index=False)
        return o,t,rows,g

    # Phase 1: exact EXP-039 reproduction.
    boof,_,brows,_=run("EXP039_REPRO",folds=(0,1,2,3,4)); bs=summary("EXP039_REPRO",boof,y,splits,range(5),brows)
    if abs(bs["oof"]-BASELINE_FULL) > .00002:
        (METRICS/"exp040_dual_repr_metrics.txt").write_text(f"STOPPED: EXP-039 failed reproduction\nobserved={bs}\nexpected={BASELINE_FULL}\n",encoding="utf-8")
        raise RuntimeError(f"EXP-039 reproduction mismatch: {bs['oof']:.9f}")
    idx3=np.concatenate([splits[f][1] for f in range(3)]); b3=roc_auc_score(y[idx3],boof[idx3]); bf=np.array([roc_auc_score(y[splits[f][1]],boof[splits[f][1]]) for f in range(3)])
    baseline3=(b3,bf)

    # Raw exact categorical screening.
    raw_rows=[]; raw_cache={}
    cards={n:int(raw[c].nunique(dropna=False)+2) for n,c in RAW_CATS.items()}
    for n in RAW_CATS:
        o,_,rr,_=run("raw_"+n,(n,)); s=summary(n,o,y,splits,range(3),rr,baseline3); s["cardinality"]=cards[n]; s["promising"]=promising(s); raw_rows.append(s); raw_cache[(n,)]=o
    raw_df=pd.DataFrame(raw_rows).sort_values("delta",ascending=False); raw_df.to_csv(REPORTS/"exp040_raw_cat_screen.csv",index=False)
    positive=list(raw_df.loc[raw_df.delta>0,"label"]); ordered=list(raw_df.label)
    sets=[]
    for k in [1,2,3,5]: sets.append((f"TOP{k}",tuple(ordered[:k])))
    sets.append(("ALL_POSITIVE",tuple(positive)))
    set_rows=[]; set_cache={}
    for label,names in sets:
        if not names: continue
        if names in set_cache: o,rr=set_cache[names]
        elif len(names)==1: o=raw_cache[names]; rr=[x for x in progress if x["label"]=="raw_"+names[0]]; set_cache[names]=(o,rr)
        else: o,_,rr,_=run("rawset_"+label,names); set_cache[names]=(o,rr)
        s=summary(label,o,y,splits,range(3),rr,baseline3); s["features"]="+".join(names); set_rows.append(s)
    best_raw=max(set_rows,key=lambda z:z["oof"]); best_raw_names=tuple(best_raw["features"].split("+"))

    # Relational categorical screening.
    rel_rows=[]; rel_cache={}
    for n in REL_CATS:
        o,_,rr,_=run("rel_"+n,(),(n,)); s=summary(n,o,y,splits,range(3),rr,baseline3); s["promising"]=promising(s); rel_rows.append(s); rel_cache[n]=o
    rel_df=pd.DataFrame(rel_rows).sort_values("delta",ascending=False); rel_df.to_csv(REPORTS/"exp040_relational_cat_screen.csv",index=False)
    best_rel_names=tuple(rel_df.loc[rel_df.promising,"label"])

    # Dual A/B/C (BASE is represented in reproduction row).
    dual_specs=[("DUAL-A",best_raw_names,()),("DUAL-B",(),best_rel_names),("DUAL-C",best_raw_names,best_rel_names)]
    dual_rows=[]; dual_cache={}
    for label,rc,lc in dual_specs:
        if not rc and not lc: continue
        o,_,rr,_=run(label,rc,lc); s=summary(label,o,y,splits,range(3),rr,baseline3); s["raw_features"]="+".join(rc);s["rel_features"]="+".join(lc);dual_rows.append(s);dual_cache[label]=(o,rc,lc)
    dual_df=pd.DataFrame(dual_rows); dual_df.to_csv(REPORTS/"exp040_dual_sets.csv",index=False)
    best_dual=max(dual_rows,key=lambda z:z["oof"]); best_label=best_dual["label"]; _,best_rc,best_lc=dual_cache[best_label]

    # Sequential categorical parameter and max-bin screens, only on improvement.
    cat_rows=[]; best_updates={}
    if best_dual["delta"]>0:
        for smooth in [10,20,50,100]:
            o,_,rr,_=run(f"catsmooth_{smooth}",best_rc,best_lc,updates={"cat_smooth":smooth}); s=summary(str(smooth),o,y,splits,range(3),rr,baseline3);s["stage"]="cat_smooth";s["cat_smooth"]=smooth;cat_rows.append(s)
        best_s=int(max(cat_rows,key=lambda z:z["oof"])["cat_smooth"]); best_updates["cat_smooth"]=best_s
        for md in [50,100,200]:
            o,_,rr,_=run(f"mindata_{md}",best_rc,best_lc,updates={**best_updates,"min_data_per_group":md});s=summary(str(md),o,y,splits,range(3),rr,baseline3);s["stage"]="min_data_per_group";s["cat_smooth"]=best_s;s["min_data_per_group"]=md;cat_rows.append(s)
        best_m=int(max([z for z in cat_rows if z["stage"]=="min_data_per_group"],key=lambda z:z["oof"])["min_data_per_group"]);best_updates["min_data_per_group"]=best_m
    pd.DataFrame(cat_rows, columns=(list(cat_rows[0]) if cat_rows else ["stage","cat_smooth","min_data_per_group","oof","delta","status"])).to_csv(REPORTS/"exp040_cat_params.csv",index=False)
    mb_rows=[]
    if best_dual["delta"]>0:
        for mb in [2047,4095]:
            o,_,rr,_=run(f"maxbin_{mb}",best_rc,best_lc,updates={**best_updates,"max_bin":mb});s=summary(str(mb),o,y,splits,range(3),rr,baseline3);s["max_bin"]=mb;mb_rows.append(s)
        best_updates["max_bin"]=int(max(mb_rows,key=lambda z:z["oof"])["max_bin"])
    pd.DataFrame(mb_rows, columns=(list(mb_rows[0]) if mb_rows else ["max_bin","oof","delta","status"])).to_csv(REPORTS/"exp040_maxbin_check.csv",index=False)

    # Full CV gate.
    eligible=[z for z in dual_rows if z["delta"]>=.00010]
    if not eligible and best_dual["delta"]>=.00005: eligible=[best_dual]
    final_oof,final_test,final_s,final_gains=None,None,None,None
    if eligible:
        chosen=max(eligible,key=lambda z:z["oof"]); _,frc,flc=dual_cache[chosen["label"]]
        final_oof,tests,rr,final_gains=run("EXP040_FINAL",frc,flc,folds=(0,1,2,3,4),updates=best_updates,test_pred=True)
        final_test=tests.mean(axis=0);final_s=summary("EXP040_FINAL",final_oof,y,splits,range(5),rr,(BASELINE_FULL,np.array([roc_auc_score(y[v],boof[v]) for _,v in splits])))
    else:
        # Close experiment without manufacturing a final model that failed its gate.
        final_oof=dual_cache[best_label][0]; frc,flc=best_rc,best_lc

    # Importance exists only for the valid full model.
    imp=pd.DataFrame()
    if final_gains is not None:
        imp=final_gains.groupby("feature",as_index=False).gain.sum()
        def family(f):
            if f in RAW_CATS:return "exact categorical copies"
            if f in REL_CATS:return "relational categorical copies"
            if f in FREQ:return "frequency features"
            if f in RELATIONS:return "relations numeric"
            return "original numeric/categorical"
        imp["family"]=imp.feature.map(family);imp["gain_share"]=imp.gain/imp.gain.sum();imp=imp.sort_values("gain",ascending=False)
    if imp.empty: imp=pd.DataFrame(columns=["feature","gain","family","gain_share","status"])
    imp.to_csv(REPORTS/"exp040_feature_importance.csv",index=False)

    # Full diagnostics/blending only when a 5-fold EXP-040 exists.
    corr=[];regional=[];pair=[];blends=[];submission=False;sub_path=""
    if final_s is not None:
        pd.DataFrame({ID:train[ID],"y_true":y,"oof_prediction":final_oof}).to_csv(PRED/"oof_exp040_dual_lgbm.csv",index=False)
        pd.DataFrame({ID:test[ID],"prediction":final_test}).to_csv(PRED/"test_exp040_dual_lgbm.csv",index=False)
        r=refs(train)
        for n,p in r.items(): corr.append(dict(comparison=n,pearson=np.corrcoef(final_oof,p)[0,1],spearman=pd.Series(final_oof).corr(pd.Series(p),method="spearman")))
        for n in ["EXP-027","EXP-039"]: corr.append(dict(comparison="residual_"+n,pearson=np.corrcoef(y-final_oof,y-r[n])[0,1],spearman=pd.Series(y-final_oof).corr(pd.Series(y-r[n]),method="spearman")))
        screen,social=train.daily_screen_time_hours,train.social_media_hours;known=screen.notna()&social.notna();regions={"clear_positive":known&((screen>8)|(social>4)),"clear_negative":known&(screen<=6)&(social<=4)};regions["ambiguous"]=known&~regions["clear_positive"]&~regions["clear_negative"]
        for seg,m in regions.items():
            for n,p in [("EXP-040",final_oof),("EXP-039",r["EXP-039"]),("EXP-027",r["EXP-027"])]:regional.append(dict(analysis="region",segment=seg,model=n,rows=int(m.sum()),auc=roc_auc_score(y[m],p[m])))
        for lo in [.3,.4,.5,.6,.7]:
            m=(r["EXP-016"]>=lo)&(r["EXP-016"]<lo+.1)
            for n,p in [("EXP-040",final_oof),("EXP-039",r["EXP-039"]),("EXP-027",r["EXP-027"])]:regional.append(dict(analysis="EXP016_band",segment=f"{lo:.2f}-{lo+.1:.2f}",model=n,rows=int(m.sum()),auc=roc_auc_score(y[m],p[m])))
        pair=pairwise_rows(y,r["EXP-016"],{"EXP-040":final_oof,"EXP-039":r["EXP-039"],"EXP-037":r["EXP-037"],"EXP-022":r["EXP-022"]})
        diverse=abs(np.corrcoef(final_oof,r["EXP-039"])[0,1])<.9999
        if final_s["oof"]>=BASELINE_FULL-.00010 and diverse:
            c=[]
            for method in ["probability","rank"]:
                a,b=(r["EXP-039"],final_oof) if method=="probability" else (rank(r["EXP-039"]),rank(final_oof))
                for w in np.arange(.1,1,.1): c.append((f"pair_{method}_w40_{w:.2f}",(1-w)*a+w*b))
            po,choices=nested_candidates(y,splits,c); blends.append(dict(stage="pair",oof=roc_auc_score(y,po),choices=json.dumps(choices)))
            c=[]
            for method in ["probability","rank"]:
                q={n:(p if method=="probability" else rank(p)) for n,p in {**r,"EXP-040":final_oof}.items()}
                for w in [.05,.10,.15,.20,.25,.30,.40]:
                    p=(1-w)*(.375*q["EXP-027"]+.225*q["EXP-037"]+.4*q["EXP-039"])+w*q["EXP-040"]
                    c.append((f"quad_{method}_w40_{w:.2f}",p))
            qo,qchoices=nested_candidates(y,splits,c); qfs=np.array([roc_auc_score(y[v],qo[v]) for _,v in splits]); blends.append(dict(stage="quad",oof=roc_auc_score(y,qo),choices=json.dumps(qchoices),folds=json.dumps(qfs.tolist())))
            basefs=np.array([.9668479888956474,.967439309426724,.9676534694408027,.9681823431103556,.9672203307563628])
            if roc_auc_score(y,qo)>=BASELINE_ENSEMBLE+.00010 and np.sum(qfs>basefs)>=4 and np.all(qfs-basefs>=-.00005):
                # Test prediction uses fold-selected formulas and averages the five fold-specific test blends.
                rt={"EXP-027":.75*(.4*pd.read_csv(PRED/"test_exp027_seed42.csv").iloc[:,-1].to_numpy()+.3*pd.read_csv(PRED/"test_exp027_seed2026.csv").iloc[:,-1].to_numpy()+.3*pd.read_csv(PRED/"test_exp027_seed777.csv").iloc[:,-1].to_numpy())+.25*pd.read_csv(PRED/"test_exp022_catboost_thresholds_9000.csv").iloc[:,-1].to_numpy(),
                    "EXP-037":pd.read_csv(PRED/"test_exp037_relational_logistic.csv").iloc[:,-1].to_numpy(),"EXP-039":pd.read_csv(PRED/"test_exp039_lgbm_highbin.csv").iloc[:,-1].to_numpy(),"EXP-040":final_test}
                tp=[]
                for _,choice,_ in qchoices:
                    method=choice.split("_")[1];w=float(choice.rsplit("_",1)[1]);q={n:(p if method=="probability" else rank(p)) for n,p in rt.items()};tp.append((1-w)*(.375*q["EXP-027"]+.225*q["EXP-037"]+.4*q["EXP-039"])+w*q["EXP-040"])
                sub_path=str(SUB/"submission_exp040_dual_repr_lgbm_ensemble.csv");pd.DataFrame({ID:test[ID],TARGET:np.mean(tp,axis=0)}).to_csv(sub_path,index=False);submission=True
    pd.DataFrame(corr,columns=(list(corr[0]) if corr else ["comparison","pearson","spearman","status"])).to_csv(REPORTS/"exp040_correlations.csv",index=False)
    pd.DataFrame(regional,columns=(list(regional[0]) if regional else ["analysis","segment","model","rows","auc","status"])).to_csv(REPORTS/"exp040_regional.csv",index=False)
    pd.DataFrame(pair,columns=(list(pair[0]) if pair else ["model","corrected_misordered","broken_correct","net_pair_gain","status"])).to_csv(REPORTS/"exp040_pairwise.csv",index=False)
    pd.DataFrame(blends,columns=(list(blends[0]) if blends else ["stage","method","weights","oof","delta","status"])).to_csv(REPORTS/"exp040_blends.csv",index=False)
    lines=["EXP-040 dual representation LightGBM",f"reproduction={bs}",f"baseline_3fold={b3};folds={bf.tolist()}",f"raw_screen=\n{raw_df.to_string(index=False)}",f"raw_sets=\n{pd.DataFrame(set_rows).to_string(index=False)}",f"relational_screen=\n{rel_df.to_string(index=False)}",f"dual_sets=\n{dual_df.to_string(index=False)}",f"cat_params=\n{pd.DataFrame(cat_rows).to_string(index=False)}",f"maxbin=\n{pd.DataFrame(mb_rows).to_string(index=False)}",f"final={final_s}",f"features_raw={frc};features_rel={flc};params={PARAMS|best_updates}",f"importance_top30=\n{imp.head(30).to_string(index=False)}",f"importance_family=\n{imp.groupby('family').gain.sum().sort_values(ascending=False).to_string() if not imp.empty else 'not run'}",f"correlations=\n{pd.DataFrame(corr).to_string(index=False)}",f"regional=\n{pd.DataFrame(regional).to_string(index=False)}",f"pairwise=\n{pd.DataFrame(pair).to_string(index=False)}",f"blends=\n{pd.DataFrame(blends).to_string(index=False)}",f"submission={submission};path={sub_path}",f"total_seconds={perf_counter()-total}","problems=none"]
    (METRICS/"exp040_dual_repr_metrics.txt").write_text("\n\n".join(lines),encoding="utf-8")
    if submission:
        log=pd.read_csv(METRICS/"experiment_log.csv"); row={c:"" for c in log.columns};row.update({"experiment":"EXP-040","model":"dual_repr_lgbm_ensemble","cv_auc":blends[-1]["oof"],"submission_file":sub_path});pd.concat([log,pd.DataFrame([row])],ignore_index=True).to_csv(METRICS/"experiment_log.csv",index=False)
    print(f"DONE final={final_s} submission={submission} seconds={perf_counter()-total:.1f}",flush=True)


if __name__ == "__main__": main()
