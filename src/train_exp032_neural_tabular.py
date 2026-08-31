"""EXP-032: fold-safe embedded-categorical MLP and OOF diversity diagnosis."""

from __future__ import annotations

import copy
import gc
import os
import platform
import random
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_xgboost_exp012_threshold_features import NEW_FEATURES, add_threshold_features


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"; OUT = ROOT / "outputs"; PRED = OUT / "predictions"
REPORTS = OUT / "reports"; METRICS = OUT / "metrics"; SUB = OUT / "submissions"
CATS = ["gender", "stress_level", "academic_work_impact"]
WEIGHTS = [.01, .02, .03, .05, .075, .10, .15, .20]
TRIPLE_WEIGHTS = [.02, .03, .05, .075, .10]
EXP027 = .965919188052602; EXP028 = .9659307051390922


def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def rss_gb(): return psutil.Process().memory_info().rss / 1024**3


def hardware():
    vm = psutil.virtual_memory(); cuda = torch.cuda.is_available()
    return {"cpu": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
            "physical_cores": psutil.cpu_count(logical=False), "logical_cores": psutil.cpu_count(),
            "ram_total_gb": vm.total/1024**3, "ram_available_gb": vm.available/1024**3,
            "python": platform.python_version(), "pytorch": torch.__version__, "cuda_available": cuda,
            "gpu": torch.cuda.get_device_name(0) if cuda else "none",
            "vram_total_gb": torch.cuda.get_device_properties(0).total_memory/1024**3 if cuda else 0.,
            "torch_cuda": torch.version.cuda, "device": "cuda" if cuda else "cpu"}


class TabularMLP(nn.Module):
    def __init__(self, n_numeric, cardinalities):
        super().__init__()
        dims = [min(16, max(2, round(card**0.5 * 2))) for card in cardinalities]
        self.embeddings = nn.ModuleList([nn.Embedding(card, dim) for card, dim in zip(cardinalities, dims)])
        total = n_numeric + sum(dims)
        self.mlp = nn.Sequential(nn.Linear(total,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(.20),
                                 nn.Linear(256,128),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(.15),
                                 nn.Linear(128,64),nn.ReLU(),nn.Dropout(.10),nn.Linear(64,1))
    def forward(self, xnum, xcat):
        embs = [emb(xcat[:,i]) for i,emb in enumerate(self.embeddings)]
        return self.mlp(torch.cat([xnum]+embs,dim=1)).squeeze(1)


def load_oof(path, train):
    df=pd.read_csv(path)
    if not df.id.equals(train.id) or not df.y_true.equals(train.addicted_label) or df.oof_prediction.isna().any():
        raise ValueError(f"OOF invalida: {path.name}")
    return df.oof_prediction.to_numpy(np.float64)


def prep_fold(raw, raw_test, tr, va, numeric, with_missing):
    med = raw.iloc[tr][numeric].median()
    tr_num = raw.iloc[tr][numeric].fillna(med).to_numpy(np.float32)
    va_num = raw.iloc[va][numeric].fillna(med).to_numpy(np.float32)
    te_num = raw_test[numeric].fillna(med).to_numpy(np.float32)
    scaler=StandardScaler(); tr_num=scaler.fit_transform(tr_num).astype(np.float32)
    va_num=scaler.transform(va_num).astype(np.float32); te_num=scaler.transform(te_num).astype(np.float32)
    if with_missing:
        tr_num=np.concatenate([tr_num,raw.iloc[tr][numeric].isna().to_numpy(np.float32)],axis=1)
        va_num=np.concatenate([va_num,raw.iloc[va][numeric].isna().to_numpy(np.float32)],axis=1)
        te_num=np.concatenate([te_num,raw_test[numeric].isna().to_numpy(np.float32)],axis=1)
    cats_tr=[];cats_va=[];cats_te=[];cards=[]
    for c in CATS:
        vals=raw.iloc[tr][c].fillna("__MISSING__").astype(str)
        observed=sorted(v for v in vals.unique() if v not in {"__MISSING__","__UNKNOWN__"})
        mapping={"__MISSING__":0,"__UNKNOWN__":1};mapping.update({v:i+2 for i,v in enumerate(observed)})
        def encode(s):
            s=s.fillna("__MISSING__").astype(str);return s.map(mapping).fillna(1).to_numpy(np.int64)
        cats_tr.append(encode(raw.iloc[tr][c]));cats_va.append(encode(raw.iloc[va][c]));cats_te.append(encode(raw_test[c]));cards.append(len(mapping))
    return tr_num,va_num,te_num,np.column_stack(cats_tr),np.column_stack(cats_va),np.column_stack(cats_te),cards


@torch.no_grad()
def predict(model,xnum,xcat,device,batch):
    model.eval();out=[]
    loader=DataLoader(TensorDataset(torch.from_numpy(xnum),torch.from_numpy(xcat)),batch_size=batch,shuffle=False,num_workers=0)
    for nb,cb in loader:
        out.append(torch.sigmoid(model(nb.to(device),cb.to(device))).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def train_fold(raw,raw_test,y,tr,va,numeric,with_missing,max_epochs,patience,device,batch,fold,variant):
    seed_all(42); start=perf_counter(); peak=rss_gb()
    tn,vn,en,tc,vc,ec,cards=prep_fold(raw,raw_test,tr,va,numeric,with_missing);peak=max(peak,rss_gb())
    model=TabularMLP(tn.shape[1],cards).to(device);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",factor=.5,patience=2)
    loss_fn=nn.BCEWithLogitsLoss();generator=torch.Generator().manual_seed(42)
    loader=DataLoader(TensorDataset(torch.from_numpy(tn),torch.from_numpy(tc),torch.from_numpy(y[tr].astype(np.float32))),
                      batch_size=batch,shuffle=True,num_workers=0,generator=generator,drop_last=False)
    best_auc=-np.inf;best_epoch=0;best_state=None;wait=0;epoch_times=[];history=[]
    for epoch in range(1,max_epochs+1):
        et=perf_counter();model.train()
        for nb,cb,yb in loader:
            opt.zero_grad(set_to_none=True);logits=model(nb.to(device),cb.to(device));loss=loss_fn(logits,yb.to(device));loss.backward();opt.step()
        pv=predict(model,vn,vc,device,batch);auc=float(roc_auc_score(y[va],pv));scheduler.step(auc)
        epoch_times.append(perf_counter()-et);history.append((epoch,auc,opt.param_groups[0]["lr"]));peak=max(peak,rss_gb())
        if auc>best_auc+1e-7:
            best_auc=auc;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};wait=0
        else: wait+=1
        print(f"{variant} fold={fold} epoch={epoch} auc={auc:.8f} seconds={epoch_times[-1]:.1f}",flush=True)
        if wait>=patience:break
    model.load_state_dict(best_state);model.to(device);pv=predict(model,vn,vc,device,batch);pt=predict(model,en,ec,device,batch)
    elapsed=perf_counter()-start
    result={"variant":variant,"fold":fold,"auc":roc_auc_score(y[va],pv),"best_epoch":best_epoch,"epochs_run":len(history),
            "seconds":elapsed,"mean_epoch_seconds":np.mean(epoch_times),"peak_rss_gb":peak,
            "peak_vram_gb":torch.cuda.max_memory_allocated()/1024**3 if device.type=="cuda" else 0.}
    del loader,tn,vn,en,tc,vc,ec,best_state;gc.collect()
    if device.type=="cuda":torch.cuda.empty_cache()
    return pv,pt,result


def rank(x):
    r=pd.Series(x).rank(method="average").to_numpy(np.float64);return (r-1)/(len(r)-1)
def folds_auc(y,p,splits):return [float(roc_auc_score(y[v],p[v])) for _,v in splits]


def nested_blend(y,base,neural,splits,base_name,weights=WEIGHTS,triple=False,xgb=None,cat=None):
    nested=np.zeros(len(y));choices=[];diagnostics=[]
    rb,rn=rank(base),rank(neural)
    if triple: rx,rc=rank(xgb),rank(cat)
    for method in ["probability","rank"]:
        for nw in weights:
            if triple:
                p=(1-nw)*(.75*xgb+.25*cat)+nw*neural if method=="probability" else (1-nw)*(.75*rx+.25*rc)+nw*rn
            else:p=(1-nw)*base+nw*neural if method=="probability" else (1-nw)*rb+nw*rn
            diagnostics.append({"base":base_name,"kind":"global_diagnostic","method":method,"neural_weight":nw,"auc":roc_auc_score(y,p)})
    for f,(tr,va) in enumerate(splits,1):
        opts=[]
        for method in ["probability","rank"]:
            for nw in weights:
                if triple:p=(1-nw)*(.75*xgb+.25*cat)+nw*neural if method=="probability" else (1-nw)*(.75*rx+.25*rc)+nw*rn
                else:p=(1-nw)*base+nw*neural if method=="probability" else (1-nw)*rb+nw*rn
                opts.append((roc_auc_score(y[tr],p[tr]),-nw,method,nw,p))
        b=max(opts,key=lambda z:(z[0],z[1]));nested[va]=b[4][va];choices.append({"base":base_name,"kind":"nested_choice","fold":f,"method":b[2],"neural_weight":b[3],"selection_auc":b[0]})
    fs=folds_auc(y,nested,splits);res={"base":base_name,"kind":"nested_result","method":"fold_specific","auc":roc_auc_score(y,nested),
        "fold1":fs[0],"fold2":fs[1],"fold3":fs[2],"fold4":fs[3],"fold5":fs[4],"std":np.std(fs)}
    return nested,choices,res,diagnostics


def main():
    start_all=perf_counter();hw=hardware();device=torch.device(hw["device"]);batch=4096 if device.type=="cuda" else 1024
    torch.set_num_threads(min(12,os.cpu_count() or 1));seed_all()
    for d in [PRED,REPORTS,METRICS,SUB]:d.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(DATA/"train.csv");test=pd.read_csv(DATA/"test.csv");y=train.addicted_label.to_numpy()
    originals=[c for c in train if c not in {"id","addicted_label"}];raw=add_threshold_features(train[originals]);rawt=add_threshold_features(test[originals])
    if raw.columns.tolist()!=rawt.columns.tolist() or len(raw.columns)!=len(originals)+len(NEW_FEATURES):raise ValueError("Features no coinciden con EXP-016")
    numeric=[c for c in raw if c not in CATS];splits=list(StratifiedKFold(5,shuffle=True,random_state=42).split(raw,y))
    fold_rows=[];variant_data={};problems=[]

    # Viability benchmark: Fold 1 NN-NOMISS, max 20, patience 4.
    tr1,va1=splits[0];bpv,bpt,bres=train_fold(raw,rawt,y,tr1,va1,numeric,False,20,4,device,batch,1,"NN-NOMISS-BENCH")
    fold_rows.append({**bres,"stage":"benchmark"});bench_minutes=bres["seconds"]/60
    viable=not (bench_minutes>60 or (bench_minutes>30 and bres["auc"]<.950) or bres["auc"]<.940)
    if not viable:
        pd.DataFrame(fold_rows).to_csv(REPORTS/"exp032_neural_folds.csv",index=False)
        pd.DataFrame(columns=["model","pearson","spearman"]).to_csv(REPORTS/"exp032_neural_correlations.csv",index=False)
        pd.DataFrame(columns=["analysis","segment","model","rows","auc"]).to_csv(REPORTS/"exp032_neural_regional.csv",index=False)
        pd.DataFrame(columns=["base","kind","method","neural_weight","auc"]).to_csv(REPORTS/"exp032_neural_blends.csv",index=False)
        future={"FT-Transformer":"VIABLE PERO COSTOSO (CPU; GPU recomendable)","TabM":"REQUIERE GPU/NUBE RECOMENDABLE",
                "TabNet":"VIABLE PERO COSTOSO (GPU recomendable)","TabPFN":"NO ADECUADO PARA 691k FILAS"}
        lines=["EXP-032 stopped after viability benchmark",f"hardware={hw}",f"benchmark={bres}",f"viable={viable}",
               "full_cv=false; nn_miss=false; ensembles=false; submission=false",f"future={future}",
               f"total_seconds={perf_counter()-start_all:.2f}","problems=quality criterion failed: Fold1 AUC < 0.940"]
        (METRICS/"exp032_neural_metrics.txt").write_text("\n".join(lines)+"\n",encoding="utf-8");print(lines);return
    del bpv,bpt;gc.collect()

    def full_variant(name,with_missing,first_only=False):
        oof=np.zeros(len(train));test_folds=[];rows=[]
        use_splits=splits[:1] if first_only else splits
        for f,(tr,va) in enumerate(use_splits,1):
            pv,pt,res=train_fold(raw,rawt,y,tr,va,numeric,with_missing,30,5,device,batch,f,name);oof[va]=pv;test_folds.append(pt);rows.append({**res,"stage":"full"})
            pd.DataFrame(fold_rows+rows).to_csv(REPORTS/"exp032_neural_folds.csv",index=False)
        return oof,np.asarray(test_folds),rows

    nomiss_oof,nomiss_testfolds,nomiss_rows=full_variant("NN-NOMISS",False);fold_rows+=nomiss_rows
    nomiss_auc=roc_auc_score(y,nomiss_oof);variant_data["NN-NOMISS"]=(nomiss_oof,nomiss_testfolds,nomiss_rows,nomiss_auc)
    if nomiss_auc>=.945:
        miss1_oof,miss1_test,miss1_rows=full_variant("NN-MISS",True,True);fold_rows+=miss1_rows
        if miss1_rows[0]["auc"]>=nomiss_rows[0]["auc"]-.0001:
            # Reuse Fold1, complete only folds 2-5.
            miss_oof=miss1_oof.copy();miss_tests=[miss1_test[0]];miss_rows=list(miss1_rows)
            for f,(tr,va) in enumerate(splits[1:],2):
                pv,pt,res=train_fold(raw,rawt,y,tr,va,numeric,True,30,5,device,batch,f,"NN-MISS");miss_oof[va]=pv;miss_tests.append(pt);miss_rows.append({**res,"stage":"full"});fold_rows.append({**res,"stage":"full"})
                pd.DataFrame(fold_rows).to_csv(REPORTS/"exp032_neural_folds.csv",index=False)
            variant_data["NN-MISS"]=(miss_oof,np.asarray(miss_tests),miss_rows,roc_auc_score(y,miss_oof))
        else:problems.append("NN-MISS stopped after Fold1: below NN-NOMISS Fold1 by >0.0001")
    best_name=max(variant_data,key=lambda k:variant_data[k][3]);nn_oof,nn_testfolds,best_rows,nn_auc=variant_data[best_name]
    nn_test=nn_testfolds.mean(axis=0,dtype=np.float64)
    pd.DataFrame({"id":train.id,"y_true":y,"oof_prediction":nn_oof}).to_csv(PRED/"oof_exp032_neural.csv",index=False)
    pd.DataFrame({"id":test.id,"prediction":nn_test}).to_csv(PRED/"test_exp032_neural.csv",index=False)

    p16=load_oof(PRED/"oof_exp016_xgboost_depth5_9000.csv",train);p22=load_oof(PRED/"oof_exp022_catboost_thresholds_9000.csv",train)
    s42=load_oof(PRED/"oof_exp027_seed42.csv",train);s2026=load_oof(PRED/"oof_exp027_seed2026.csv",train);s777=load_oof(PRED/"oof_exp027_seed777.csv",train)
    s31415=load_oof(PRED/"oof_exp028_seed31415.csv",train);s1234=load_oof(PRED/"oof_exp028_seed1234.csv",train)
    xgb=.4*s42+.3*s2026+.3*s777;p27=.75*xgb+.25*p22;p28=.75*((s42+s2026+s777+s31415+s1234)/5)+.25*p22;p6=load_oof(PRED/"oof_exp006_lightgbm_features.csv",train)
    controls={"EXP-016":p16,"EXP-022":p22,"EXP-027":p27,"EXP-028":p28,"EXP-006":p6}
    corr=pd.DataFrame([{"model":k,"pearson":pd.Series(nn_oof).corr(pd.Series(p)),"spearman":pd.Series(nn_oof).corr(pd.Series(p),method="spearman")} for k,p in controls.items()])
    resid=y-p27;corr=pd.concat([corr,pd.DataFrame([{"model":"residual_EXP027","pearson":pd.Series(nn_oof).corr(pd.Series(resid)),"spearman":pd.Series(nn_oof).corr(pd.Series(resid),method="spearman")}])],ignore_index=True);corr.to_csv(REPORTS/"exp032_neural_correlations.csv",index=False)

    screen,social=train.daily_screen_time_hours,train.social_media_hours;valid=screen.notna()&social.notna();cp=valid&((screen>8)|(social>4));cn=valid&screen.le(6)&social.le(4);amb=valid&~cp&~cn;rr=[]
    for label,mask in [("clear_positive",cp),("clear_negative",cn),("ambiguous",amb)]:
        for k,p in [(best_name,nn_oof),("EXP-016",p16)]:rr.append({"analysis":"region","segment":label,"model":k,"rows":mask.sum(),"auc":roc_auc_score(y[mask],p[mask])})
    for label,lo,hi in [("0.30-0.40",.3,.4),("0.40-0.50",.4,.5),("0.50-0.60",.5,.6),("0.60-0.70",.6,.7),("0.70-0.80",.7,.8)]:
        mask=(p16>=lo)&(p16<hi)
        for k,p in [(best_name,nn_oof),("EXP-016",p16)]:rr.append({"analysis":"score_band","segment":label,"model":k,"rows":mask.sum(),"auc":roc_auc_score(y[mask],p[mask])})
    regional=pd.DataFrame(rr);regional.to_csv(REPORTS/"exp032_neural_regional.csv",index=False)

    pear27=float(corr.loc[corr.model.eq("EXP-027"),"pearson"].iloc[0]);allow=nn_auc>=.945 or (nn_auc>=.935 and pear27<.93);blendrows=[];nested={}
    if allow:
        for k,b in [("EXP-027",p27),("EXP-028",p28)]:
            result=nested_blend(y,b,nn_oof,splits,k);nested[k]=result
            blendrows.extend(result[3]);blendrows.extend(result[1]);blendrows.append(result[2])
        if max(nested[k][2]["auc"]-(EXP027 if k=="EXP-027" else EXP028) for k in nested)>=.00002:
            result=nested_blend(y,p27,nn_oof,splits,"TRIPLE",TRIPLE_WEIGHTS,True,xgb,p22);nested["TRIPLE"]=result;blendrows.extend(result[3]);blendrows.extend(result[1]);blendrows.append(result[2])
    pd.DataFrame(blendrows).to_csv(REPORTS/"exp032_neural_blends.csv",index=False)

    submission=False;subpath="none";best_ensemble=None
    for k,v in nested.items():
        ref=EXP027 if k in {"EXP-027","TRIPLE"} else EXP028;fs=folds_auc(y,v[0],splits);base= p27 if k in {"EXP-027","TRIPLE"} else p28;d=np.asarray(fs)-np.asarray(folds_auc(y,base,splits))
        if v[2]["auc"]-EXP027>=.00003 and np.sum(d>0)>=4 and not np.any(d<-.00003):
            if best_ensemble is None or v[2]["auc"]>best_ensemble[0]:best_ensemble=(v[2]["auc"],k,v,d)
    if best_ensemble:
        _,k,v,d=best_ensemble;base_test=pd.read_csv(SUB/"submission_exp027_seed_ensemble.csv").addicted_label.to_numpy(np.float64)
        testparts=[]
        for i,c in enumerate(v[1]):
            nt=nn_testfolds[i];nw=c["neural_weight"]
            if c["method"]=="probability":testparts.append((1-nw)*base_test+nw*nt)
            else:testparts.append((1-nw)*rank(base_test)+nw*rank(nt))
        sample=pd.read_csv(DATA/"sample_submission.csv");sub=pd.DataFrame({"id":sample.id,"addicted_label":np.mean(testparts,axis=0)})
        if len(sub)==296302 and sub.id.equals(sample.id) and not sub.isna().any().any() and sub.addicted_label.between(0,1).all():
            subpath=str(SUB/"submission_exp032_neural_ensemble.csv");sub.to_csv(subpath,index=False);submission=True

    useful=(nn_auc>=.950 and pear27<=.985) or any(v[2]["auc"]-EXP027>=.00002 for v in nested.values())
    future={"FT-Transformer":"VIABLE PERO COSTOSO (CPU; GPU recomendable)","TabM":"REQUIERE GPU/NUBE RECOMENDABLE",
            "TabNet":"VIABLE PERO COSTOSO (GPU recomendable)","TabPFN":"NO ADECUADO PARA 691k FILAS"}
    pd.DataFrame(fold_rows).to_csv(REPORTS/"exp032_neural_folds.csv",index=False)
    elapsed=perf_counter()-start_all
    variants={k:{"auc":v[3],"folds":folds_auc(y,v[0],splits),"best_epochs":[r["best_epoch"] for r in v[2]],"seconds":sum(r["seconds"] for r in v[2])} for k,v in variant_data.items()}
    lines=["EXP-032 neural tabular",f"hardware={hw}",f"benchmark={bres}; viable={viable}",f"variants={variants}; best={best_name}",
           "correlations:\n"+corr.to_string(index=False),"regional_bands:\n"+regional.to_string(index=False),f"nested={{{', '.join(f'{k}:{v[2]}' for k,v in nested.items())}}}",
           f"useful={useful}; submission={submission}; path={subpath}",f"future={future}",f"total_seconds={elapsed:.2f}",f"problems={problems or 'none'}"]
    (METRICS/"exp032_neural_metrics.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"variants={variants}; best={best_name}; useful={useful}; submission={submission}; seconds={elapsed:.2f}")

if __name__=="__main__":main()
