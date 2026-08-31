"""EXP-041: GPU-ready, fold-safe RealMLP experiment for Kaggle.

This runner owns outer CV, feature engineering, missing-value treatment,
screening gates, diagnostics and nested blends. RealMLP only owns its internal
preprocessing/training and best-epoch selection inside each outer fold.
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import random
import shutil
import sys
import time
import warnings
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import sklearn
import torch
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ID, TARGET = "id", "addicted_label"
ORIGINAL = ["age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
            "work_study_hours", "sleep_hours", "notifications_per_day", "app_opens_per_day",
            "weekend_screen_time", "gender", "stress_level", "academic_work_impact"]
CATEGORICALS = ["gender", "stress_level", "academic_work_impact"]
BASE_FREQ = {"screen_freq": "daily_screen_time_hours", "weekend_freq": "weekend_screen_time"}
EXTRA_FREQ = {"social_freq": "social_media_hours", "gaming_freq": "gaming_hours",
              "work_freq": "work_study_hours", "notifications_freq": "notifications_per_day",
              "appopens_freq": "app_opens_per_day"}
RELATIONS = {
    "social_over_screen": ("social_media_hours", "daily_screen_time_hours", "ratio"),
    "gaming_over_screen": ("gaming_hours", "daily_screen_time_hours", "ratio"),
    "work_over_screen": ("work_study_hours", "daily_screen_time_hours", "ratio"),
    "work_over_social": ("work_study_hours", "social_media_hours", "ratio"),
    "gaming_over_social": ("gaming_hours", "social_media_hours", "ratio"),
    "screen_minus_social": ("daily_screen_time_hours", "social_media_hours", "difference"),
}
MISSING_RATES = [0.00, 0.02, 0.05, 0.10]
SEEDS = [42, 2026]
REALMLP_PARAMS = dict(n_cv=1, n_refit=0, n_epochs=256, batch_size=1024,
                      val_metric_name="1-auc_ovr", use_ls=False, verbosity=1)
REF39 = 0.966683766179125
REF_ENSEMBLE = 0.9674680607837304


@dataclass(frozen=True)
class Config:
    feature_set: str
    extra_freq: tuple[str, ...] = ()
    missing_rate: float = 0.0
    seed: int = 42

    @property
    def label(self):
        extra = "+".join(self.extra_freq) or "none"
        return f"{self.feature_set}|extra={extra}|mask={self.missing_rate:.2f}|seed={self.seed}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True, help="Directory containing train.csv/test.csv")
    p.add_argument("--output-dir", type=Path, default=Path("/kaggle/working"))
    p.add_argument("--reference-dir", type=Path, default=None,
                   help="Optional directory with prior OOF/test predictions for diagnostics and blends")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--max-epochs", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--skip-seed2-full", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Validate data/features/folds without importing RealMLP")
    return p.parse_args()


def hardware():
    cuda = torch.cuda.is_available()
    return {"gpu": torch.cuda.get_device_name(0) if cuda else "none",
            "cuda_available": cuda, "torch_cuda": torch.version.cuda,
            "vram_total_gb": torch.cuda.get_device_properties(0).total_memory/1024**3 if cuda else 0.0,
            "ram_total_gb": psutil.virtual_memory().total/1024**3,
            "ram_available_gb": psutil.virtual_memory().available/1024**3,
            "python": platform.python_version(), "torch": torch.__version__,
            "sklearn": sklearn.__version__, "pytabkit_available": bool(importlib.util.find_spec("pytabkit")),
            "pytabkit": (importlib.metadata.version("pytabkit") if importlib.util.find_spec("pytabkit") else "not-installed")}


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def safe_ratio(a, b):
    av, bv = a.to_numpy(np.float64), b.to_numpy(np.float64)
    out = np.full(len(a), np.nan, np.float64)
    ok = np.isfinite(av) & np.isfinite(bv) & (np.abs(bv) > 1e-12)
    out[ok] = av[ok] / bv[ok]
    return pd.Series(out, index=a.index)


def exact_key(s):
    # Float stringification matches the exact-value convention used in EXP-039.
    if pd.api.types.is_numeric_dtype(s):
        x = s.to_numpy(np.float64)
        return pd.Series(x,index=s.index).map(lambda z:"__MISSING__" if pd.isna(z) else np.format_float_positional(z,trim="-"))
    return s.fillna("__MISSING__").astype(str)


def add_relations(df):
    out = df.copy()
    for name, (a, b, kind) in RELATIONS.items():
        out[name] = safe_ratio(df[a], df[b]) if kind == "ratio" else df[a] - df[b]
    return out


def mask_training(df, rate, seed):
    if rate <= 0: return df.copy()
    rng = np.random.default_rng(seed); out = df.copy()
    observed = out.notna().to_numpy(); draw = rng.random(out.shape) < rate
    out.mask(observed & draw, inplace=True)
    return out


def prepare_fold(train_raw, test_raw, tr, va, cfg):
    """All learned mappings/statistics are fit only on outer_train."""
    base_tr = train_raw.iloc[tr][ORIGINAL].copy()
    base_va = train_raw.iloc[va][ORIGINAL].copy()
    base_te = test_raw[ORIGINAL].copy()
    feature_freq = {} if cfg.feature_set == "A" else {**BASE_FREQ, **{n: EXTRA_FREQ[n] for n in cfg.extra_freq}}

    def engineered(base, source_indices=None):
        x = base.copy()
        if cfg.feature_set == "B": x = add_relations(x)
        return x
    xtr, xva, xte = engineered(base_tr), engineered(base_va), engineered(base_te)
    for name, col in feature_freq.items():
        counts = exact_key(base_tr[col]).value_counts(dropna=False) / len(base_tr)
        xtr[name] = exact_key(base_tr[col]).map(counts).fillna(0).astype(np.float32)
        xva[name] = exact_key(base_va[col]).map(counts).fillna(0).astype(np.float32)
        xte[name] = exact_key(base_te[col]).map(counts).fillna(0).astype(np.float32)

    # Augmentation is train-only and happens after leakage-safe feature construction.
    xtr = mask_training(xtr, cfg.missing_rate, cfg.seed * 1009 + int(tr[0]) % 1000003)
    numeric = [c for c in xtr.columns if c not in CATEGORICALS]
    med = xtr[numeric].median()  # post-mask, outer_train only
    med = med.fillna(0.0)
    for x in [xtr, xva, xte]:
        x[numeric] = x[numeric].replace([np.inf, -np.inf], np.nan).fillna(med).astype(np.float32)
    # Explicit category dtype + tokens; mapping itself remains RealMLP's internal unsupervised preprocessing.
    for c in CATEGORICALS:
        observed = sorted(xtr[c].dropna().astype(str).unique())
        cats = ["__MISSING__", "__UNSEEN__"] + [v for v in observed if v not in {"__MISSING__", "__UNSEEN__"}]
        for x in [xtr, xva, xte]:
            s = x[c].where(x[c].notna(), "__MISSING__").astype(str)
            x[c] = pd.Categorical(s.where(s.isin(cats), "__UNSEEN__"), categories=cats)
    return xtr.reset_index(drop=True), xva.reset_index(drop=True), xte.reset_index(drop=True)


def best_epoch(model):
    """Version-tolerant metadata probe; unavailable is reported, never guessed."""
    names = ["best_epoch_", "best_epoch", "best_epoch_idx_", "n_epochs_", "n_epochs"]
    queue, seen = [model], set()
    for _ in range(30):
        if not queue: break
        obj = queue.pop(0)
        if id(obj) in seen: continue
        seen.add(id(obj))
        for n in names:
            try:
                v = getattr(obj, n)
                if isinstance(v, (int, np.integer, float, np.floating)): return int(v)
                if isinstance(v, (list, tuple, np.ndarray)) and len(v): return int(np.max(v))
            except Exception: pass
        try:
            for n, v in vars(obj).items():
                if any(k in n.lower() for k in ["alg", "model", "interface", "trainer"]) and hasattr(v, "__dict__"): queue.append(v)
        except Exception: pass
    return np.nan


def peak_vram_gb():
    return torch.cuda.max_memory_allocated()/1024**3 if torch.cuda.is_available() else 0.0


class Experiment:
    def __init__(self, train, test, output, device, max_epochs, batch_size):
        self.train, self.test, self.output = train, test, output
        self.y = train[TARGET].to_numpy(np.int8)
        self.splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(train, self.y))
        self.device, self.max_epochs, self.batch_size = device, max_epochs, batch_size
        self.cache, self.fold_rows, self.screen_rows = {}, [], []

    def train_fold(self, cfg, fold):
        key = (cfg, fold)
        if key in self.cache: return self.cache[key]
        from pytabkit import RealMLP_TD_Classifier
        tr, va = self.splits[fold]; seed_all(cfg.seed + fold * 100003)
        xtr, xva, xte = prepare_fold(self.train, self.test, tr, va, cfg)
        params = {**REALMLP_PARAMS, "device": self.device, "random_state": cfg.seed + fold * 100003,
                  "n_epochs": self.max_epochs, "batch_size": self.batch_size}
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter(); model = RealMLP_TD_Classifier(**params)
        model.fit(xtr, self.y[tr], xva, self.y[va], cat_col_names=CATEGORICALS)
        pv = model.predict_proba(xva)[:, 1].astype(np.float64)
        pt = model.predict_proba(xte)[:, 1].astype(np.float64)
        row = {"stage": "", "config": cfg.label, "feature_set": cfg.feature_set,
               "extra_freq": "+".join(cfg.extra_freq), "missing_rate": cfg.missing_rate,
               "seed": cfg.seed, "fold": fold+1, "auc": roc_auc_score(self.y[va], pv),
               "seconds": time.perf_counter()-start, "best_epoch": best_epoch(model),
               "peak_vram_gb": peak_vram_gb(), "rss_gb": psutil.Process().memory_info().rss/1024**3,
               "n_train_features": xtr.shape[1], "device": self.device,
               "params": json.dumps(params, sort_keys=True)}
        self.cache[key] = (pv, pt, row); del model, xtr, xva, xte; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f"{cfg.label} fold={fold+1} auc={row['auc']:.8f} sec={row['seconds']:.1f} epoch={row['best_epoch']}", flush=True)
        return self.cache[key]

    def evaluate(self, cfg, folds, stage):
        oof = np.full(len(self.train), np.nan); tests=[]; rows=[]
        for f in folds:
            cache_hit = (cfg, f) in self.cache
            pv, pt, row = self.train_fold(cfg, f); _, va = self.splits[f]
            oof[va] = pv; tests.append(pt); rows.append({**row, "stage": stage,
                "cache_hit": cache_hit, "effective_seconds": 0.0 if cache_hit else row["seconds"]})
        idx = np.concatenate([self.splits[f][1] for f in folds])
        fs = [roc_auc_score(self.y[self.splits[f][1]], oof[self.splits[f][1]]) for f in folds]
        s = {"stage": stage, "config": cfg.label, "feature_set": cfg.feature_set,
             "extra_freq": "+".join(cfg.extra_freq), "missing_rate": cfg.missing_rate, "seed": cfg.seed,
             "n_folds": len(folds), "combined_oof_auc": roc_auc_score(self.y[idx], oof[idx]),
             "mean_auc": float(np.mean(fs)), "std_auc": float(np.std(fs)), "fold_aucs": json.dumps(fs),
             "seconds": sum(r["effective_seconds"] for r in rows),
             "model_seconds_including_reused": sum(r["seconds"] for r in rows),
             "mean_seconds": np.mean([r["seconds"] for r in rows]), "cached_folds": int(sum(r["cache_hit"] for r in rows)),
             "peak_vram_gb": max(r["peak_vram_gb"] for r in rows),
             "best_epochs": json.dumps([None if pd.isna(r["best_epoch"]) else int(r["best_epoch"]) for r in rows])}
        self.fold_rows.extend(rows); self.screen_rows.append(s); self.save_progress()
        return oof, np.asarray(tests), s

    def save_progress(self):
        pd.DataFrame(self.fold_rows).drop_duplicates(["config","fold","stage"], keep="last").to_csv(self.output/"exp041_realmlp_folds.csv", index=False)
        pd.DataFrame(self.screen_rows).to_csv(self.output/"exp041_realmlp_screen.csv", index=False)


def load_prediction(path, train, test=False):
    df = pd.read_csv(path); expected = train[ID]
    if not df[ID].equals(expected): raise ValueError(f"ID mismatch: {path}")
    candidates = ["prediction", "oof_prediction", TARGET]
    col = next((c for c in candidates if c in df), None)
    if col is None: raise ValueError(f"Prediction column missing: {path}")
    p = df[col].to_numpy(np.float64)
    if not np.isfinite(p).all(): raise ValueError(f"Non-finite predictions: {path}")
    return p


def find_ref(refdir, names):
    if refdir is None or not refdir.exists(): return None
    for n in names:
        hits = list(refdir.rglob(n))
        if hits: return hits[0]
    return None


def load_refs(refdir, train, test):
    specs = {
        "EXP016": ["oof_exp016_xgboost_depth5_9000.csv"],
        "EXP022": ["oof_exp022_catboost_thresholds_9000.csv"],
        "EXP037": ["oof_exp037_relational_logistic.csv"],
        "EXP039": ["oof_exp039_lgbm_highbin.csv"],
        "EXP027_s42": ["oof_exp027_seed42.csv"], "EXP027_s2026": ["oof_exp027_seed2026.csv"],
        "EXP027_s777": ["oof_exp027_seed777.csv"],
    }
    out={}
    for k,names in specs.items():
        p=find_ref(refdir,names)
        if p is not None:
            try: out[k]=load_prediction(p,train)
            except Exception as e: warnings.warn(str(e))
    if all(k in out for k in ["EXP022","EXP027_s42","EXP027_s2026","EXP027_s777"]):
        out["EXP027"]=.75*(.4*out["EXP027_s42"]+.3*out["EXP027_s2026"]+.3*out["EXP027_s777"])+.25*out["EXP022"]
    return out


def ranks(p): return rankdata(p, method="average") / len(p)
def fold_aucs(y,p,splits): return np.array([roc_auc_score(y[v],p[v]) for _,v in splits])


def nested_blends(y, splits, refs, realmlp):
    rows=[]; initial=[]
    base=.375*ranks(refs["EXP027"])+.225*ranks(refs["EXP037"])+.4*ranks(refs["EXP039"])
    for method in ["probability","rank"]:
        q={k:(v if method=="probability" else ranks(v)) for k,v in {"EXP027":refs["EXP027"],"EXP037":refs["EXP037"],"EXP039":refs["EXP039"],"RealMLP":realmlp}.items()}
        base_m=.375*q["EXP027"]+.225*q["EXP037"]+.4*q["EXP039"]
        for w in [.05,.10,.15,.20,.25,.30,.35,.40]:
            initial.append((method,w,(1-w)*base_m+w*q["RealMLP"]))
    out=np.zeros(len(y)); choices=[]
    for f,(tr,va) in enumerate(splits,1):
        coarse=max(initial,key=lambda z:roc_auc_score(y[tr],z[2][tr]))
        refined=[]
        for method in ["probability","rank"]:
            q={k:(v if method=="probability" else ranks(v)) for k,v in {"EXP027":refs["EXP027"],"EXP037":refs["EXP037"],"EXP039":refs["EXP039"],"RealMLP":realmlp}.items()}
            base_m=.375*q["EXP027"]+.225*q["EXP037"]+.4*q["EXP039"]
            for w in sorted(set([coarse[1],max(.0,coarse[1]-.025),min(.45,coarse[1]+.025)])):
                refined.append((method,w,(1-w)*base_m+w*q["RealMLP"]))
        best=max(refined,key=lambda z:roc_auc_score(y[tr],z[2][tr]));out[va]=best[2][va]
        choices.append({"fold":f,"method":best[0],"realmlp_weight":best[1],
            "coarse_weight":coarse[1],"selection_auc":roc_auc_score(y[tr],best[2][tr]),"refined":True})
    fs=fold_aucs(y,out,splits);basefs=fold_aucs(y,base,splits)
    rows.extend(choices);rows.append({"fold":0,"method":"nested","realmlp_weight":np.nan,
        "oof":roc_auc_score(y,out),"delta_vs_exp039_ensemble":roc_auc_score(y,out)-REF_ENSEMBLE,
        "fold_aucs":json.dumps(fs.tolist()),"fold_deltas":json.dumps((fs-basefs).tolist()),
        "folds_improved":int(np.sum(fs>basefs))})
    return out, rows, choices, fs, basefs


def diagnostics(train, y, realmlp, normal_oof, refs, splits, output):
    corr=[]
    for n in ["EXP027","EXP037","EXP039","EXP022"]:
        if n in refs: corr.append({"comparison":n,"pearson":np.corrcoef(realmlp,refs[n])[0,1],"spearman":pd.Series(realmlp).corr(pd.Series(refs[n]),method="spearman")})
    if "EXP039" in refs: corr.append({"comparison":"residual_EXP039","pearson":np.corrcoef(y-realmlp,y-refs["EXP039"])[0,1],"spearman":pd.Series(y-realmlp).corr(pd.Series(y-refs["EXP039"]),method="spearman")})
    if all(n in refs for n in ["EXP027","EXP037","EXP039"]):
        ens=.375*ranks(refs["EXP027"])+.225*ranks(refs["EXP037"])+.4*ranks(refs["EXP039"])
        corr.append({"comparison":"residual_current_ensemble","pearson":np.corrcoef(y-realmlp,y-ens)[0,1],"spearman":pd.Series(y-realmlp).corr(pd.Series(y-ens),method="spearman")})
    pd.DataFrame(corr,columns=["comparison","pearson","spearman"]).to_csv(output/"exp041_realmlp_correlations.csv",index=False)
    region=[];screen=train.daily_screen_time_hours;social=train.social_media_hours;known=screen.notna()&social.notna()
    regs={"clear_positive":known&((screen>8)|(social>4)),"clear_negative":known&(screen<=6)&(social<=4)};regs["ambiguous"]=known&~regs["clear_positive"]&~regs["clear_negative"]
    for seg,m in regs.items():
        for n,p in [("RealMLP",realmlp),("EXP039",refs.get("EXP039"))]:
            if p is not None: region.append({"analysis":"region","segment":seg,"model":n,"rows":int(m.sum()),"auc":roc_auc_score(y[m],p[m])})
    if "EXP016" in refs:
        for lo in [.3,.4,.5,.6,.7]:
            m=(refs["EXP016"]>=lo)&(refs["EXP016"]<lo+.1)
            for n,p in [("RealMLP",realmlp),("EXP039",refs.get("EXP039"))]:
                if p is not None:region.append({"analysis":"EXP016_band","segment":f"{lo:.2f}-{lo+.1:.2f}","model":n,"rows":int(m.sum()),"auc":roc_auc_score(y[m],p[m])})
    pd.DataFrame(region,columns=["analysis","segment","model","rows","auc"]).to_csv(output/"exp041_realmlp_regions.csv",index=False)
    miss=train[ORIGINAL].isna().sum(axis=1);groups={"0 missing":miss==0,"1 missing":miss==1,"2 missing":miss==2,"3+ missing":miss>=3};mr=[]
    for seg,m in groups.items():
        for n,p in [("RealMLP_normal",normal_oof),("RealMLP_best",realmlp),("EXP039",refs.get("EXP039"))]:
            if p is not None and np.isfinite(p[m]).all() and y[m].min()!=y[m].max():mr.append({"missing_group":seg,"model":n,"rows":int(m.sum()),"auc":roc_auc_score(y[m],p[m])})
    pd.DataFrame(mr,columns=["missing_group","model","rows","auc"]).to_csv(output/"exp041_realmlp_missingness.csv",index=False)


def make_zip(output):
    names=["exp041_realmlp_metrics.txt","exp041_realmlp_folds.csv","exp041_realmlp_screen.csv",
           "exp041_realmlp_missingness.csv","exp041_realmlp_correlations.csv","exp041_realmlp_regions.csv",
           "exp041_realmlp_blends.csv","oof_exp041_realmlp.csv","test_exp041_realmlp.csv",
           "oof_exp041_realmlp_seed42.csv","test_exp041_realmlp_seed42.csv",
           "oof_exp041_realmlp_seed2026.csv","test_exp041_realmlp_seed2026.csv",
           "submission_exp041_realmlp_ensemble.csv"]
    with zipfile.ZipFile(output/"exp041_artifacts.zip","w",zipfile.ZIP_DEFLATED) as z:
        for n in names:
            p=output/n
            if p.exists(): z.write(p,arcname=n)


def main():
    args=parse_args();start_all=time.perf_counter();args.output_dir.mkdir(parents=True,exist_ok=True)
    hw=hardware();print(json.dumps(hw,indent=2),flush=True)
    train=pd.read_csv(args.input_dir/"train.csv");test=pd.read_csv(args.input_dir/"test.csv")
    missing=[c for c in ORIGINAL+[ID,TARGET] if c not in train.columns]
    if missing:raise ValueError(f"Missing train columns: {missing}")
    if [c for c in ORIGINAL+[ID] if c not in test.columns]:raise ValueError("Test schema mismatch")
    if not train[ID].is_unique or not test[ID].is_unique:raise ValueError("IDs are not unique")
    splits=list(StratifiedKFold(5,shuffle=True,random_state=42).split(train,train[TARGET]))
    if args.dry_run:
        for cfg in [Config("A"),Config("B")]:
            a,b,c=prepare_fold(train,test,*splits[0],cfg);print(cfg.label,a.shape,b.shape,c.shape,a.dtypes.astype(str).to_dict())
        return
    if not hw["pytabkit_available"]:raise ImportError("pytabkit is not installed. Run the notebook installation cell, then restart/run.")
    if args.device=="cuda" and not torch.cuda.is_available():raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device=("cuda" if torch.cuda.is_available() else "cpu") if args.device=="auto" else args.device
    exp=Experiment(train,test,args.output_dir,device,args.max_epochs,args.batch_size)

    # Fold-1 viability.
    a1=exp.evaluate(Config("A"),[0],"viability")
    b1=exp.evaluate(Config("B"),[0],"viability")
    best1=max(a1[2]["combined_oof_auc"],b1[2]["combined_oof_auc"])
    if best1<.96:
        lines=["EXP-041 STOPPED AFTER FOLD1",f"hardware={hw}",f"A={a1[2]}",f"B={b1[2]}","reason=best Fold1 AUC < 0.96; diagnose before more GPU"]
        (args.output_dir/"exp041_realmlp_metrics.txt").write_text("\n".join(lines),encoding="utf-8")
        pd.DataFrame(columns=["missing_group","model","rows","auc"]).to_csv(args.output_dir/"exp041_realmlp_missingness.csv",index=False);make_zip(args.output_dir);return

    # Complete 3-fold A/B (Fold1 reused).
    a3=exp.evaluate(Config("A"),[0,1,2],"feature_screen")
    b3=exp.evaluate(Config("B"),[0,1,2],"feature_screen")
    best_features=Config("B") if b3[2]["combined_oof_auc"]>a3[2]["combined_oof_auc"] else Config("A")

    # Optional extra frequencies only when B shows a real positive signal.
    strong_b=(b3[2]["combined_oof_auc"]>=.965 and b3[2]["combined_oof_auc"]-a3[2]["combined_oof_auc"]>=.00005)
    if strong_b:
        extra_results=[]
        for name in EXTRA_FREQ:
            r=exp.evaluate(Config("B",(name,)),[0],"extra_freq_fold1");extra_results.append(r)
        eb=max(extra_results,key=lambda z:z[2]["combined_oof_auc"])
        if eb[2]["combined_oof_auc"]>=b1[2]["combined_oof_auc"]+.00005:
            verified=exp.evaluate(Config("B",tuple(eb[2]["extra_freq"].split("+"))),[0,1,2],"extra_freq_3fold")
            if verified[2]["combined_oof_auc"]>b3[2]["combined_oof_auc"]:best_features=Config("B",tuple(verified[2]["extra_freq"].split("+")))

    # Missing augmentation screen; rate zero reuses prior cache.
    aug=[]
    for rate in MISSING_RATES:aug.append(exp.evaluate(Config(best_features.feature_set,best_features.extra_freq,rate,42),[0,1,2],"missing_screen"))
    best_aug=max(aug,key=lambda z:z[2]["combined_oof_auc"]);best_cfg=Config(best_features.feature_set,best_features.extra_freq,best_aug[2]["missing_rate"],42)
    normal3=next(z for z in aug if z[2]["missing_rate"]==0.0)

    # Two-seed screen. Seed 42 is reused; 2026 is new.
    seed42=exp.evaluate(best_cfg,[0,1,2],"seed_screen")
    seed26_cfg=Config(best_cfg.feature_set,best_cfg.extra_freq,best_cfg.missing_rate,2026)
    seed26=exp.evaluate(seed26_cfg,[0,1,2],"seed_screen")
    seed_idx=np.concatenate([exp.splits[f][1] for f in [0,1,2]])
    seed_diversity={"pearson":float(np.corrcoef(seed42[0][seed_idx],seed26[0][seed_idx])[0,1]),
                    "spearman":float(pd.Series(seed42[0][seed_idx]).corr(pd.Series(seed26[0][seed_idx]),method="spearman")),
                    "auc_gap":abs(seed42[2]["combined_oof_auc"]-seed26[2]["combined_oof_auc"])}
    winning=seed26_cfg if seed26[2]["combined_oof_auc"]>seed42[2]["combined_oof_auc"] else best_cfg

    # Full five-fold winner; cached folds 1-3 are reused, only 4-5 are trained.
    full=exp.evaluate(winning,[0,1,2,3,4],"full_cv")
    full_by_seed={winning.seed:(winning,full)}
    oof=full[0];test_pred=full[1].mean(axis=0)
    pd.DataFrame({ID:train[ID],"y_true":exp.y,"oof_prediction":oof}).to_csv(args.output_dir/"oof_exp041_realmlp.csv",index=False)
    pd.DataFrame({ID:test[ID],"prediction":test_pred}).to_csv(args.output_dir/"test_exp041_realmlp.csv",index=False)

    # Optional second full seed only if screen gap <= 0.00015 and time was allowed.
    second_full=None
    screen_gap=abs(seed42[2]["combined_oof_auc"]-seed26[2]["combined_oof_auc"])
    loser=seed26_cfg if winning==best_cfg else best_cfg
    if screen_gap<=.00015 and not args.skip_seed2_full:
        second_full=exp.evaluate(loser,[0,1,2,3,4],"full_cv_seed2")
        full_by_seed[loser.seed]=(loser,second_full)
        if second_full[2]["combined_oof_auc"]>full[2]["combined_oof_auc"]:
            full,winning,oof,test_pred=second_full,loser,second_full[0],second_full[1].mean(axis=0)
            pd.DataFrame({ID:train[ID],"y_true":exp.y,"oof_prediction":oof}).to_csv(args.output_dir/"oof_exp041_realmlp.csv",index=False)
            pd.DataFrame({ID:test[ID],"prediction":test_pred}).to_csv(args.output_dir/"test_exp041_realmlp.csv",index=False)
    for seed,(cfg,result) in full_by_seed.items():
        pd.DataFrame({ID:train[ID],"y_true":exp.y,"oof_prediction":result[0]}).to_csv(args.output_dir/f"oof_exp041_realmlp_seed{seed}.csv",index=False)
        pd.DataFrame({ID:test[ID],"prediction":result[1].mean(axis=0)}).to_csv(args.output_dir/f"test_exp041_realmlp_seed{seed}.csv",index=False)

    refs=load_refs(args.reference_dir,train,test)
    normal_full=None
    if winning.missing_rate==0: normal_full=oof
    else:
        # Diagnostic requires a complete normal prediction; only train it when augmentation actually won.
        normal_cfg=Config(winning.feature_set,winning.extra_freq,0.0,winning.seed)
        normal_full=exp.evaluate(normal_cfg,[0,1,2,3,4],"missingness_normal_control")[0]
    diagnostics(train,exp.y,oof,normal_full,refs,exp.splits,args.output_dir)

    blend_rows=[];submission=False;submission_path=""
    if full[2]["combined_oof_auc"]>=.9665 and all(n in refs for n in ["EXP027","EXP037","EXP039"]):
        blend,blend_rows,choices,bfs,basefs=nested_blends(exp.y,exp.splits,refs,oof)
        result=next(r for r in blend_rows if r["fold"]==0)
        if result["delta_vs_exp039_ensemble"]>=.00010 and result["folds_improved"]>=4:
            # Submission requires explicit prior test predictions; never substitute public predictions.
            test_specs={"EXP027":["test_exp027_seed_ensemble.csv"],"EXP037":["test_exp037_relational_logistic.csv"],"EXP039":["test_exp039_lgbm_highbin.csv"]}
            rt={}
            for n,names in test_specs.items():
                p=find_ref(args.reference_dir,names)
                if p is not None:rt[n]=load_prediction(p,test)
            if len(rt)==3:
                parts=[]
                for c in choices:
                    q={n:(p if c["method"]=="probability" else ranks(p)) for n,p in {**rt,"RealMLP":test_pred}.items()}
                    bm=.375*q["EXP027"]+.225*q["EXP037"]+.4*q["EXP039"]
                    parts.append((1-c["realmlp_weight"])*bm+c["realmlp_weight"]*q["RealMLP"])
                sub=pd.DataFrame({ID:test[ID],TARGET:np.mean(parts,axis=0)});submission_path=str(args.output_dir/"submission_exp041_realmlp_ensemble.csv");sub.to_csv(submission_path,index=False);submission=True
    pd.DataFrame(blend_rows,columns=(list(blend_rows[0]) if blend_rows else ["fold","method","realmlp_weight","selection_auc","oof","delta_vs_exp039_ensemble","fold_aucs","fold_deltas","folds_improved"])).to_csv(args.output_dir/"exp041_realmlp_blends.csv",index=False)
    faucs=fold_aucs(exp.y,oof,exp.splits)
    lines=["EXP-041 RealMLP",f"hardware={json.dumps(hw,sort_keys=True)}",f"realmlp_params={json.dumps({**REALMLP_PARAMS,'device':device,'n_epochs':args.max_epochs,'batch_size':args.batch_size},sort_keys=True)}",
           f"feature_A={a3[2]}",f"feature_B={b3[2]}",f"missing_screens={[z[2] for z in aug]}",f"seed42={seed42[2]}",f"seed2026={seed26[2]}",f"seed_diversity={seed_diversity}",
           f"winner={winning.label}",f"oof={roc_auc_score(exp.y,oof)}",f"folds={faucs.tolist()}",f"mean={faucs.mean()};std={faucs.std()}",
           f"delta_vs_EXP039={roc_auc_score(exp.y,oof)-REF39}",f"blend={blend_rows}",f"submission={submission};path={submission_path}",
           f"total_seconds={time.perf_counter()-start_all}","preprocessing=outer-train median numeric imputation; explicit categorical missing/unseen tokens; RealMLP robust numeric preprocessing; no manual OHE; no target encoding"]
    (args.output_dir/"exp041_realmlp_metrics.txt").write_text("\n\n".join(lines),encoding="utf-8")
    make_zip(args.output_dir);print("EXP-041 complete",lines[-4:],flush=True)


if __name__ == "__main__": main()
