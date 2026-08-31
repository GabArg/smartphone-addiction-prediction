"""EXP-034: CPU-safe binary Factorization Machine and diversity diagnosis."""
from __future__ import annotations

import copy, gc, random
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import psutil
import torch
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

from train_xgboost_exp012_threshold_features import NEW_FEATURES, add_threshold_features
from diagnose_exp029_pairwise_ranking import sample_pairs

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'; OUT=ROOT/'outputs'
PRED=OUT/'predictions'; REPORTS=OUT/'reports'; METRICS=OUT/'metrics'; SUB=OUT/'submissions'
CATS=['gender','stress_level','academic_work_impact']; WEIGHTS=[.005,.01,.02,.03,.05,.075,.10,.15,.20]
TRIPLE_WEIGHTS=[.01,.02,.03,.05,.075,.10]; EXP027=.965919188052602; EXP028=.9659307051390922

def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
def rss(): return psutil.Process().memory_info().rss/1024**3
def rank(x):
    r=pd.Series(x).rank(method='average').to_numpy(np.float64); return (r-1)/(len(r)-1)
def load_oof(path,train):
    d=pd.read_csv(path)
    if not d.id.equals(train.id) or not d.y_true.equals(train.addicted_label) or d.oof_prediction.isna().any(): raise ValueError(f'OOF invalida {path.name}')
    return d.oof_prediction.to_numpy(np.float64)

class FM(nn.Module):
    def __init__(self,n_features,k):
        super().__init__(); self.bias=nn.Parameter(torch.zeros(1)); self.linear=nn.Parameter(torch.zeros(n_features)); self.v=nn.Parameter(torch.empty(n_features,k))
        nn.init.normal_(self.v,mean=0,std=.01)
    def forward(self,x):
        xv=x@self.v; interaction=.5*((xv*xv)-(x*x)@(self.v*self.v)).sum(1)
        return self.bias+(x*self.linear).sum(1)+interaction

def make_preprocessor(numeric):
    num=Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())])
    cat=Pipeline([('imputer',SimpleImputer(strategy='constant',fill_value='__MISSING__')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=True,dtype=np.float32))])
    return ColumnTransformer([('num',num,numeric),('cat',cat,CATS)],sparse_threshold=1.0)

def predict(model,x,batch=4096):
    model.eval(); out=[]
    with torch.no_grad():
        for start in range(0,x.shape[0],batch):
            xb=torch.from_numpy(x[start:start+batch].toarray().astype(np.float32,copy=False)); out.append(torch.sigmoid(model(xb)).numpy())
    return np.concatenate(out).astype(np.float64)

def train_fold(raw,raw_test,y,tr,va,k,fold,max_epochs=25,patience=4,batch=4096):
    seed_all(); begin=perf_counter(); peak=rss(); numeric=[c for c in raw if c not in CATS]
    prep=make_preprocessor(numeric); xt=prep.fit_transform(raw.iloc[tr]); xv=prep.transform(raw.iloc[va]); xe=prep.transform(raw_test)
    xt=sparse.csr_matrix(xt,dtype=np.float32); xv=sparse.csr_matrix(xv,dtype=np.float32); xe=sparse.csr_matrix(xe,dtype=np.float32); peak=max(peak,rss())
    names=prep.get_feature_names_out().tolist(); model=FM(xt.shape[1],k); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-5); lossfn=nn.BCEWithLogitsLoss()
    rng=np.random.default_rng(42); best=-1.; best_epoch=0; best_state=None; wait=0; epoch_times=[]
    for epoch in range(1,max_epochs+1):
        et=perf_counter(); model.train(); order=rng.permutation(len(tr))
        for start in range(0,len(order),batch):
            idx=order[start:start+batch]; xb=torch.from_numpy(xt[idx].toarray().astype(np.float32,copy=False)); yb=torch.from_numpy(y[tr][idx].astype(np.float32,copy=False))
            opt.zero_grad(set_to_none=True); loss=lossfn(model(xb),yb); loss.backward(); opt.step()
        pv=predict(model,xv,batch); auc=float(roc_auc_score(y[va],pv)); epoch_times.append(perf_counter()-et); peak=max(peak,rss())
        print(f'FM-k{k} fold={fold} epoch={epoch} auc={auc:.8f} sec={epoch_times[-1]:.1f}',flush=True)
        if auc>best+1e-7: best=auc; best_epoch=epoch; best_state=copy.deepcopy(model.state_dict()); wait=0
        else: wait+=1
        if wait>=patience: break
    model.load_state_dict(best_state); pv=predict(model,xv,batch); pt=predict(model,xe,batch); v=model.v.detach().numpy().copy()
    row={'k':k,'fold':fold,'auc':float(roc_auc_score(y[va],pv)),'best_epoch':best_epoch,'epochs_run':len(epoch_times),'seconds':perf_counter()-begin,'mean_epoch_seconds':float(np.mean(epoch_times)),'peak_rss_gb':peak}
    del model,opt,prep,xt,xv,xe,best_state; gc.collect(); return pv,pt,row,names,v

def fold_auc(y,p,splits): return [float(roc_auc_score(y[v],p[v])) for _,v in splits]
def nested_blend(y,base,fm,splits,label,weights=WEIGHTS,xgb=None,cat=None):
    out=np.zeros(len(y)); rows=[]; rb=rank(base); rf=rank(fm); rx=rank(xgb) if xgb is not None else None; rc=rank(cat) if cat is not None else None
    for f,(tr,va) in enumerate(splits,1):
        opts=[]
        for method in ['probability','rank']:
            for w in weights:
                if xgb is None: p=(1-w)*base+w*fm if method=='probability' else (1-w)*rb+w*rf
                else: p=(1-w)*(.75*xgb+.25*cat)+w*fm if method=='probability' else (1-w)*(.75*rx+.25*rc)+w*rf
                opts.append((roc_auc_score(y[tr],p[tr]),-w,method,w,p))
        b=max(opts,key=lambda z:(z[0],z[1])); out[va]=b[4][va]; rows.append({'candidate':label,'fold':f,'method':b[2],'fm_weight':b[3],'selection_auc':b[0]})
    fs=fold_auc(y,out,splits); summary={'candidate':label,'fold':0,'method':'nested','auc':roc_auc_score(y,out),'std':np.std(fs),'folds':str(fs),'improved_folds':int(sum(np.asarray(fs)>np.asarray(fold_auc(y,base,splits))))}
    return out,rows,summary

def interaction_table(names,vs,numeric):
    # Average signed pair strength over folds, aligned by feature name.
    idx={n:i for i,n in enumerate(names)}; selected=[f'num__{c}' for c in numeric if f'num__{c}' in idx]; rows=[]
    for i,a in enumerate(selected):
        for b in selected[i+1:]:
            vals=[float(np.dot(v[idx[a]],v[idx[b]])) for v in vs]
            rows.append({'feature_a':a.removeprefix('num__'),'feature_b':b.removeprefix('num__'),'mean_strength':np.mean(vals),'std_strength':np.std(vals),'abs_mean_strength':abs(np.mean(vals))})
    return pd.DataFrame(rows).sort_values('abs_mean_strength',ascending=False)

def main():
    start=perf_counter(); seed_all(); torch.set_num_threads(min(12,psutil.cpu_count() or 1)); [d.mkdir(parents=True,exist_ok=True) for d in [PRED,REPORTS,METRICS,SUB]]
    train=pd.read_csv(DATA/'train.csv'); test=pd.read_csv(DATA/'test.csv'); y=train.addicted_label.to_numpy(); originals=[c for c in train if c not in {'id','addicted_label'}]
    raw=add_threshold_features(train[originals]); rawtest=add_threshold_features(test[originals]); numeric=[c for c in raw if c not in CATS]
    if raw.columns.tolist()!=rawtest.columns.tolist() or set(NEW_FEATURES)-set(raw): raise ValueError('Features no identicas a EXP-016')
    splits=list(StratifiedKFold(5,shuffle=True,random_state=42).split(raw,y)); folds=[]; variants={}; problems=[]
    # Fold-1 viability for k=16.
    tr,va=splits[0]; pv1,pt1,r1,names1,v1=train_fold(raw,rawtest,y,tr,va,16,1); folds.append({**r1,'stage':'viability'}); viable=r1['auc']>=.94 or (r1['auc']>=.92 and r1['seconds']<600)
    if not viable or (r1['seconds']>1800 and r1['auc']<.95):
        pd.DataFrame(folds).to_csv(REPORTS/'exp034_folds.csv',index=False)
        for f,cols in [('exp034_correlations.csv',['model','pearson','spearman']),('exp034_regional.csv',['analysis','segment','model','rows','auc']),('exp034_pairwise.csv',['model','corrected_misordered_rate','broken_correct_rate','net_pair_gain']),('exp034_blends.csv',['candidate','fold','method','fm_weight','auc']),('exp034_interactions.csv',['feature_a','feature_b','mean_strength'])]: pd.DataFrame(columns=cols).to_csv(REPORTS/f,index=False)
        (METRICS/'exp034_factorization_machine.txt').write_text(f'implementation=PyTorch sparse-batched FM\nviability={r1}\nfull_cv=false\nproblems=failed viability criterion\ntotal_seconds={perf_counter()-start:.2f}\n',encoding='utf8'); return
    # Reuse viability Fold1, train remaining folds.
    oof=np.zeros(len(y)); oof[va]=pv1; tests=[pt1]; rows=[r1]; vs=[v1]
    for f,(tr,va) in enumerate(splits[1:],2):
        pv,pt,r,nm,v=train_fold(raw,rawtest,y,tr,va,16,f); oof[va]=pv; tests.append(pt); rows.append(r); vs.append(v); folds.append({**r,'stage':'full'}); pd.DataFrame(folds).to_csv(REPORTS/'exp034_folds.csv',index=False)
    variants[16]=(oof,np.mean(tests,axis=0,dtype=np.float64),rows,names1,vs,roc_auc_score(y,oof))
    # Single controlled k=32 variant.
    if variants[16][5]>=.94:
        tr,va=splits[0]; qv,qt,qr,qnames,qvec=train_fold(raw,rawtest,y,tr,va,32,1); folds.append({**qr,'stage':'k32_screen'}); corr16=float(pd.Series(qv).corr(pd.Series(oof[va])))
        if qr['auc']>=rows[0]['auc'] or (qr['auc']>=rows[0]['auc']-.0001 and corr16<=.9995):
            qoof=np.zeros(len(y)); qoof[va]=qv; qtests=[qt]; qrows=[qr]; qvs=[qvec]
            for f,(tr,va) in enumerate(splits[1:],2):
                pv,pt,r,nm,v=train_fold(raw,rawtest,y,tr,va,32,f); qoof[va]=pv; qtests.append(pt); qrows.append(r); qvs.append(v); folds.append({**r,'stage':'full'}); pd.DataFrame(folds).to_csv(REPORTS/'exp034_folds.csv',index=False)
            variants[32]=(qoof,np.mean(qtests,axis=0,dtype=np.float64),qrows,qnames,qvs,roc_auc_score(y,qoof))
        else: problems.append(f'k32 stopped after Fold1: auc={qr["auc"]:.8f}, corr_k16={corr16:.8f}')
    pd.DataFrame(folds).to_csv(REPORTS/'exp034_folds.csv',index=False)
    bk=max(variants,key=lambda k:variants[k][5]); fm,fmt,fmrows,names,vs,fm_auc=variants[bk]
    pd.DataFrame({'id':train.id,'y_true':y,'oof_prediction':fm}).to_csv(PRED/'oof_exp034_fm.csv',index=False); pd.DataFrame({'id':test.id,'prediction':fmt}).to_csv(PRED/'test_exp034_fm.csv',index=False)
    p16=load_oof(PRED/'oof_exp016_xgboost_depth5_9000.csv',train); p22=load_oof(PRED/'oof_exp022_catboost_thresholds_9000.csv',train); p6=load_oof(PRED/'oof_exp006_lightgbm_features.csv',train)
    ss=[load_oof(PRED/f'oof_exp027_seed{x}.csv',train) for x in [42,2026,777]]; xgb=.4*ss[0]+.3*ss[1]+.3*ss[2]; p27=.75*xgb+.25*p22
    ss2=ss+[load_oof(PRED/f'oof_exp028_seed{x}.csv',train) for x in [31415,1234]]; p28=.75*np.mean(ss2,axis=0)+.25*p22
    controls={'EXP-016':p16,'EXP-022':p22,'EXP-027':p27,'EXP-028':p28,'EXP-006':p6,'residual_EXP027':y-p27}
    cor=pd.DataFrame([{'model':n,'pearson':pd.Series(fm).corr(pd.Series(p)),'spearman':pd.Series(fm).corr(pd.Series(p),method='spearman')} for n,p in controls.items()]); cor.to_csv(REPORTS/'exp034_correlations.csv',index=False)
    # Regions and score bands.
    screen=train.daily_screen_time_hours; social=train.social_media_hours; known=screen.notna()&social.notna(); cp=known&((screen>8)|(social>4)); cn=known&(screen<=6)&(social<=4); amb=known&~cp&~cn; rr=[]
    for lab,m in [('clear_positive',cp),('clear_negative',cn),('ambiguous',amb)]:
        for name,p in [('FM',fm),('EXP-027',p27)]: rr.append({'analysis':'region','segment':lab,'model':name,'rows':m.sum(),'auc':roc_auc_score(y[m],p[m])})
    for lab,lo,hi in [('0.30-0.40',.3,.4),('0.40-0.50',.4,.5),('0.50-0.60',.5,.6),('0.60-0.70',.6,.7),('0.70-0.80',.7,.8)]:
        m=(p16>=lo)&(p16<hi)
        for name,p in [('FM',fm),('EXP-027',p27)]: rr.append({'analysis':'score_band','segment':lab,'model':name,'rows':m.sum(),'auc':roc_auc_score(y[m],p[m])})
    regional=pd.DataFrame(rr); regional.to_csv(REPORTS/'exp034_regional.csv',index=False)
    # Pairwise diagnosis using the same deterministic sampler.
    fi=np.empty(len(y),int)
    for f,(_,v) in enumerate(splits): fi[v]=f
    bp,bn,gp,gn,_=sample_pairs(train,p16,np.random.default_rng(42),fi); pairrows=[]
    for name,p in [('FM',fm),('EXP-022',p22),('EXP-027',p27),('EXP-028',p28)]:
        corrected=float(np.mean(p[bp]>p[bn])); broken=float(np.mean(p[gp]<=p[gn])); pairrows.append({'model':name,'misordered_pairs':len(bp),'corrected_misordered_rate':corrected,'broken_correct_rate':broken,'net_pair_gain':corrected-broken})
    pairdf=pd.DataFrame(pairrows); pairdf.to_csv(REPORTS/'exp034_pairwise.csv',index=False)
    interactions=interaction_table(names,vs,numeric); interactions.to_csv(REPORTS/'exp034_interactions.csv',index=False)
    blendrows=[]; nested_results={}
    if fm_auc>=.94 or (fm_auc>=.92 and float(cor.loc[cor.model.eq('EXP-027'),'pearson'].iloc[0])<=.90):
        for label,b in [('EXP-027',p27),('EXP-028',p28)]:
            npred,ch,s=nested_blend(y,b,fm,splits,label); nested_results[label]=(npred,s); blendrows+=ch+[s]
        if nested_results['EXP-027'][1]['auc']>EXP027:
            npred,ch,s=nested_blend(y,p27,fm,splits,'TRIPLE',TRIPLE_WEIGHTS,xgb,p22); nested_results['TRIPLE']=(npred,s); blendrows+=ch+[s]
    blend_columns=['candidate','fold','method','fm_weight','selection_auc','auc','std','folds','improved_folds']
    blends=pd.DataFrame(blendrows).reindex(columns=blend_columns); blends.to_csv(REPORTS/'exp034_blends.csv',index=False)
    bestens=max([(v[1]['auc'],k,v) for k,v in nested_results.items()],default=(fm_auc,'FM',None)); delta=bestens[0]-EXP027
    generated=False; subpath=''
    if bestens[2] is not None:
        s=bestens[2][1]; fs=np.asarray(eval(s['folds'])); basefs=np.asarray(fold_auc(y,p27,splits)); stable=(fs>basefs).sum()>=4 and np.min(fs-basefs)>=-.00003
        if delta>=.00003 and bestens[0]>EXP028 and stable:
            # Test deployment requires fold-specific choices; unavailable choices are deliberately not approximated.
            problems.append('ensemble passed score screen but fold-specific test deployment not generated automatically')
    lines=['EXP-034 Factorization Machine', 'implementation=custom PyTorch binary FM; CSR retained until minibatch',f'viability={r1}',f'variant_aucs={ {k:v[5] for k,v in variants.items()} }',f'best_k={bk}; best_auc={fm_auc}',f'folds={folds}',f'correlations=\n{cor.to_string(index=False)}',f'regional=\n{regional.to_string(index=False)}',f'pairwise=\n{pairdf.to_string(index=False)}',f'interactions=\n{interactions.head(20).to_string(index=False)}',f'blends=\n{blends.to_string(index=False)}',f'submission_generated={generated}; path={subpath}',f'total_seconds={perf_counter()-start:.2f}',f'problems={problems or "none"}']
    (METRICS/'exp034_factorization_machine.txt').write_text('\n'.join(lines)+'\n',encoding='utf8'); print(f'best_k={bk} auc={fm_auc:.10f} ensemble={bestens[0]:.10f} seconds={perf_counter()-start:.1f}',flush=True)
if __name__=='__main__': main()
