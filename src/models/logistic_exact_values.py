"""EXP-035: exact-value sparse one-hot Logistic Regression diagnosis."""
from __future__ import annotations
import gc, sys
from time import perf_counter
import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

from src.diagnose_exp029_pairwise_ranking import sample_pairs
from src.features.exact_values import ORIGINAL, build_rep, safe_ratio, stringify
from src.project_paths import (
    DATA_DIR as DATA,
    METRICS_DIR as METRICS,
    OUTPUTS_DIR as OUT,
    PREDICTIONS_DIR as PRED,
    PROJECT_ROOT as ROOT,
    REPORTS_DIR as REPORTS,
    SUBMISSIONS_DIR as SUB,
)

WEIGHTS=[.005,.01,.02,.03,.05,.075,.10,.15,.20]; TRIPLE_WEIGHTS=[.01,.02,.03,.05,.075,.10]; CS=[.1,.3,1.,3.]
EXP027=.965919188052602; EXP028=.9659307051390922

def rss(): return psutil.Process().memory_info().rss/1024**3
def rank(x):
    r=pd.Series(x).rank(method='average').to_numpy(np.float64); return (r-1)/(len(r)-1)
def load_oof(path,train):
    d=pd.read_csv(path)
    if not d.id.equals(train.id) or not d.y_true.equals(train.addicted_label) or d.oof_prediction.isna().any(): raise ValueError(f'OOF invalida {path.name}')
    return d.oof_prediction.to_numpy(np.float64)
def fit_fold(x,xt,y,tr,va,C,fold,variant):
    start=perf_counter(); peak=rss(); enc=OneHotEncoder(handle_unknown='ignore',dtype=np.float32,sparse_output=True)
    a=enc.fit_transform(x.iloc[tr]); b=enc.transform(x.iloc[va]); e=enc.transform(xt); a=sparse.csr_matrix(a,dtype=np.float32); b=sparse.csr_matrix(b,dtype=np.float32); e=sparse.csr_matrix(e,dtype=np.float32); peak=max(peak,rss())
    # SAGA and LBFGS viability tests were both excessively slow on the exact-pair
    # cardinality. Liblinear is the final controlled sparse binary solver fallback.
    model=LogisticRegression(solver='liblinear',penalty='l2',C=C,max_iter=1000,random_state=42)
    model.fit(a,y[tr]); pv=model.predict_proba(b)[:,1].astype(np.float64); pt=model.predict_proba(e)[:,1].astype(np.float64)
    row={'variant':variant,'C':C,'fold':fold,'auc':roc_auc_score(y[va],pv),'n_columns':a.shape[1],'train_nnz':a.nnz,'valid_nnz':b.nnz,'test_nnz':e.nnz,'sparse_memory_mb':(a.data.nbytes+a.indices.nbytes+a.indptr.nbytes)/1024**2,'peak_rss_gb':peak,'iterations':int(model.n_iter_[0]),'seconds':perf_counter()-start}
    del model,enc,a,b,e; gc.collect(); return pv,pt,row
def folds_auc(y,p,splits): return [float(roc_auc_score(y[v],p[v])) for _,v in splits]

def full_variant(train,test,y,splits,variant,C=1.,reuse=None):
    x=build_rep(train,variant); xt=build_rep(test,variant); oof=np.zeros(len(y)); tests=[]; rows=[]
    for f,(tr,va) in enumerate(splits,1):
        if reuse is not None and f==1: pv,pt,row=reuse
        else: pv,pt,row=fit_fold(x,xt,y,tr,va,C,f,variant)
        oof[va]=pv; tests.append(pt); rows.append(row); pd.DataFrame(rows).to_csv(REPORTS/'exp035_variants_progress.csv',index=False)
    del x,xt; gc.collect(); return oof,np.asarray(tests),rows

def nested_choice(y,preds,splits,label):
    # preds maps C -> full fold-safe OOF; choose C only on the other four folds.
    out=np.zeros(len(y)); rows=[]
    for f,(tr,va) in enumerate(splits,1):
        scores=[(roc_auc_score(y[tr],p[tr]),-abs(np.log10(c)),c,p) for c,p in preds.items()]; b=max(scores,key=lambda z:(z[0],z[1])); out[va]=b[3][va]; rows.append({'candidate':label,'fold':f,'choice_type':'C','C':b[2],'selection_auc':b[0]})
    return out,rows
def nested_blend(y,base,other,splits,label,weights=WEIGHTS,xgb=None,cat=None):
    out=np.zeros(len(y)); rows=[]; rb=rank(base); ro=rank(other); rx=rank(xgb) if xgb is not None else None; rc=rank(cat) if cat is not None else None
    for f,(tr,va) in enumerate(splits,1):
        opts=[]
        for method in ['probability','rank']:
            for w in weights:
                if xgb is None: p=(1-w)*base+w*other if method=='probability' else (1-w)*rb+w*ro
                else: p=(1-w)*(.75*xgb+.25*cat)+w*other if method=='probability' else (1-w)*(.75*rx+.25*rc)+w*ro
                opts.append((roc_auc_score(y[tr],p[tr]),-w,method,w,p))
        b=max(opts,key=lambda z:(z[0],z[1])); out[va]=b[4][va]; rows.append({'candidate':label,'fold':f,'choice_type':'blend','method':b[2],'weight':b[3],'selection_auc':b[0]})
    fs=folds_auc(y,out,splits); summary={'candidate':label,'fold':0,'choice_type':'result','auc':roc_auc_score(y,out),'std':np.std(fs),'folds':str(fs),'improved_folds':int(sum(np.asarray(fs)>np.asarray(folds_auc(y,base,splits))))}
    return out,rows,summary

def finalize_existing():
    """Deploy the unanimously selected EXP-027 rank blend without retraining."""
    train=pd.read_csv(DATA/'train.csv'); sample=pd.read_csv(DATA/'sample_submission.csv'); y=train.addicted_label.to_numpy()
    exact=load_oof(PRED/'oof_exp035_exact_logistic.csv',train)
    p22=load_oof(PRED/'oof_exp022_catboost_thresholds_9000.csv',train)
    seeds=[load_oof(PRED/f'oof_exp027_seed{x}.csv',train) for x in [42,2026,777]]
    p27=.75*(.4*seeds[0]+.3*seeds[1]+.3*seeds[2])+.25*p22
    final_oof=.8*rank(p27)+.2*rank(exact); auc=float(roc_auc_score(y,final_oof))
    splits=list(StratifiedKFold(5,shuffle=True,random_state=42).split(train,y)); fs=folds_auc(y,final_oof,splits); bfs=folds_auc(y,p27,splits)
    choices=pd.read_csv(REPORTS/'exp035_blends.csv'); chosen=choices[(choices.candidate=='EXP-027')&(choices.fold>0)]
    if not (len(chosen)==5 and chosen.method.eq('rank').all() and np.allclose(chosen.weight,.2)): raise ValueError('Las elecciones nested no son unanimemente rank/0.20')
    base_test=pd.read_csv(SUB/'submission_exp027_seed_ensemble.csv'); et=pd.read_csv(PRED/'test_exp035_exact_logistic.csv')
    if not base_test.id.equals(sample.id) or not et.id.equals(sample.id): raise ValueError('IDs test desalineados')
    pred=.8*rank(base_test.addicted_label.to_numpy(np.float64))+.2*rank(et.prediction.to_numpy(np.float64))
    sub=pd.DataFrame({'id':sample.id,'addicted_label':pred}); path=SUB/'submission_exp035_exact_logistic_ensemble.csv'; sub.to_csv(path,index=False)
    if len(sub)!=296302 or sub.isna().any().any() or not sub.id.equals(sample.id) or list(sub)!=['id','addicted_label'] or not sub.addicted_label.between(0,1).all(): raise ValueError('Submission invalida')
    logpath=METRICS/'experiment_log.csv'; log=pd.read_csv(logpath)
    if not log.experiment_id.astype(str).eq('EXP-035').any():
        row={'experiment_id':'EXP-035','datetime':pd.Timestamp.now().isoformat(timespec='seconds'),'model':'ExactValueSparseLogistic_Ensemble','features':'original_exact_categories_plus_selected_interactions_ratios','cv_strategy':'Nested_OOF_ensemble_optimization','cv_roc_auc':auc,'kaggle_score':'','notes':'rank blend EXP-027 0.80 + exact-value sparse Logistic variant C 0.20'}
        log=pd.concat([log,pd.DataFrame([row])],ignore_index=True); log.to_csv(logpath,index=False)
    with (METRICS/'exp035_exact_logistic_metrics.txt').open('a',encoding='utf8') as h:
        h.write(f'finalized_submission=true; selected=rank EXP027 0.80 + EXP035-C 0.20; oof={auc}; folds={fs}; deltas={[a-b for a,b in zip(fs,bfs)]}; path={path}\n')
    print(f'finalized auc={auc:.12f} folds={fs} path={path}')

def main():
    start=perf_counter(); [d.mkdir(parents=True,exist_ok=True) for d in [PRED,REPORTS,METRICS,SUB]]
    train=pd.read_csv(DATA/'train.csv'); test=pd.read_csv(DATA/'test.csv'); y=train.addicted_label.to_numpy(); splits=list(StratifiedKFold(5,shuffle=True,random_state=42).split(train,y)); allrows=[]; problems=[]
    # Variant A Fold-1 viability.
    xa=build_rep(train,'A'); xat=build_rep(test,'A'); tr,va=splits[0]; pv,pt,r1=fit_fold(xa,xat,y,tr,va,1.,1,'A'); allrows.append({**r1,'stage':'viability'}); del xa,xat; gc.collect()
    if r1['auc']<.950:
        pd.DataFrame(allrows).to_csv(REPORTS/'exp035_variants.csv',index=False)
        for f,cols in [('exp035_correlations.csv',['model','pearson','spearman']),('exp035_regional.csv',['analysis','segment','model','rows','auc']),('exp035_pairwise.csv',['model','corrected_misordered_rate','broken_correct_rate','net_pair_gain']),('exp035_blends.csv',['candidate','fold','method','weight','auc'])]: pd.DataFrame(columns=cols).to_csv(REPORTS/f,index=False)
        (METRICS/'exp035_exact_logistic_metrics.txt').write_text(f'variant_A_fold1={r1}\ncontinued=false\nsubmission=false\ntotal_seconds={perf_counter()-start:.2f}\nproblems=Fold1 AUC < 0.950\n',encoding='utf8'); print(f'stopped fold1 auc={r1["auc"]:.8f}',flush=True); return
    aoof,atests,arows=full_variant(train,test,y,splits,'A',1.,(pv,pt,r1)); allrows=arows; variants={'A':(aoof,atests,arows,roc_auc_score(y,aoof),1.)}
    # Small C screen, with full OOF per C and nested selection.
    if variants['A'][3]>=.955:
        cpreds={1.:aoof}; ctests={1.:atests}
        for c in [.1,.3,3.]:
            o,t,rows=full_variant(train,test,y,splits,'A',c); cpreds[c]=o; ctests[c]=t; allrows+=rows
        chosen,crows=nested_choice(y,cpreds,splits,'A-C-NESTED'); chosen_auc=roc_auc_score(y,chosen); variants['A-C-NESTED']=(chosen,None,crows,chosen_auc,'fold-specific')
    if variants['A'][3]>=.957:
        boof,btests,brows=full_variant(train,test,y,splits,'B',1.); allrows+=brows; variants['B']=(boof,btests,brows,roc_auc_score(y,boof),1.)
        if max(variants['A'][3],variants['B'][3])>=.957:
            coof,ctests,crows=full_variant(train,test,y,splits,'C',1.); allrows+=crows; variants['C']=(coof,ctests,crows,roc_auc_score(y,coof),1.)
    pd.DataFrame(allrows).to_csv(REPORTS/'exp035_variants.csv',index=False)
    best=max([k for k in variants if variants[k][1] is not None],key=lambda k:variants[k][3]); exact,etest,erows,eauc,selected_c=variants[best]
    pd.DataFrame({'id':train.id,'y_true':y,'oof_prediction':exact}).to_csv(PRED/'oof_exp035_exact_logistic.csv',index=False); pd.DataFrame({'id':test.id,'prediction':etest.mean(axis=0,dtype=np.float64)}).to_csv(PRED/'test_exp035_exact_logistic.csv',index=False)
    p16=load_oof(PRED/'oof_exp016_xgboost_depth5_9000.csv',train); p22=load_oof(PRED/'oof_exp022_catboost_thresholds_9000.csv',train); p6=load_oof(PRED/'oof_exp006_lightgbm_features.csv',train); fm=load_oof(PRED/'oof_exp034_fm.csv',train)
    seeds=[load_oof(PRED/f'oof_exp027_seed{x}.csv',train) for x in [42,2026,777]]; xgb=.4*seeds[0]+.3*seeds[1]+.3*seeds[2]; p27=.75*xgb+.25*p22
    seeds5=seeds+[load_oof(PRED/f'oof_exp028_seed{x}.csv',train) for x in [31415,1234]]; p28=.75*np.mean(seeds5,axis=0)+.25*p22
    controls={'EXP-016':p16,'EXP-022':p22,'EXP-027':p27,'EXP-028':p28,'EXP-006':p6,'EXP-033-TE':pd.read_csv(PRED/'oof_exp033_best_te_nested.csv').prediction.to_numpy(),'residual_EXP027':y-p27}
    corr=pd.DataFrame([{'model':n,'pearson':pd.Series(exact).corr(pd.Series(p)),'spearman':pd.Series(exact).corr(pd.Series(p),method='spearman')} for n,p in controls.items()]); corr.to_csv(REPORTS/'exp035_correlations.csv',index=False)
    screen=train.daily_screen_time_hours; social=train.social_media_hours; known=screen.notna()&social.notna(); cp=known&((screen>8)|(social>4)); cn=known&(screen<=6)&(social<=4); amb=known&~cp&~cn; rr=[]
    for lab,m in [('clear_positive',cp),('clear_negative',cn),('ambiguous',amb)]:
        for n,p in [('EXP-035',exact),('EXP-027',p27)]: rr.append({'analysis':'region','segment':lab,'model':n,'rows':m.sum(),'auc':roc_auc_score(y[m],p[m])})
    for lab,lo,hi in [('0.30-0.40',.3,.4),('0.40-0.50',.4,.5),('0.50-0.60',.5,.6),('0.60-0.70',.6,.7),('0.70-0.80',.7,.8)]:
        m=(p16>=lo)&(p16<hi)
        for n,p in [('EXP-035',exact),('EXP-027',p27)]: rr.append({'analysis':'score_band','segment':lab,'model':n,'rows':m.sum(),'auc':roc_auc_score(y[m],p[m])})
    regional=pd.DataFrame(rr); regional.to_csv(REPORTS/'exp035_regional.csv',index=False)
    fi=np.empty(len(y),int)
    for f,(_,v) in enumerate(splits): fi[v]=f
    bp,bn,gp,gn,_=sample_pairs(train,p16,np.random.default_rng(42),fi); prows=[]
    for n,p in [('EXP-035',exact),('EXP-022',p22),('EXP-027',p27),('EXP-028',p28),('FM',fm)]:
        corrected=np.mean(p[bp]>p[bn]); broken=np.mean(p[gp]<=p[gn]); prows.append({'model':n,'corrected_misordered_rate':corrected,'broken_correct_rate':broken,'net_pair_gain':corrected-broken})
    pair=pd.DataFrame(prows); pair.to_csv(REPORTS/'exp035_pairwise.csv',index=False)
    blendrows=[]; nested={}
    if eauc>=.955:
        for n,b in [('EXP-027',p27),('EXP-028',p28)]:
            pred,ch,s=nested_blend(y,b,exact,splits,n); nested[n]=(pred,s); blendrows+=ch+[s]
        if nested['EXP-027'][1]['auc']>EXP027:
            pred,ch,s=nested_blend(y,p27,exact,splits,'TRIPLE',TRIPLE_WEIGHTS,xgb,p22); nested['TRIPLE']=(pred,s); blendrows+=ch+[s]
    cols=['candidate','fold','choice_type','method','weight','C','selection_auc','auc','std','folds','improved_folds']; blends=pd.DataFrame(blendrows).reindex(columns=cols); blends.to_csv(REPORTS/'exp035_blends.csv',index=False)
    generated=False; subpath=''; bestens=max([(v[1]['auc'],n,v) for n,v in nested.items()],default=(eauc,'EXP-035',None)); delta=bestens[0]-EXP027
    # No approximation of fold-specific test choices: only create when deployment can be reconstructed exactly.
    if bestens[2] is not None and delta>=.00003 and bestens[0]>EXP028:
        problems.append('score threshold reached but exact fold-specific test deployment was not persisted; no submission generated')
    lines=['EXP-035 exact-value sparse Logistic',f'variant_A_fold1={r1}',f'variants={ {k:v[3] for k,v in variants.items()} }',f'best_variant={best}; best_auc={eauc}; selected_C={selected_c}',f'folds=\n{pd.DataFrame(allrows).to_string(index=False)}',f'correlations=\n{corr.to_string(index=False)}',f'regional=\n{regional.to_string(index=False)}',f'pairwise=\n{pair.to_string(index=False)}',f'blends=\n{blends.to_string(index=False)}',f'submission_generated={generated}; path={subpath}',f'total_seconds={perf_counter()-start:.2f}',f'problems={problems or "none"}']
    (METRICS/'exp035_exact_logistic_metrics.txt').write_text('\n'.join(lines)+'\n',encoding='utf8'); print(f'best={best} auc={eauc:.10f} ensemble={bestens[0]:.10f} seconds={perf_counter()-start:.1f}',flush=True)
if __name__=='__main__':
    if '--finalize-existing' in sys.argv: finalize_existing()
    else: main()
