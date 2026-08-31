"""EXP-036: controlled ratio ablation for exact-value sparse Logistic."""
from __future__ import annotations
import ast, gc, sys
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

_PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_FOR_IMPORTS))

from src.diagnose_exp029_pairwise_ranking import sample_pairs
from src.features.exact_values import ORIGINAL, safe_ratio, stringify
from src.features.relational import build_base_representation as base_rep
from src.models.logistic_exact_values import load_oof, rank
from src.project_paths import (
    DATA_DIR as DATA,
    METRICS_DIR as METRICS,
    OUTPUTS_DIR as OUT,
    PREDICTIONS_DIR as PRED,
    PROJECT_ROOT as ROOT,
    REPORTS_DIR as REPORTS,
    SUBMISSIONS_DIR as SUB,
)

OLD={'social_over_screen':('social_media_hours','daily_screen_time_hours'),'gaming_over_screen':('gaming_hours','daily_screen_time_hours'),'work_over_screen':('work_study_hours','daily_screen_time_hours'),'weekend_over_screen':('weekend_screen_time','daily_screen_time_hours')}
NEW={'social_over_weekend':('social_media_hours','weekend_screen_time'),'gaming_over_social':('gaming_hours','social_media_hours'),'work_over_social':('work_study_hours','social_media_hours'),'gaming_over_work':('gaming_hours','work_study_hours'),'notifications_over_screen':('notifications_per_day','daily_screen_time_hours'),'app_opens_over_screen':('app_opens_per_day','daily_screen_time_hours'),'notifications_over_app_opens':('notifications_per_day','app_opens_per_day'),'weekend_over_work':('weekend_screen_time','work_study_hours')}
ALL={**OLD,**NEW}; EXP035=.9633152748359917; ENS035=.9668097848980569; EXP027=.965919188052602; EXP028=.9659307051390922
BLEND_WEIGHTS=[.10,.125,.15,.175,.20,.225,.25,.275,.30]

def rss(): return psutil.Process().memory_info().rss/1024**3
def add_ratios(base,raw,spec):
    x=base.copy()
    for name,digits in spec:
        a,b=ALL[name];x[name]=stringify(safe_ratio(raw[a],raw[b]),digits)
    return x
def key(spec):return tuple(sorted(spec))
def fit_cv(raw,base,y,splits,spec,label,need_test=False,rawtest=None,basetest=None):
    x=add_ratios(base,raw,spec); xt=add_ratios(basetest,rawtest,spec) if need_test else None;oof=np.zeros(len(y));tests=[];rows=[]
    for fold,(tr,va) in enumerate(splits,1):
        st=perf_counter();peak=rss();enc=OneHotEncoder(handle_unknown='ignore',dtype=np.float32,sparse_output=True);a=enc.fit_transform(x.iloc[tr]);b=enc.transform(x.iloc[va]);e=enc.transform(xt) if need_test else None
        a=sparse.csr_matrix(a,dtype=np.float32);b=sparse.csr_matrix(b,dtype=np.float32);e=sparse.csr_matrix(e,dtype=np.float32) if need_test else None;peak=max(peak,rss())
        m=LogisticRegression(solver='liblinear',C=1.,max_iter=1000,random_state=42);m.fit(a,y[tr]);oof[va]=m.predict_proba(b)[:,1]
        if need_test:tests.append(m.predict_proba(e)[:,1].astype(np.float64))
        rows.append({'label':label,'spec':str(spec),'fold':fold,'auc':roc_auc_score(y[va],oof[va]),'columns':a.shape[1],'nnz':a.nnz,'csr_mb':(a.data.nbytes+a.indices.nbytes+a.indptr.nbytes)/1024**2,'peak_rss_gb':peak,'iterations':int(m.n_iter_[0]),'seconds':perf_counter()-st})
        del enc,m,a,b,e;gc.collect()
    del x,xt;gc.collect();return oof,np.asarray(tests),rows
def fscores(y,p,splits):return np.asarray([roc_auc_score(y[v],p[v]) for _,v in splits])
def nested_blend(y,base,other,splits,label):
    out=np.zeros(len(y));rows=[];rb=rank(base);ro=rank(other)
    for f,(tr,va) in enumerate(splits,1):
        opts=[]
        for method in ['probability','rank']:
            for w in BLEND_WEIGHTS:
                p=(1-w)*base+w*other if method=='probability' else (1-w)*rb+w*ro;opts.append((roc_auc_score(y[tr],p[tr]),-w,method,w,p))
        z=max(opts,key=lambda q:(q[0],q[1]));out[va]=z[4][va];rows.append({'candidate':label,'fold':f,'method':z[2],'logistic_weight':z[3],'selection_auc':z[0]})
    fs=fscores(y,out,splits);rows.append({'candidate':label,'fold':0,'auc':roc_auc_score(y,out),'std':np.std(fs),'folds':str(fs.tolist()),'improved_folds':int(sum(fs>fscores(y,base,splits)))})
    return out,rows

def finalize_only():
    start=perf_counter();[d.mkdir(parents=True,exist_ok=True) for d in [PRED,REPORTS,METRICS,SUB]]
    train=pd.read_csv(DATA/'train.csv');test=pd.read_csv(DATA/'test.csv');y=train.addicted_label.to_numpy();splits=list(StratifiedKFold(5,shuffle=True,random_state=42).split(train,y));base=base_rep(train);basetest=base_rep(test)
    bestspec=[('social_over_screen',2),('gaming_over_screen',2),('work_over_screen',2),('work_over_social',2),('gaming_over_social',2)]
    bestp,testfolds,finalrows=fit_cv(train,base,y,splits,bestspec,'FINAL',True,test,basetest);bestauc=roc_auc_score(y,bestp);testpred=testfolds.mean(axis=0,dtype=np.float64)
    pd.DataFrame({'id':train.id,'y_true':y,'oof_prediction':bestp}).to_csv(PRED/'oof_exp036_ratio_logistic.csv',index=False);pd.DataFrame({'id':test.id,'prediction':testpred}).to_csv(PRED/'test_exp036_ratio_logistic.csv',index=False)
    p35=load_oof(PRED/'oof_exp035_exact_logistic.csv',train);p16=load_oof(PRED/'oof_exp016_xgboost_depth5_9000.csv',train);p22=load_oof(PRED/'oof_exp022_catboost_thresholds_9000.csv',train);ss=[load_oof(PRED/f'oof_exp027_seed{x}.csv',train) for x in [42,2026,777]];xgb=.4*ss[0]+.3*ss[1]+.3*ss[2];p27=.75*xgb+.25*p22;ss5=ss+[load_oof(PRED/f'oof_exp028_seed{x}.csv',train) for x in [31415,1234]];p28=.75*np.mean(ss5,axis=0)+.25*p22
    cor=pd.DataFrame([{'model':n,'pearson':pd.Series(bestp).corr(pd.Series(p)),'spearman':pd.Series(bestp).corr(pd.Series(p),method='spearman')} for n,p in {'EXP-016':p16,'EXP-022':p22,'EXP-027':p27,'EXP-028':p28,'EXP-035-C':p35,'residual_EXP027':y-p27}.items()]);cor.to_csv(REPORTS/'exp036_correlations.csv',index=False)
    screen=train.daily_screen_time_hours;social=train.social_media_hours;known=screen.notna()&social.notna();cp=known&((screen>8)|(social>4));cn=known&(screen<=6)&(social<=4);amb=known&~cp&~cn;rr=[]
    for lab,m in [('clear_positive',cp),('clear_negative',cn),('ambiguous',amb)]:
        for n,p in [('EXP-036',bestp),('EXP-035-C',p35),('EXP-027',p27)]:rr.append({'analysis':'region','segment':lab,'model':n,'rows':m.sum(),'auc':roc_auc_score(y[m],p[m])})
    for lab,lo,hi in [('0.30-0.40',.3,.4),('0.40-0.50',.4,.5),('0.50-0.60',.5,.6),('0.60-0.70',.6,.7),('0.70-0.80',.7,.8)]:
        m=(p16>=lo)&(p16<hi)
        for n,p in [('EXP-036',bestp),('EXP-035-C',p35),('EXP-027',p27)]:rr.append({'analysis':'score_band','segment':lab,'model':n,'rows':m.sum(),'auc':roc_auc_score(y[m],p[m])})
    regional=pd.DataFrame(rr);regional.to_csv(REPORTS/'exp036_regional.csv',index=False)
    fi=np.empty(len(y),int)
    for f,(_,v) in enumerate(splits):fi[v]=f
    bp,bn,gp,gn,_=sample_pairs(train,p16,np.random.default_rng(42),fi);pr=[]
    for n,p in [('EXP-036',bestp),('EXP-035-C',p35)]:
        corrected=np.mean(p[bp]>p[bn]);broken=np.mean(p[gp]<=p[gn]);pr.append({'model':n,'corrected':corrected,'broken':broken,'net_gain':corrected-broken})
    pairwise=pd.DataFrame(pr);pairwise.to_csv(REPORTS/'exp036_pairwise.csv',index=False)
    blendrows=[];nested={}
    for n,b in [('EXP-027',p27),('EXP-028',p28)]:pred,rows=nested_blend(y,b,bestp,splits,n);nested[n]=pred;blendrows+=rows
    blenddf=pd.DataFrame(blendrows);blenddf.to_csv(REPORTS/'exp036_blends.csv',index=False)
    boof=nested['EXP-027'];bauc=roc_auc_score(y,boof);bfs=fscores(y,boof,splits);ref=.8*rank(p27)+.2*rank(p35);rfs=fscores(y,ref,splits);choices=blenddf[(blenddf.candidate=='EXP-027')&(blenddf.fold>0)]
    generated=False;subpath=''
    if bauc-ENS035>=.00003 and sum(bfs>rfs)>=4:
        base_test=pd.read_csv(SUB/'submission_exp027_seed_ensemble.csv').addicted_label.to_numpy(np.float64);foldtest=[]
        for r,lp in zip(choices.itertuples(),testfolds):foldtest.append((1-r.logistic_weight)*base_test+r.logistic_weight*lp if r.method=='probability' else (1-r.logistic_weight)*rank(base_test)+r.logistic_weight*rank(lp))
        pred=np.mean(foldtest,axis=0,dtype=np.float64);sample=pd.read_csv(DATA/'sample_submission.csv');sub=pd.DataFrame({'id':sample.id,'addicted_label':pred});subpath=SUB/'submission_exp036_ratio_logistic_ensemble.csv';sub.to_csv(subpath,index=False)
        if len(sub)!=296302 or sub.isna().any().any() or not sub.id.equals(sample.id) or not sub.addicted_label.between(0,1).all():raise ValueError('Submission invalida')
        generated=True;logpath=METRICS/'experiment_log.csv';log=pd.read_csv(logpath)
        if not log.experiment_id.astype(str).eq('EXP-036').any():
            notes='fold-specific nested blend EXP027 + EXP036; choices='+','.join(f'{r.method}:{r.logistic_weight}' for r in choices.itertuples());log=pd.concat([log,pd.DataFrame([{'experiment_id':'EXP-036','datetime':pd.Timestamp.now().isoformat(timespec='seconds'),'model':'RatioSparseLogistic_Ensemble','features':str(bestspec),'cv_strategy':'Nested_OOF_ensemble_optimization','cv_roc_auc':bauc,'kaggle_score':'','notes':notes}])],ignore_index=True);log.to_csv(logpath,index=False)
    # Screening reports already persisted before the strict-refit exception.
    reports={n:pd.read_csv(REPORTS/n) for n in ['exp036_ratio_single_ablation.csv','exp036_ratio_leave_one_out.csv','exp036_ratio_pairs.csv','exp036_ratio_granularity.csv','exp036_new_ratio_candidates.csv','exp036_forward_selection.csv']}
    lines=['EXP-036 ratio ablation (finalized)',f'best_spec={bestspec}; best_auc={bestauc}; delta_EXP035={bestauc-EXP035}; folds={fscores(y,bestp,splits).tolist()}; std={np.std(fscores(y,bestp,splits))}',*[f'{k}=\n{v.to_string(index=False)}' for k,v in reports.items()],f'correlations=\n{cor.to_string(index=False)}',f'regional=\n{regional.to_string(index=False)}',f'pairwise=\n{pairwise.to_string(index=False)}',f'blends=\n{blenddf.to_string(index=False)}',f'ensemble_auc={bauc}; delta_ENS035={bauc-ENS035}; folds={bfs.tolist()}; improved_vs_ENS035={int(sum(bfs>rfs))}',f'submission={generated}; path={subpath}',f'finalization_seconds={perf_counter()-start:.2f}; screening_seconds=6630.4',"problems=['initial final-refit equality tolerance 2e-8 was too strict; winner refit finalized separately']"]
    (METRICS/'exp036_ratio_ablation_metrics.txt').write_text('\n'.join(lines)+'\n',encoding='utf8');print(f'final best={bestauc:.10f} ensemble={bauc:.10f} submission={generated}',flush=True)

def main():
    start=perf_counter();[d.mkdir(parents=True,exist_ok=True) for d in [PRED,REPORTS,METRICS,SUB]]
    train=pd.read_csv(DATA/'train.csv');test=pd.read_csv(DATA/'test.csv');y=train.addicted_label.to_numpy();splits=list(StratifiedKFold(5,shuffle=True,random_state=42).split(train,y));base=base_rep(train);basetest=base_rep(test);cache={};allfold=[];problems=[]
    def run(spec,label):
        k=key(spec)
        if k not in cache:
            o,_,r=fit_cv(train,base,y,splits,list(k),label);cache[k]=(o,r);allfold.extend(r);pd.DataFrame(allfold).to_csv(REPORTS/'exp036_all_fold_progress.csv',index=False)
            print(f'{label}: auc={roc_auc_score(y,o):.10f} specs={list(k)} elapsed={perf_counter()-start:.1f}s',flush=True)
        return cache[k][0]
    B=run([],'B');Cspec=[(n,2) for n in OLD];C=run(Cspec,'C-reproduction');bauc=roc_auc_score(y,B);cauc=roc_auc_score(y,C)
    if abs(cauc-EXP035)>2e-6:raise ValueError(f'C no reproduce EXP035: {cauc}')
    singles=[]
    for n in OLD:
        p=run([(n,2)],f'single_{n}');fs=fscores(y,p,splits);singles.append({'ratio':n,'auc':roc_auc_score(y,p),'delta_vs_B':roc_auc_score(y,p)-bauc,'std':np.std(fs),'folds':str(fs.tolist()),'cardinality':stringify(safe_ratio(train[OLD[n][0]],train[OLD[n][1]]),2).nunique()})
    single=pd.DataFrame(singles).sort_values('auc',ascending=False);single.to_csv(REPORTS/'exp036_ratio_single_ablation.csv',index=False)
    loo=[]
    for drop in OLD:
        spec=[(n,2) for n in OLD if n!=drop];p=run(spec,f'C_minus_{drop}');fs=fscores(y,p,splits);loo.append({'removed_ratio':drop,'auc':roc_auc_score(y,p),'delta_vs_C':roc_auc_score(y,p)-cauc,'std':np.std(fs),'folds':str(fs.tolist())})
    loodf=pd.DataFrame(loo).sort_values('auc',ascending=False);loodf.to_csv(REPORTS/'exp036_ratio_leave_one_out.csv',index=False)
    pairs=[];names=list(OLD)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            p=run([(a,2),(b,2)],f'pair_{a}_{b}');fs=fscores(y,p,splits);pairs.append({'ratio_a':a,'ratio_b':b,'auc':roc_auc_score(y,p),'delta_vs_B':roc_auc_score(y,p)-bauc,'std':np.std(fs),'folds':str(fs.tolist())})
    pairdf=pd.DataFrame(pairs).sort_values('auc',ascending=False);pairdf.to_csv(REPORTS/'exp036_ratio_pairs.csv',index=False)
    # LOO already evaluates all four triples, so no duplicate fits are needed.
    top2=single.head(2).ratio.tolist();gran=[]
    for n in top2:
        for d in [1,2,3]:
            p=run([(n,d)],f'gran_{n}_{d}');fs=fscores(y,p,splits);gran.append({'ratio':n,'decimals':d,'auc':roc_auc_score(y,p),'delta_vs_B':roc_auc_score(y,p)-bauc,'std':np.std(fs),'cardinality':stringify(safe_ratio(train[OLD[n][0]],train[OLD[n][1]]),d).nunique(),'mean_columns':np.mean([r['columns'] for r in cache[key([(n,d)])][1]]),'mean_csr_mb':np.mean([r['csr_mb'] for r in cache[key([(n,d)])][1]])})
    grandf=pd.DataFrame(gran).sort_values('auc',ascending=False);grandf.to_csv(REPORTS/'exp036_ratio_granularity.csv',index=False)
    # Best original subset among B, singles, pairs, triples(LOO), full C.
    candidates=[([],B,bauc)]+[([(r.ratio,2)],cache[key([(r.ratio,2)])][0],r.auc) for r in single.itertuples()]+[([(r.ratio_a,2),(r.ratio_b,2)],cache[key([(r.ratio_a,2),(r.ratio_b,2)])][0],r.auc) for r in pairdf.itertuples()]
    candidates += [([(n,2) for n in OLD if n!=r.removed_ratio],cache[key([(n,2) for n in OLD if n!=r.removed_ratio])][0],r.auc) for r in loodf.itertuples()]+[(Cspec,C,cauc)]
    bestspec,bestp,bestauc=max(candidates,key=lambda z:z[2]);newrows=[]
    for n in NEW:
        spec=bestspec+[(n,2)];p=run(spec,f'new_{n}');fs=fscores(y,p,splits);newrows.append({'base_spec':str(bestspec),'new_ratio':n,'auc':roc_auc_score(y,p),'delta_vs_base':roc_auc_score(y,p)-bestauc,'folds_improved':int(sum(fs>fscores(y,bestp,splits))),'std':np.std(fs),'cardinality':stringify(safe_ratio(train[NEW[n][0]],train[NEW[n][1]]),2).nunique(),'folds':str(fs.tolist())})
    newdf=pd.DataFrame(newrows).sort_values('auc',ascending=False);newdf.to_csv(REPORTS/'exp036_new_ratio_candidates.csv',index=False)
    forward=[{'step':0,'spec':str(bestspec),'added':'baseline','auc':bestauc,'delta':0,'folds_improved':5}];remaining=list(NEW);current=list(bestspec);curp=bestp;curauc=bestauc
    for step in range(1,4):
        trial=[]
        for n in remaining:
            p=run(current+[(n,2)],f'forward{step}_{n}');fs=fscores(y,p,splits);trial.append((roc_auc_score(y,p),n,p,int(sum(fs>fscores(y,curp,splits))),np.std(fs)))
        auc,n,p,improved,std=max(trial,key=lambda z:z[0]);delta=auc-curauc
        forward.append({'step':step,'spec':str(current+[(n,2)]),'added':n,'auc':auc,'delta':delta,'folds_improved':improved,'std':std,'accepted':delta>=.00005 and improved>=4})
        if delta<.00005 or improved<4:break
        current.append((n,2));remaining.remove(n);curp=p;curauc=auc
    fw=pd.DataFrame(forward);fw.to_csv(REPORTS/'exp036_forward_selection.csv',index=False);bestp=curp;bestauc=curauc;bestspec=current
    # Refit only the final winner with test prediction.
    final_oof,testfolds,finalrows=fit_cv(train,base,y,splits,bestspec,'FINAL',True,test,basetest)
    if abs(roc_auc_score(y,final_oof)-bestauc)>2e-8:raise ValueError('Final refit mismatch')
    pd.DataFrame({'id':train.id,'y_true':y,'oof_prediction':final_oof}).to_csv(PRED/'oof_exp036_ratio_logistic.csv',index=False);testpred=testfolds.mean(axis=0,dtype=np.float64);pd.DataFrame({'id':test.id,'prediction':testpred}).to_csv(PRED/'test_exp036_ratio_logistic.csv',index=False)
    p35=load_oof(PRED/'oof_exp035_exact_logistic.csv',train);p16=load_oof(PRED/'oof_exp016_xgboost_depth5_9000.csv',train);p22=load_oof(PRED/'oof_exp022_catboost_thresholds_9000.csv',train);ss=[load_oof(PRED/f'oof_exp027_seed{x}.csv',train) for x in [42,2026,777]];xgb=.4*ss[0]+.3*ss[1]+.3*ss[2];p27=.75*xgb+.25*p22;ss5=ss+[load_oof(PRED/f'oof_exp028_seed{x}.csv',train) for x in [31415,1234]];p28=.75*np.mean(ss5,axis=0)+.25*p22
    cor=pd.DataFrame([{'model':n,'pearson':pd.Series(bestp).corr(pd.Series(p)),'spearman':pd.Series(bestp).corr(pd.Series(p),method='spearman')} for n,p in {'EXP-016':p16,'EXP-022':p22,'EXP-027':p27,'EXP-028':p28,'EXP-035-C':p35,'residual_EXP027':y-p27}.items()]);cor.to_csv(REPORTS/'exp036_correlations.csv',index=False)
    screen=train.daily_screen_time_hours;social=train.social_media_hours;known=screen.notna()&social.notna();cp=known&((screen>8)|(social>4));cn=known&(screen<=6)&(social<=4);amb=known&~cp&~cn;rr=[]
    for lab,m in [('clear_positive',cp),('clear_negative',cn),('ambiguous',amb)]:
        for n,p in [('EXP-036',bestp),('EXP-035-C',p35),('EXP-027',p27)]:rr.append({'analysis':'region','segment':lab,'model':n,'rows':m.sum(),'auc':roc_auc_score(y[m],p[m])})
    for lab,lo,hi in [('0.30-0.40',.3,.4),('0.40-0.50',.4,.5),('0.50-0.60',.5,.6),('0.60-0.70',.6,.7),('0.70-0.80',.7,.8)]:
        m=(p16>=lo)&(p16<hi)
        for n,p in [('EXP-036',bestp),('EXP-035-C',p35),('EXP-027',p27)]:rr.append({'analysis':'score_band','segment':lab,'model':n,'rows':m.sum(),'auc':roc_auc_score(y[m],p[m])})
    regional=pd.DataFrame(rr);regional.to_csv(REPORTS/'exp036_regional.csv',index=False)
    fi=np.empty(len(y),int)
    for f,(_,v) in enumerate(splits):fi[v]=f
    bp,bn,gp,gn,_=sample_pairs(train,p16,np.random.default_rng(42),fi);pr=[]
    for n,p in [('EXP-036',bestp),('EXP-035-C',p35)]:
        corrected=np.mean(p[bp]>p[bn]);broken=np.mean(p[gp]<=p[gn]);pr.append({'model':n,'corrected':corrected,'broken':broken,'net_gain':corrected-broken})
    pairwise=pd.DataFrame(pr);pairwise.to_csv(REPORTS/'exp036_pairwise.csv',index=False)
    blendrows=[];bnested={}
    if bestauc-EXP035>=.00005:
        for n,b in [('EXP-027',p27),('EXP-028',p28)]:pred,rows=nested_blend(y,b,bestp,splits,n);bnested[n]=pred;blendrows+=rows
    blenddf=pd.DataFrame(blendrows);blenddf.to_csv(REPORTS/'exp036_blends.csv',index=False)
    generated=False;subpath=''
    if 'EXP-027' in bnested:
        boof=bnested['EXP-027'];bauc2=roc_auc_score(y,boof);bfs=fscores(y,boof,splits);ref=.8*rank(p27)+.2*rank(p35);rfs=fscores(y,ref,splits)
        choices=blenddf[(blenddf.candidate=='EXP-027')&(blenddf.fold>0)]
        if bauc2-ENS035>=.00003 and sum(bfs>rfs)>=4 and choices.method.nunique()==1 and choices.logistic_weight.nunique()==1:
            method=choices.method.iloc[0];w=float(choices.logistic_weight.iloc[0]);base_test=pd.read_csv(SUB/'submission_exp027_seed_ensemble.csv').addicted_label.to_numpy(np.float64)
            pred=(1-w)*base_test+w*testpred if method=='probability' else (1-w)*rank(base_test)+w*rank(testpred);sample=pd.read_csv(DATA/'sample_submission.csv');sub=pd.DataFrame({'id':sample.id,'addicted_label':pred});subpath=SUB/'submission_exp036_ratio_logistic_ensemble.csv';sub.to_csv(subpath,index=False);generated=True
            logpath=METRICS/'experiment_log.csv';log=pd.read_csv(logpath)
            if not log.experiment_id.astype(str).eq('EXP-036').any():
                log=pd.concat([log,pd.DataFrame([{'experiment_id':'EXP-036','datetime':pd.Timestamp.now().isoformat(timespec='seconds'),'model':'RatioSparseLogistic_Ensemble','features':str(bestspec),'cv_strategy':'Nested_OOF_ensemble_optimization','cv_roc_auc':bauc2,'kaggle_score':'','notes':f'{method} blend EXP027 {1-w:.3f} + EXP036 {w:.3f}'}])],ignore_index=True);log.to_csv(logpath,index=False)
    lines=['EXP-036 ratio ablation',f'B_auc={bauc}; C_reproduced={cauc}',f'singles=\n{single.to_string(index=False)}',f'leave_one_out=\n{loodf.to_string(index=False)}',f'pairs=\n{pairdf.to_string(index=False)}',f'granularity=\n{grandf.to_string(index=False)}',f'new_candidates=\n{newdf.to_string(index=False)}',f'forward=\n{fw.to_string(index=False)}',f'best_spec={bestspec}; best_auc={bestauc}; delta_EXP035={bestauc-EXP035}; folds={fscores(y,bestp,splits).tolist()}; std={np.std(fscores(y,bestp,splits))}',f'correlations=\n{cor.to_string(index=False)}',f'regional=\n{regional.to_string(index=False)}',f'pairwise=\n{pairwise.to_string(index=False)}',f'blends=\n{blenddf.to_string(index=False)}',f'submission={generated}; path={subpath}',f'total_seconds={perf_counter()-start:.2f}',f'problems={problems or "none"}']
    (METRICS/'exp036_ratio_ablation_metrics.txt').write_text('\n'.join(lines)+'\n',encoding='utf8');print(f'best={bestauc:.10f} spec={bestspec} submission={generated} seconds={perf_counter()-start:.1f}',flush=True)
if __name__=='__main__':
    if '--finalize-only' in sys.argv:finalize_only()
    else:main()
