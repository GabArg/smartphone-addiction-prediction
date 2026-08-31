"""EXP-037: sparse categorical relational features with strict forward selection."""
from __future__ import annotations
import gc
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
from src.features.exact_values import ORIGINAL, safe_ratio, stringify
from src.features.relational import (
    FINAL_RATIO_RELATIONS,
    add_categorical_relations,
    build_base_representation as base_rep,
    relational_value,
)
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

BASE_R=dict(FINAL_RATIO_RELATIONS)
DIFF={'screen_minus_social':('daily_screen_time_hours','social_media_hours'),'screen_minus_work':('daily_screen_time_hours','work_study_hours'),'screen_minus_gaming':('daily_screen_time_hours','gaming_hours'),'weekend_minus_screen':('weekend_screen_time','daily_screen_time_hours'),'social_minus_gaming':('social_media_hours','gaming_hours'),'social_minus_work':('social_media_hours','work_study_hours'),'work_minus_gaming':('work_study_hours','gaming_hours'),'weekend_minus_social':('weekend_screen_time','social_media_hours')}
INV={'screen_over_work':('daily_screen_time_hours','work_study_hours'),'screen_over_social':('daily_screen_time_hours','social_media_hours'),'screen_over_gaming':('daily_screen_time_hours','gaming_hours'),'social_over_work':('social_media_hours','work_study_hours'),'social_over_gaming':('social_media_hours','gaming_hours'),'work_over_gaming':('work_study_hours','gaming_hours')}
EXP036=.9635099637855411;ENS036=.9670368278351087;EXP027=.965919188052602;EXP028=.9659307051390922;WEIGHTS=[.20,.225,.25,.275,.30,.325,.35,.375,.40]

def rss():return psutil.Process().memory_info().rss/1024**3
def feature_value(raw,name):
    if name=='screen_minus_social':return relational_value(raw,name)
    if name in DIFF:a,b=DIFF[name];return raw[a]-raw[b]
    base=name.removeprefix('log__');a,b=INV[base];z=safe_ratio(raw[a],raw[b]);return np.sign(z)*np.log1p(np.abs(z)) if name.startswith('log__') else z
def add_features(base,raw,spec):
    x=add_categorical_relations(base,raw,include_screen_difference=False)
    for n,digits in spec:x[n]=stringify(feature_value(raw,n),digits)
    return x
def ckey(spec):return tuple(sorted(spec))
def fit_cv(raw,base,y,splits,spec,label,test=None,basetest=None):
    x=add_features(base,raw,spec);xt=add_features(basetest,test,spec) if test is not None else None;oof=np.zeros(len(y));tests=[];rows=[]
    for f,(tr,va) in enumerate(splits,1):
        st=perf_counter();peak=rss();enc=OneHotEncoder(handle_unknown='ignore',dtype=np.float32,sparse_output=True);a=sparse.csr_matrix(enc.fit_transform(x.iloc[tr]),dtype=np.float32);b=sparse.csr_matrix(enc.transform(x.iloc[va]),dtype=np.float32);e=sparse.csr_matrix(enc.transform(xt),dtype=np.float32) if xt is not None else None;peak=max(peak,rss())
        m=LogisticRegression(solver='liblinear',C=1.,max_iter=1000,random_state=42);m.fit(a,y[tr]);oof[va]=m.predict_proba(b)[:,1]
        if e is not None:tests.append(m.predict_proba(e)[:,1].astype(np.float64))
        rows.append({'label':label,'spec':str(spec),'fold':f,'auc':roc_auc_score(y[va],oof[va]),'columns':a.shape[1],'nnz':a.nnz,'csr_mb':(a.data.nbytes+a.indices.nbytes+a.indptr.nbytes)/1024**2,'peak_rss_gb':peak,'seconds':perf_counter()-st})
        del enc,m,a,b,e;gc.collect()
    del x,xt;gc.collect();return oof,np.asarray(tests),rows
def fs(y,p,splits):return np.asarray([roc_auc_score(y[v],p[v]) for _,v in splits])
def nested_blend(y,base,other,splits,label):
    out=np.zeros(len(y));rows=[];rb=rank(base);ro=rank(other)
    for f,(tr,va) in enumerate(splits,1):
        opts=[]
        for method in ['probability','rank']:
            for w in WEIGHTS:
                p=(1-w)*base+w*other if method=='probability' else (1-w)*rb+w*ro;opts.append((roc_auc_score(y[tr],p[tr]),-w,method,w,p))
        z=max(opts,key=lambda q:(q[0],q[1]));out[va]=z[4][va];rows.append({'candidate':label,'fold':f,'method':z[2],'logistic_weight':z[3],'selection_auc':z[0]})
    ff=fs(y,out,splits);rows.append({'candidate':label,'fold':0,'auc':roc_auc_score(y,out),'std':np.std(ff),'folds':str(ff.tolist()),'improved_folds':int(sum(ff>fs(y,base,splits)))})
    return out,rows
def nested_triple(y,p27,p22,logi,splits):
    out=np.zeros(len(y));rows=[];r27,r22,rl=rank(p27),rank(p22),rank(logi)
    for f,(tr,va) in enumerate(splits,1):
        opts=[]
        for method in ['probability','rank']:
            for lw in [.20,.25,.30,.35]:
                for cw in [.05,.10,.15,.20]:
                    if lw+cw>=1:continue
                    p=(1-lw-cw)*p27+cw*p22+lw*logi if method=='probability' else (1-lw-cw)*r27+cw*r22+lw*rl
                    opts.append((roc_auc_score(y[tr],p[tr]),method,lw,cw,p))
        z=max(opts,key=lambda q:q[0]);out[va]=z[4][va];rows.append({'candidate':'TRIPLE','fold':f,'method':z[1],'logistic_weight':z[2],'catboost_weight':z[3],'selection_auc':z[0]})
    ff=fs(y,out,splits);rows.append({'candidate':'TRIPLE','fold':0,'auc':roc_auc_score(y,out),'std':np.std(ff),'folds':str(ff.tolist())});return out,rows

def main():
    start=perf_counter();[d.mkdir(parents=True,exist_ok=True) for d in [PRED,REPORTS,METRICS,SUB]];train=pd.read_csv(DATA/'train.csv');test=pd.read_csv(DATA/'test.csv');y=train.addicted_label.to_numpy();splits=list(StratifiedKFold(5,shuffle=True,random_state=42).split(train,y));base=base_rep(train);basetest=base_rep(test);basep=load_oof(PRED/'oof_exp036_ratio_logistic.csv',train)
    reproduced=roc_auc_score(y,basep)
    if abs(reproduced-EXP036)>2e-6:raise ValueError(f'EXP036 control mismatch {reproduced}')
    cache={};allrows=[]
    def run(spec,label):
        k=ckey(spec)
        if k not in cache:
            p,_,r=fit_cv(train,base,y,splits,list(k),label);cache[k]=(p,r);allrows.extend(r);pd.DataFrame(allrows).to_csv(REPORTS/'exp037_all_fold_progress.csv',index=False);print(f'{label} auc={roc_auc_score(y,p):.10f} elapsed={perf_counter()-start:.1f}',flush=True)
        return cache[k][0]
    cand=[]
    for n in DIFF:
        p=run([(n,2)],f'diff_{n}');ff=fs(y,p,splits);cand.append({'feature':n,'auc':roc_auc_score(y,p),'delta':roc_auc_score(y,p)-reproduced,'folds_improved':int(sum(ff>fs(y,basep,splits))),'std':np.std(ff),'cardinality':stringify(feature_value(train,n),2).nunique(),'folds':str(ff.tolist())})
    diffdf=pd.DataFrame(cand).sort_values('auc',ascending=False);diffdf.to_csv(REPORTS/'exp037_difference_candidates.csv',index=False)
    gran=[]
    for n in diffdf.head(2).feature:
        for d in [1,2,3]:
            p=run([(n,d)],f'gran_{n}_{d}');ff=fs(y,p,splits);gran.append({'feature':n,'decimals':d,'auc':roc_auc_score(y,p),'delta':roc_auc_score(y,p)-reproduced,'std':np.std(ff),'cardinality':stringify(feature_value(train,n),d).nunique(),'folds_improved':int(sum(ff>fs(y,basep,splits)))})
    grandf=pd.DataFrame(gran).sort_values('auc',ascending=False);grandf.to_csv(REPORTS/'exp037_difference_granularity.csv',index=False)
    invrows=[]
    for n in INV:
        p=run([(n,2)],f'inverse_{n}');ff=fs(y,p,splits);invrows.append({'feature':n,'auc':roc_auc_score(y,p),'delta':roc_auc_score(y,p)-reproduced,'folds_improved':int(sum(ff>fs(y,basep,splits))),'std':np.std(ff),'cardinality':stringify(feature_value(train,n),2).nunique(),'folds':str(ff.tolist())})
    invdf=pd.DataFrame(invrows).sort_values('auc',ascending=False)
    logs=[]
    passing=invdf[invdf.delta>=.00003].head(2).feature.tolist()
    for n in passing:
        ln='log__'+n;p=run([(ln,2)],f'log_{n}');ff=fs(y,p,splits);logs.append({'feature':ln,'auc':roc_auc_score(y,p),'delta':roc_auc_score(y,p)-reproduced,'folds_improved':int(sum(ff>fs(y,basep,splits))),'std':np.std(ff),'cardinality':stringify(feature_value(train,ln),2).nunique(),'folds':str(ff.tolist())})
    invout=pd.concat([invdf,pd.DataFrame(logs)],ignore_index=True);invout.to_csv(REPORTS/'exp037_inverse_ratios.csv',index=False)
    # Candidate pool uses best tested granularity for top differences, otherwise 2 decimals.
    pool={n:(n,2) for n in list(DIFF)+list(INV)}
    for n in diffdf.head(2).feature:
        q=grandf[grandf.feature.eq(n)].sort_values(['auc','std'],ascending=[False,True]).iloc[0];pool[n]=(n,int(q.decimals))
    for n in passing:pool['log__'+n]=('log__'+n,2)
    current=[];curp=basep;curauc=reproduced;forward=[{'step':0,'feature':'baseline','spec':'[]','auc':curauc,'delta':0,'folds_improved':5,'accepted':True}]
    remaining=dict(pool)
    for step in range(1,5):
        trials=[]
        for keyname,spec in remaining.items():
            p=run(current+[spec],f'forward{step}_{keyname}');ff=fs(y,p,splits);trials.append((roc_auc_score(y,p),keyname,spec,p,int(sum(ff>fs(y,curp,splits))),np.std(ff)))
        auc,n,spec,p,improved,std=max(trials,key=lambda q:q[0]);delta=auc-curauc;accepted=delta>=.00004 and improved>=4 and std<=np.std(fs(y,curp,splits))+.00005
        forward.append({'step':step,'feature':n,'spec':str(current+[spec]),'auc':auc,'delta':delta,'folds_improved':improved,'std':std,'accepted':accepted})
        if not accepted:break
        current.append(spec);curp=p;curauc=auc;remaining.pop(n)
    fw=pd.DataFrame(forward);fw.to_csv(REPORTS/'exp037_forward_selection.csv',index=False)
    # Final leave-one-out; remove any accepted feature that no longer contributes.
    abl=[];changed=True
    while changed and current:
        changed=False
        for spec in list(current):
            reduced=[x for x in current if x!=spec];p=basep if not reduced else run(reduced,'ablate_'+spec[0]);auc=roc_auc_score(y,p);abl.append({'removed':spec[0],'full_auc':curauc,'reduced_auc':auc,'contribution':curauc-auc,'keep':curauc-auc>0})
            if auc>=curauc:current=reduced;curp=p;curauc=auc;changed=True;break
    abldf=pd.DataFrame(abl);abldf.to_csv(REPORTS/'exp037_final_ablation.csv',index=False)
    # Exact final refit with test.
    finalp,testfolds,finalrows=fit_cv(train,base,y,splits,current,'FINAL',test,basetest);finalauc=roc_auc_score(y,finalp);testpred=testfolds.mean(axis=0,dtype=np.float64)
    pd.DataFrame({'id':train.id,'y_true':y,'oof_prediction':finalp}).to_csv(PRED/'oof_exp037_relational_logistic.csv',index=False);pd.DataFrame({'id':test.id,'prediction':testpred}).to_csv(PRED/'test_exp037_relational_logistic.csv',index=False)
    p16=load_oof(PRED/'oof_exp016_xgboost_depth5_9000.csv',train);p22=load_oof(PRED/'oof_exp022_catboost_thresholds_9000.csv',train);p35=load_oof(PRED/'oof_exp035_exact_logistic.csv',train);ss=[load_oof(PRED/f'oof_exp027_seed{x}.csv',train) for x in [42,2026,777]];p27=.75*(.4*ss[0]+.3*ss[1]+.3*ss[2])+.25*p22;ss5=ss+[load_oof(PRED/f'oof_exp028_seed{x}.csv',train) for x in [31415,1234]];p28=.75*np.mean(ss5,axis=0)+.25*p22
    cor=pd.DataFrame([{'model':n,'pearson':pd.Series(finalp).corr(pd.Series(p)),'spearman':pd.Series(finalp).corr(pd.Series(p),method='spearman')} for n,p in {'EXP-016':p16,'EXP-022':p22,'EXP-027':p27,'EXP-028':p28,'EXP-035-C':p35,'EXP-036':basep,'residual_EXP027':y-p27}.items()]);cor.to_csv(REPORTS/'exp037_correlations.csv',index=False)
    screen=train.daily_screen_time_hours;social=train.social_media_hours;known=screen.notna()&social.notna();cp=known&((screen>8)|(social>4));cn=known&(screen<=6)&(social<=4);amb=known&~cp&~cn;rr=[]
    for lab,m in [('clear_positive',cp),('clear_negative',cn),('ambiguous',amb)]:
        for n,p in [('EXP-037',finalp),('EXP-036',basep),('EXP-027',p27)]:rr.append({'analysis':'region','segment':lab,'model':n,'rows':m.sum(),'auc':roc_auc_score(y[m],p[m])})
    for lab,lo,hi in [('0.30-0.40',.3,.4),('0.40-0.50',.4,.5),('0.50-0.60',.5,.6),('0.60-0.70',.6,.7),('0.70-0.80',.7,.8)]:
        m=(p16>=lo)&(p16<hi)
        for n,p in [('EXP-037',finalp),('EXP-036',basep),('EXP-027',p27)]:rr.append({'analysis':'score_band','segment':lab,'model':n,'rows':m.sum(),'auc':roc_auc_score(y[m],p[m])})
    regional=pd.DataFrame(rr);regional.to_csv(REPORTS/'exp037_regional.csv',index=False)
    fi=np.empty(len(y),int)
    for f,(_,v) in enumerate(splits):fi[v]=f
    bp,bn,gp,gn,_=sample_pairs(train,p16,np.random.default_rng(42),fi);pr=[]
    for n,p in [('EXP-037',finalp),('EXP-036',basep),('EXP-035',p35),('EXP-022',p22)]:
        corrected=np.mean(p[bp]>p[bn]);broken=np.mean(p[gp]<=p[gn]);pr.append({'model':n,'corrected':corrected,'broken':broken,'net_gain':corrected-broken})
    pair=pd.DataFrame(pr);pair.to_csv(REPORTS/'exp037_pairwise.csv',index=False)
    blendrows=[];bn={}
    if finalauc-reproduced>=.00005:
        for n,b in [('EXP-027',p27),('EXP-028',p28)]:z,rows=nested_blend(y,b,finalp,splits,n);bn[n]=z;blendrows+=rows
        z,rows=nested_triple(y,p27,p22,finalp,splits);bn['TRIPLE']=z;blendrows+=rows
    blends=pd.DataFrame(blendrows);blends.to_csv(REPORTS/'exp037_blends.csv',index=False)
    generated=False;subpath='';bestname=max(bn,key=lambda n:roc_auc_score(y,bn[n])) if bn else None
    # EXP-028 is a diagnostic OOF control and has no persisted deployable test
    # prediction. Evaluate submission eligibility on the deployable EXP-027
    # blend instead of allowing a slightly higher control score to suppress it.
    deployname='EXP-027' if 'EXP-027' in bn else None
    if deployname:
        boof=bn[deployname];bauc=roc_auc_score(y,boof);bfs=fs(y,boof,splits);ref=.7*rank(p27)+.3*rank(basep);rfs=fs(y,ref,splits);choices=blends[(blends.candidate==deployname)&(blends.fold>0)]
        if bauc-ENS036>=.00003 and sum(bfs>rfs)>=4 and np.min(bfs-rfs)>=-.00003:
            bt=pd.read_csv(SUB/'submission_exp027_seed_ensemble.csv').addicted_label.to_numpy(np.float64);foldtest=[]
            for r,lp in zip(choices.itertuples(),testfolds):foldtest.append((1-r.logistic_weight)*bt+r.logistic_weight*lp if r.method=='probability' else (1-r.logistic_weight)*rank(bt)+r.logistic_weight*rank(lp))
            sample=pd.read_csv(DATA/'sample_submission.csv');sub=pd.DataFrame({'id':sample.id,'addicted_label':np.mean(foldtest,axis=0)});subpath=SUB/'submission_exp037_relational_logistic_ensemble.csv';sub.to_csv(subpath,index=False);generated=True
            logpath=METRICS/'experiment_log.csv';log=pd.read_csv(logpath)
            if not log.experiment_id.astype(str).eq('EXP-037').any():log=pd.concat([log,pd.DataFrame([{'experiment_id':'EXP-037','datetime':pd.Timestamp.now().isoformat(timespec='seconds'),'model':'RelationalSparseLogistic_Ensemble','features':str(current),'cv_strategy':'Nested_OOF_ensemble_optimization','cv_roc_auc':bauc,'kaggle_score':'','notes':'fold-safe blend EXP027 + EXP037'}])],ignore_index=True);log.to_csv(logpath,index=False)
    lines=['EXP-037 relational sparse Logistic',f'EXP036_reproduced={reproduced}',f'differences=\n{diffdf.to_string(index=False)}',f'granularity=\n{grandf.to_string(index=False)}',f'inverse_and_logs=\n{invout.to_string(index=False)}',f'forward=\n{fw.to_string(index=False)}',f'ablation=\n{abldf.to_string(index=False)}',f'final_features={current}; final_auc={finalauc}; delta_EXP036={finalauc-reproduced}; folds={fs(y,finalp,splits).tolist()}; std={np.std(fs(y,finalp,splits))}',f'correlations=\n{cor.to_string(index=False)}',f'regional=\n{regional.to_string(index=False)}',f'pairwise=\n{pair.to_string(index=False)}',f'blends=\n{blends.to_string(index=False)}',f'submission={generated};path={subpath}',f'total_seconds={perf_counter()-start:.2f}','problems=none']
    (METRICS/'exp037_relational_metrics.txt').write_text('\n'.join(lines)+'\n',encoding='utf8');print(f'final={finalauc:.10f} features={current} submission={generated} seconds={perf_counter()-start:.1f}',flush=True)
if __name__=='__main__':main()
