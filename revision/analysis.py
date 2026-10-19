"""Outer-prediction reports, quality/confounding audits, repeatability and figures."""
from pathlib import Path
import itertools
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import roc_curve, auc
from .features import finite_mean, patient_weights
from .metrics import (scores, bootstrap_metrics, clinical_curves, prevalence_table,
                      paired_cluster_comparison, holm)
from .pipeline import write_json


def aggregate_repeats(frame, nclasses):
    """One row/site; mean probabilities. Require coverage in EVERY repetition.

    Sensitivity and fixed-specificity decisions are majority votes of thresholds
    selected independently inside each outer-training population; ties refer.
    """
    probcols = [f'p_{k}' for k in range(nclasses)]
    rawcols = [f'raw_p_{k}' for k in range(nclasses)]
    rows = []
    for sample, group in frame.groupby('sample_id', sort=False):
        if group.repeat.duplicated().any():
            raise ValueError('More than one outer prediction per site/repetition/stream')
        row = group.iloc[0][['task','stream','objective','model','evaluation','sample_id','patient_id','y_true']].to_dict()
        row['covered'] = bool(group.covered.all())
        row['repeat_coverage'] = float(group.covered.mean())
        for col in probcols + rawcols:
            row[col] = group[col].mean() if row['covered'] else np.nan
        row['y_pred'] = (-1 if not row['covered'] else int((group.y_pred == 1).mean() >= .5)
                         if group.objective.iloc[0] == 'sensitivity' and nclasses == 2 else int(np.argmax([row[c] for c in probcols])))
        for col in [c for c in group if c.startswith('pred_at_specificity_')]:
            row[col] = int((group[col] == 1).mean() >= .5) if row['covered'] else -1
        rows.append(row)
    return pd.DataFrame(rows)


def patient_frame(frame, nclasses):
    """Binary: any-positive pathology, max site risk. Multiclass: homogeneous patients.

    Mixed-pathology patients remain in the equal-patient site analysis; they are
    excluded only from the separate homogeneous-patient multiclass endpoint.
    """
    rows = []; excluded = []
    for patient, g in frame.groupby('patient_id'):
        if nclasses > 2 and g.y_true.nunique() > 1:
            excluded.append(patient); continue
        covered = bool(g.covered.all())
        if nclasses == 2:
            prob = np.array([1 - g.p_1.max(), g.p_1.max()])
            truth = int(g.y_true.max())
            pred = int(g.y_pred.max()) if covered else -1
        else:
            prob = g[[f'p_{i}' for i in range(nclasses)]].mean().to_numpy()
            truth = int(g.y_true.iloc[0]); pred = int(prob.argmax()) if covered else -1
        row = dict(sample_id=patient, patient_id=patient, y_true=truth, y_pred=pred, covered=covered)
        row.update({f'p_{k}': float(prob[k]) if covered else np.nan for k in range(nclasses)})
        rows.append(row)
    return pd.DataFrame(rows), excluded


def _pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def plot_outputs(folder, frame, classes, title):
    plt = _pyplot(); folder=Path(folder); folder.mkdir(parents=True, exist_ok=True)
    covered = frame[frame.covered]
    if not len(covered):
        return
    p = covered[[f'p_{k}' for k in range(len(classes))]].to_numpy()
    y = covered.y_true.to_numpy(int); pred = covered.y_pred.to_numpy(int)
    summary, perclass, cm = scores(y,p,pred)
    for normalized in [False, True]:
        shown = np.divide(cm, cm.sum(1, keepdims=True), out=np.zeros_like(cm,dtype=float), where=cm.sum(1, keepdims=True)>0) if normalized else cm
        fig, ax = plt.subplots(figsize=(6,5)); image = ax.imshow(shown, cmap='Blues')
        ax.set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=classes, yticklabels=classes,
               xlabel='Predicted class', ylabel='Reference class', title=title)
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        for i,j in itertools.product(range(len(classes)),repeat=2):
            ax.text(j,i,f'{shown[i,j]:.2f}' if normalized else f'{shown[i,j]:.0f}',ha='center',va='center')
        fig.colorbar(image,ax=ax); fig.tight_layout()
        suffix = '_norm' if normalized else ''
        fig.savefig(folder / f'CM{suffix}.png',dpi=160); fig.savefig(folder / f'CM{suffix}.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5)); bundles={}
    for k,name in enumerate(classes):
        yy=(y==k).astype(int)
        if len(np.unique(yy))<2: continue
        fpr,tpr,threshold=roc_curve(yy,p[:,k])
        ax.plot(fpr,tpr,label=f'{name} (AUC {auc(fpr,tpr):.3f})')
        bundles[name]={'class_index':k,'fpr':fpr.tolist(),'tpr':tpr.tolist(),'thresholds':threshold.tolist()}
    ax.plot([0,1],[0,1],'k--',lw=.7);ax.set(xlabel='False-positive rate',ylabel='Sensitivity',title=title);ax.legend(fontsize='small');fig.tight_layout()
    fig.savefig(folder/'ROC.png',dpi=160);fig.savefig(folder/'ROC.pdf');plt.close(fig)
    write_json(folder/'ROC_metadata.json',{'classes':list(classes),'curves':bundles,'evaluation':'pooled outer predictions; no test-based threshold selection'})
    calibration,decision=clinical_curves(y,p,classes)
    fig,ax=plt.subplots(figsize=(6,5));ax.plot([0,1],[0,1],'k--')
    for name,g in calibration.groupby('class_name'): ax.plot(g.mean_probability,g.observed_fraction,'o-',label=name)
    ax.set(xlabel='Predicted probability',ylabel='Observed fraction',title=title+' calibration');ax.legend(fontsize='small');fig.tight_layout();fig.savefig(folder/'calibration.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5))
    shown_decision=decision[decision.class_name==classes[1]] if len(classes)==2 else decision
    for name,g in shown_decision.groupby('class_name'):
        line,=ax.plot(g.threshold,g.net_benefit,label=name)
        ax.plot(g.threshold,g.treat_all,'--',color=line.get_color(),label=f'Treat all ({name})')
    ax.axhline(0,color='black',lw=.7);ax.set(xlabel='Decision threshold',ylabel='Net benefit (study prevalence)',ylim=(-.25,1),title=title+' decision curves');ax.legend(fontsize='small');fig.tight_layout();fig.savefig(folder/'decision_curve.png',dpi=160);plt.close(fig)


def report_task(dataset, task, predictions, selections, config, outdir):
    _,_,classes=dataset.task(task); k=len(classes); folder=Path(outdir)/task
    reports, class_reports, comparisons = [], [], []
    pooled = {}
    for (evaluation,stream), part in predictions.groupby(['evaluation','stream']):
        frame=aggregate_repeats(part,k);pooled[(evaluation,stream)]=frame
        prefix=f'{evaluation}_{stream}'
        frame.to_csv(folder/f'{prefix}_site_predictions.csv',index=False)
        endpoint,excluded=patient_frame(frame,k)
        write_json(folder/f'{prefix}_patient_endpoint.json',dict(rule='any-positive; max site risk' if k==2 else 'homogeneous-pathology patients only',
                                                                excluded_mixed_pathology_patients=excluded,
                                                                repeat_aggregation='mean probability; require coverage in every repeat; sensitivity decisions majority, ties refer'))
        endpoint.to_csv(folder/f'{prefix}_patient_predictions.csv',index=False)
        for unit,data,weighted in [('site',frame,False),('patient_weighted_site',frame,True),('patient',endpoint,False)]:
            if data.empty: continue
            good=data.covered.to_numpy(bool); c=data.loc[good]
            if c.empty: continue
            pp=c[[f'p_{j}' for j in range(k)]].to_numpy();yy=c.y_true.to_numpy(int);pred=c.y_pred.to_numpy(int)
            # Derive equal-patient weights on the full task BEFORE dropping rejected sites.
            full_weights=patient_weights(data.patient_id.to_numpy()) if weighted else np.ones(len(data))
            w=full_weights[good] if weighted else None
            metrics,pc,cm=scores(yy,pp,pred,w)
            row=dict(task=task,stream=stream,evaluation=evaluation,unit=unit,n_sites=len(data),n_covered=len(c),
                     n_patients=data.patient_id.nunique(),coverage=full_weights[good].sum()/full_weights.sum(),**metrics)
            reports.append(row)
            pc['class_name']=list(classes);pc['task']=task;pc['stream']=stream;pc['evaluation']=evaluation;pc['unit']=unit;class_reports.append(pc)
            pd.DataFrame(cm,index=classes,columns=classes).to_csv(folder/f'{prefix}_{unit}_confusion_counts.csv')
            boot,class_boot=bootstrap_metrics(yy,pp,pred,c.patient_id.to_numpy(),n_boot=config['bootstrap_samples'],
                                   seed=config['seed'],patient_weighted=weighted,base_weights=w,return_perclass=True)
            boot.to_csv(folder/f'{prefix}_{unit}_confidence_intervals.csv',index=False)
            class_boot['class_name']=class_boot.class_index.map(dict(enumerate(classes)))
            class_boot.to_csv(folder/f'{prefix}_{unit}_per_class_confidence_intervals.csv',index=False)
            if unit=='site':
                prevalence_table(pc,config['prevalences']).assign(class_name=lambda d:d.class_index.map(dict(enumerate(classes)))).to_csv(folder/f'{prefix}_prevalence_projections.csv',index=False)
                calibration,decision=clinical_curves(yy,pp,classes)
                calibration.to_csv(folder/f'{prefix}_calibration.csv',index=False)
                decision.to_csv(folder/f'{prefix}_decision_curve.csv',index=False)
                # Cancer/high-risk operating thresholds were selected on inner OOF data, never these outcomes.
                operating=[]
                for col in [v for v in c if v.startswith('pred_at_specificity_')]:
                    m,_,_=scores(yy,pp,c[col].to_numpy(int))
                    operating.append(dict(training_specificity_target=float(col.removeprefix('pred_at_specificity_')),**m))
                pd.DataFrame(operating).to_csv(folder/f'{prefix}_operating_points.csv',index=False)
                if k==2:
                    workflow=data.copy();workflow_pred=workflow.y_pred.to_numpy(int)
                    workflow_pred[~good]=1 # explicit clinical fallback: an unevaluable measurement is referred.
                    from sklearn.metrics import confusion_matrix
                    tn,fp,fn,tp=confusion_matrix(workflow.y_true,workflow_pred,labels=[0,1]).ravel()
                    from .metrics import ratio
                    write_json(folder/f'{prefix}_referral_workflow.json',dict(abstentions_referred=int((~good).sum()),
                               sensitivity=ratio(tp,tp+fn),specificity=ratio(tn,tn+fp),ppv=ratio(tp,tp+fp),
                               npv=ratio(tn,tn+fn),referral_fraction=float(workflow_pred.mean()),
                               note='Prespecified referral fallback, not a scored/calibrated prediction for missing measurements'))
        plotdir=Path(outdir)/f'plots_{task}_{"baseline" if stream.startswith("baseline_") else stream.split("_")[0]}'/stream
        if config.get('plots',True):
            plot_outputs(plotdir,frame,classes,f'{task}: {stream} ({evaluation})')
        # Keep recognizable legacy plot names inside this new results directory only.
        if evaluation=='nested_cv' and (stream.startswith('baseline_') or any(stream==f'{o}_{m}' for o in config['objectives'] for m in config['models'])):
            import shutil
            model=stream.split('_')[-1];metric='baseline' if stream.startswith('baseline_') else stream.split('_')[0]
            legacy=Path(outdir)/f'plots_{task}_{metric}';legacy.mkdir(exist_ok=True)
            stem=f'baseline_{task}_{model}' if metric=='baseline' else f'{task}_{model}_{metric}'
            for src,dst in [('CM.png',f'CM_{stem}.png'),('CM_norm.png',f'CM_{stem}_norm.png'),('ROC.png',f'ROC_{stem}.png'),('ROC.pdf',f'ROC_{stem}.pdf')]:
                if (plotdir/src).exists(): shutil.copy2(plotdir/src,legacy/dst)
    # Predetermined family: every same-model/objective tuned-vs-baseline comparison, per task.
    for model,objective in itertools.product(config['models'],config['objectives']):
        if ('nested_cv',f'{objective}_{model}') not in pooled: continue
        a=pooled[('nested_cv',f'{objective}_{model}')];b=pooled[('nested_cv',f'baseline_{model}')]
        both=a.merge(b,on=['sample_id','patient_id','y_true'],suffixes=('_a','_b'),validate='one_to_one')
        both=both[both.covered_a & both.covered_b]
        if both.empty:continue
        result=paired_cluster_comparison(both.y_true.to_numpy(int),both[[f'p_{j}_a' for j in range(k)]].to_numpy(),
                                         both[[f'p_{j}_b' for j in range(k)]].to_numpy(),both.patient_id.to_numpy(),
                                         n_resamples=config['permutation_samples'],seed=config['seed'])
        comparisons.append(dict(task=task,objective=objective,model=model,**result,
                                conditional_on_common_quality_coverage=True))
    summary=pd.DataFrame(reports);summary.to_csv(folder/'metrics_summary.csv',index=False)
    if class_reports:pd.concat(class_reports).to_csv(folder/'per_class_metrics.csv',index=False)
    # Site/location-stratified results use the SAME untouched outer predictions.
    if 'location' in dataset.table:
        strata=[]
        for (evaluation,stream),frame in pooled.items():
            joined=frame.merge(dataset.table[['sample_id','location']],on='sample_id',validate='one_to_one')
            for location,g in joined.groupby('location',dropna=False):
                c=g[g.covered]
                if c.empty:continue
                m,_,_=scores(c.y_true.to_numpy(int),c[[f'p_{j}' for j in range(k)]].to_numpy(),c.y_pred.to_numpy(int))
                strata.append(dict(stream=stream,evaluation=evaluation,location=location,n_sites=len(g),coverage=len(c)/len(g),**m))
        pd.DataFrame(strata).to_csv(folder/'location_stratified_metrics.csv',index=False)
    export_search.plots=config.get('plots',True)
    export_search(folder,selections,dataset,task)
    return summary,pd.DataFrame(comparisons)


def export_search(folder,selections,dataset,task):
    inner=pd.read_csv(folder/'inner_search.csv')
    for stage,name in [('frequency','FreqSel_ranking'),('mask','MaskSel_ranking'),('z_threshold','zthres_tuning')]:
        part=inner[inner.stage==stage].copy()
        part.to_csv(folder/f'{name}_{task}_inner_folds.csv',index=False)
    # Outer refit frequencies/patterns are the actual identities, separated from subset-size rankings.
    actual=[];timings=[];hardware=[]
    for item in selections:
        actual.append(dict(stream=item['stream'],repeat=item['repeat'],outer_fold=item['outer_fold'],evaluation=item['evaluation'],
                           selected_frequency_indices_0based=item['frequency_indices_0based'],selected_frequencies_hz=item['frequencies_hz'],
                           n_frequencies=len(item['frequencies_hz']),selected_pattern_indices_0based=item['pattern_indices_0based']))
        timing={key:item[key] for key in ['stream','repeat','outer_fold','evaluation','input_features','eligible_test_sites',
                                                   'test_transform_seconds','test_predict_seconds']}
        timing.update(preparation_and_fit_seconds=item.get('preparation_and_fit_seconds',item['training_seconds']),
                      model_fit_seconds=item.get('model_fit_seconds',np.nan),
                      scope='cached full-rank feature preparation; test transform may overlap preparation span; not acquisition or end-to-end deployment')
        timings.append(timing)
        if dataset.patterns is not None:
            selected=dataset.patterns[item['pattern_indices_0based']]
            hardware.append(dict(stream=item['stream'],repeat=item['repeat'],outer_fold=item['outer_fold'],
                                 n_patterns=len(selected),n_current_electrodes=len(np.unique(selected[:,:2])),
                                 n_voltage_electrodes=len(np.unique(selected[:,2:])),
                                 n_drive_pairs=len(np.unique(np.sort(selected[:,:2],axis=1),axis=0)),
                                 input_features=item['input_features'],physical_reduction_validated=False))
    write_json(folder/'frequency_and_pattern_identities.json',{'schema_version':2,'scope':'outer-training refits; reporting only; never reused in evaluation','selections':actual})
    pd.DataFrame(timings).to_csv(folder/'measured_computation_times.csv',index=False)
    pd.DataFrame(hardware).to_csv(folder/'hardware_requirements.csv',index=False)
    # Pairwise Jaccard is a descriptive stability report; outer training sets overlap.
    stability=[]
    for stream,parts in itertools.groupby(sorted(actual,key=lambda r:r['stream']),key=lambda r:r['stream']):
        parts=list(parts)
        for a,b in itertools.combinations(parts,2):
            aa=set(a['selected_frequency_indices_0based']);bb=set(b['selected_frequency_indices_0based'])
            stability.append(dict(stream=stream,repeat_a=a['repeat'],fold_a=a['outer_fold'],repeat_b=b['repeat'],fold_b=b['outer_fold'],
                                  frequency_jaccard=len(aa&bb)/len(aa|bb) if aa|bb else np.nan))
    pd.DataFrame(stability).to_csv(folder/'ranking_stability.csv',index=False)
    if not getattr(export_search,'plots',True):
        return
    plt=_pyplot()
    for stage,xlabel,column,name in [('frequency','Number of frequencies','n_freq','frequency_reduction'),('mask','IIVV mask','mask','iivv_selection'),('quality','Maximum missing-pattern percentage','cutoff','quality_sensitivity')]:
        part=inner[inner.stage==stage]
        if part.empty:continue
        values=part.groupby(column).macro_auc.agg(['mean','min','max'])
        fig,ax=plt.subplots(figsize=(8,4));x=np.arange(len(values));ax.plot(x,values['mean'],'o-')
        ax.set(xticks=x,xticklabels=values.index,xlabel=xlabel,ylabel='Inner-selection macro AUC (descriptive)',title=f'{task}: {stage}')
        if stage=='mask':plt.setp(ax.get_xticklabels(),rotation=45,ha='right')
        fig.tight_layout();fig.savefig(folder/f'{name}.png',dpi=160);fig.savefig(folder/f'{name}.pdf');plt.close(fig)


def cluster_wald(table,outcome,columns):
    """Descriptive class/location association via patient-cluster sandwich OLS Wald test."""
    clean=table[['patient_id',outcome,*columns]].dropna()
    if clean.empty:return dict(status='unavailable',reason='no complete metadata rows')
    design=pd.get_dummies(clean[columns].astype(str),drop_first=True,dtype=float)
    x=np.column_stack([np.ones(len(clean)),design.to_numpy()]);y=clean[outcome].to_numpy(float)
    n,p=x.shape;g=clean.patient_id.nunique()
    if g <= p+1 or np.linalg.matrix_rank(x)<p or p<2:
        return dict(status='unavailable',reason='insufficient independent patients or singular design',n_patients=g)
    inv=np.linalg.pinv(x.T@x);beta=inv@x.T@y;residual=y-x@beta;meat=np.zeros((p,p))
    for _,ix in clean.groupby('patient_id').indices.items():
        u=x[ix].T@residual[ix];meat+=np.outer(u,u)
    covariance=inv@meat@inv*(g/(g-1))*((n-1)/(n-p))
    cov=covariance[1:,1:]
    cov=(cov+cov.T)/2
    eigenvalues,eigenvectors=np.linalg.eigh(cov)
    tolerance=max(float(np.max(np.abs(eigenvalues)))*1e-10,1e-14)
    # Sparse location cells can give singular cluster covariance even with a
    # full-rank design. A pseudoinverse must not turn floating-point negative
    # eigenvalues into a negative Wald statistic or a spurious p=1 result.
    if np.any(eigenvalues <= tolerance):
        return dict(status='unavailable',reason='singular or ill-conditioned patient-cluster covariance; sparse factor levels',
                    n_patients=g,n_coefficients=p-1,
                    covariance_rank=int((eigenvalues>tolerance).sum()))
    projected=eigenvectors.T@beta[1:]
    statistic=float(np.sum(projected**2/eigenvalues))
    rank=len(eigenvalues)
    return dict(status='computed',wald_chi2=statistic,df=rank,p_value=float(chi2.sf(statistic,rank)),n_patients=g,
                method='asymptotic patient-cluster sandwich OLS Wald; descriptive association, not causality')


def quality_audit(dataset,config,outdir):
    folder=Path(outdir)/'quality';folder.mkdir(exist_ok=True)
    table=dataset.table.copy();table.to_csv(folder/'site_inclusion_audit.csv',index=False)
    table.groupby('patient_id').size().rename('n_sites').to_csv(folder/'sites_per_patient.csv')
    table.groupby('pathology',dropna=False).missing_percent.agg(['count','mean','std','min','median','max']).to_csv(folder/'missing_patterns_by_class.csv')
    if 'location' in table:
        pd.crosstab(table.pathology,table.location,dropna=False).to_csv(folder/'class_by_location.csv')
        table.groupby(['pathology','location'],dropna=False).missing_percent.agg(['count','mean','std']).to_csv(folder/'missing_patterns_by_class_location.csv')
    rows=[]
    for cutoff in sorted(set(config['cutoffs']+[config['default_cutoff']])):
        for label,part in table.groupby('pathology',dropna=False):
            keep=part.missing_percent<cutoff
            rows.append(dict(cutoff=cutoff,class_name=label,n_total_sites=len(part),n_retained_sites=int(keep.sum()),
                             n_total_patients=part.patient_id.nunique(),n_retained_patients=part.loc[keep].patient_id.nunique()))
    pd.DataFrame(rows).to_csv(folder/'retention_by_threshold_class.csv',index=False)
    mismatch=np.abs(table.missing_percent-table.observed_missing_pattern_percent)
    write_json(folder/'missingness_encoding_audit.json',dict(max_reported_vs_observed_difference=float(mismatch.max()),
               n_sites_disagreeing_by_more_than_one_percentage_point=int((mismatch>1).sum()),
               note='Observed missing patterns have any non-finite frequency. Avg_Cutpats may be averaged over repeated acquisitions; discrepancies need upstream audit.'))
    tests=[]
    for factors in [['pathology'],['location'],['pathology','location']]:
        if all(c in table for c in factors):
            tests.append(dict(factors='+'.join(factors),**cluster_wald(table,'missing_percent',factors)))
    frame=pd.DataFrame(tests)
    if 'p_value' in frame:frame['p_holm']=holm(frame.p_value)
    frame.to_csv(folder/'quality_cluster_associations.csv',index=False)
    available=[v for v in config['clinical_covariates'] if v in table and table[v].notna().any()]
    covariates=[]
    for col in available:
        if pd.api.types.is_numeric_dtype(table[col]):
            for label,g in table.groupby('pathology',dropna=False):
                covariates.append(dict(variable=col,class_name=label,n=g[col].count(),mean=g[col].mean(),std=g[col].std()))
        else:
            pd.crosstab(table.pathology,table[col],dropna=False).to_csv(folder/f'covariate_{col}_counts.csv')
    pd.DataFrame(covariates).to_csv(folder/'numeric_covariates_by_class.csv',index=False)
    covtests=[]
    for col in available:
        if pd.api.types.is_numeric_dtype(table[col]):
            covtests.append(dict(variable=col,**cluster_wald(table,col,['pathology'])))
        else:
            for level in table[col].dropna().unique():
                indicator=table.copy()
                indicator['_indicator']=np.where(table[col].notna(),(table[col]==level).astype(float),np.nan)
                covtests.append(dict(variable=col,level=str(level),**cluster_wald(indicator,'_indicator',['pathology'])))
    ct=pd.DataFrame(covtests)
    if 'p_value' in ct:ct['p_holm']=holm(ct.p_value)
    ct.to_csv(folder/'covariate_cluster_associations.csv',index=False)
    write_json(folder/'clinical_metadata_status.json',dict(available=available,missing=[v for v in config['clinical_covariates'] if v not in available],
                pathology_registration='requires clinical records',mixed_histology_rule='requires clinical records',
                pathology_blinding='requires clinical records',contact_conditions='analyzed only if supplied; no controls assumed'))
    return available


def icc_absolute(matrix):
    """ICC(2,1): two-way random-effects absolute agreement on a balanced finite matrix."""
    x=np.asarray(matrix);x=x[np.isfinite(x).all(1)]
    n,k=x.shape
    if n<2 or k<2:return np.nan
    grand=x.mean();rows=x.mean(1);cols=x.mean(0)
    msr=k*np.sum((rows-grand)**2)/(n-1);msc=n*np.sum((cols-grand)**2)/(k-1)
    mse=np.sum((x-rows[:,None]-cols[None,:]+grand)**2)/((n-1)*(k-1))
    denominator=msr+(k-1)*mse+k*(msc-mse)/n
    return float((msr-mse)/denominator) if denominator else np.nan


def repeatability_audit(dataset,outdir,n_boot=1000,seed=0):
    folder=Path(outdir)/'repeatability';folder.mkdir(exist_ok=True)
    if dataset.repeats is None:
        write_json(folder/'status.json',dict(status='unavailable',reason='Zrepeats unaveraged recordings not supplied; averaging is not repeatability evidence'))
        return
    magnitude=np.abs(dataset.repeats)
    spectra=finite_mean(magnitude,axis=2) # site,freq,repeat
    values=finite_mean(spectra,axis=1)   # site,repeat: clearly defined scalar endpoint
    rows=[]
    for i,row in dataset.table.iterrows():
        mean=finite_mean(magnitude[i],axis=2)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            std=np.nanstd(magnitude[i],axis=2,ddof=1)
        cv=np.divide(std,mean,out=np.full_like(std,np.nan),where=mean>1e-12)
        item=dict(sample_id=row.sample_id,patient_id=row.patient_id,pathology=row.pathology,
                  mean_coordinate_cv=finite_mean(cv),mean_magnitude=finite_mean(values[i]),
                  scalar_repeat_cv=np.nanstd(values[i],ddof=1)/finite_mean(values[i]),
                  n_finite_test_means=int(np.isfinite(values[i]).sum()),
                  repositioning_sd=float(np.nanstd(values[i],ddof=1)))
        if magnitude.shape[-1]==9:
            acquisitions=values[i].reshape(3,3)
            item.update(within_acquisition_sd=float(np.nanmean(np.nanstd(acquisitions,axis=1,ddof=1))),
                        repositioning_sd=float(np.nanstd(np.nanmean(acquisitions,axis=1),ddof=1)))
        rows.append(item)
    pd.DataFrame(rows).to_csv(folder/'site_repeatability.csv',index=False)
    # Use the first three test means for a balanced primary ICC; sites lacking
    # any of those means are explicitly excluded from that endpoint only.
    balanced=values[:,:min(3,values.shape[1])]
    result=[]
    for label in [None,*dataset.table.pathology.dropna().unique()]:
        keep=np.ones(len(values),bool) if label is None else (dataset.table.pathology==label).to_numpy()
        from .metrics import cluster_indices
        rng=np.random.default_rng(seed);draws=[]
        groups=dataset.table.loc[keep,'patient_id'].to_numpy()
        for _ in range(n_boot):
            ix,_=cluster_indices(groups,rng)
            draws.append(icc_absolute(balanced[keep][ix]))
        draws=np.asarray(draws);draws=draws[np.isfinite(draws)]
        result.append(dict(class_name='All' if label is None else label,n_sites=int(keep.sum()),
                           icc_2_1_absolute=icc_absolute(balanced[keep]),
                           balanced_icc_sites=int(np.isfinite(balanced[keep]).all(axis=1).sum()),
                           balanced_icc_replicates=balanced.shape[1],
                           icc_ci_low=float(np.quantile(draws,.025)) if len(draws) else np.nan,
                           icc_ci_high=float(np.quantile(draws,.975)) if len(draws) else np.nan,
                           valid_cluster_bootstraps=len(draws),
                           between_site_sd=float(np.nanstd(finite_mean(values[keep],1),ddof=1))))
    pd.DataFrame(result).to_csv(folder/'icc_and_inter_site_variability.csv',index=False)
    write_json(folder/'status.json',dict(status='computed',icc_endpoint='mean magnitude over available frequencies and patterns per acquisition replicate',
              icc_model='ICC(2,1), absolute agreement; first three available test positions, complete sites; patient-cluster intervals',
              n_replicates=dataset.repeats.shape[-1],
              caveat='Only supplied replicate units are analyzed. The two to four MATLAB Full_data spectra are test means, not nine bursts; within-burst repeatability is unavailable. Multiple sites within patients are clustered for intervals.'))
