#!/usr/bin/env python3
"""Run nested EIS revision analyses, or validate prerequisites without fitting."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

# Writable cache and noninteractive plotting before importing the scientific stack.
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'iivv-matplotlib'))
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
import pandas as pd
from revision.data import load_dataset, digest, CLASS_NAMES
from revision.pipeline import write_json, run_task
from revision.analysis import quality_audit, repeatability_audit, report_task
from revision.metrics import holm


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare-source',action='store_true',help='Verify/extract optional MATLAB source metadata and test means before fitting')
    p.add_argument('--source-mat',default='source/Python-Labeled-CaseData-17-Sep-2025.mat')
    p.add_argument('--source-output',default='results/revision-source')
    p.add_argument('--data',default='data',help='One NPZ per site, preserving complex values and NaNs')
    p.add_argument('--patterns',default='source/IIVV_patterns.txt',help='Verified ordered IIVV coordinates')
    p.add_argument('--metadata',help='Optional site-keyed clinical CSV (patient_id,site_id)')
    p.add_argument('--repeats',help='Optional memory-mapped raw or test-mean repeat array')
    p.add_argument('--repeat-index',help='CSV with explicit repeat-array sample_id row order')
    p.add_argument('--label-map',help='Optional explicit raw-pathology-to-canonical-class JSON')
    p.add_argument('--config',default=str(Path(__file__).parent/'configs/revision.json'))
    p.add_argument('--output',default='results/revision',help='Dedicated new output directory; existing result plots stay untouched')
    p.add_argument('--tasks',nargs='+',choices=list(CLASS_NAMES))
    p.add_argument('--models',nargs='+',choices=['svm','rf','lr'])
    p.add_argument('--objectives',nargs='+',choices=['auc','acc','f1','sensitivity'])
    p.add_argument('--outer-repeats',type=int)
    p.add_argument('--outer-folds',type=int)
    p.add_argument('--inner-folds',type=int)
    p.add_argument('--bootstrap-samples',type=int)
    p.add_argument('--resume',action='store_true',help='Resume fully checkpointed folds with unchanged configuration/input hashes; source versions are archived')
    p.add_argument('--preflight',action='store_true',help='Validate data/config and write missing-input status without fitting')
    p.add_argument('--no-plots',action='store_true',help='Skip figure creation (all numeric reports are still computed)')
    p.add_argument('--trust-object-arrays',action='store_true',help='Allow pickle only for locally trusted historical NPZ object arrays')
    p.add_argument('--synthetic',action='store_true',help='Generate an explicitly labeled smoke-test fixture; NEVER clinical results')
    return p.parse_args(argv)


def synthetic_fixture(folder):
    """Deterministic integration fixture with repeated sites and deliberately missing data."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(311);frequencies=np.geomspace(100,100000,8)
    labels=['healthy','invasive carcinoma','moderate dysplasia','benign']
    # Synthetic coordinates have a declared order; not a reconstruction of the study.
    patterns=np.array([[26,27,1,2],[26,28,1,3],[27,29,2,4],[28,30,3,5],[29,31,4,6],[30,32,5,7]])
    np.savetxt(folder/'patterns.txt',patterns,fmt='%d')
    for patient in range(32):
        for site in range(2):
            k=patient%4
            z=(200+60*k+20*site)/(1+(1j*frequencies[:,None]/(1000+200*k))**.8)
            z=np.broadcast_to(z,(8,6)).copy()+rng.normal(0,8,(8,6))+1j*rng.normal(0,3,(8,6))
            missing=np.zeros(6,bool)
            if patient%5==0:missing[0]=True
            z[:,missing]=np.nan
            repeats=z[...,None]*(1+rng.normal(0,.02,(8,6,9)))
            np.savez(folder/f'sample_{patient:03d}_{site}.npz',Zdata=z,patient_no=str(patient),patient_sample=str(site),
                     LesionAssessment=labels[k],Avg_Cutpats=100*missing.mean(),frequencies_hz=frequencies,
                     LesionLoc=['tongue','cheek'][site],age=40+patient,sex=['female','male'][patient%2],Zrepeats=repeats)
    return folder,folder/'patterns.txt'


def runtime_manifest(args,config):
    versions={}
    for name in ['numpy','scipy','scikit-learn','pandas','matplotlib','Pillow','joblib']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    root=Path(__file__).resolve().parent
    source_paths=[Path(__file__),*sorted((root/'revision').glob('*.py')),*sorted((root/'scripts').glob('*.py')),root/'toolbox/masking.py']
    try:git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError,FileNotFoundError):git_revision=None
    return dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),arguments=vars(args),config=config,
                synthetic=args.synthetic,python=sys.version,platform=platform.platform(),processor=platform.processor(),
                cpu_count=os.cpu_count(),versions=versions,git_revision=git_revision,
                source_hashes={str(p.relative_to(root)):digest(p) for p in source_paths},
                notes=['CPU scikit-learn is the reference implementation; no GPU calibration differences.',
                       'Inner searches select configurations; outer outcomes are used only for assessment.',
                       'Patient bootstrap/permutation inference is conditional on fitted outer predictions, not full model refitting.',
                       'Clinical sensitivity/specificity targets are analysis assumptions, not validated safety requirements.'])


def validate_config(config):
    if config.get('schema_version')!=2:raise ValueError('Expected revision config schema_version 2')
    for key in ['outer_folds','inner_folds']:
        if config[key]<2:raise ValueError(f'{key} must be >=2')
    if int(config.get('task_workers',1))<1:raise ValueError('task_workers must be positive')
    for key in ['outer_repeats','bootstrap_samples','permutation_samples','top_k']:
        if config[key]<1:raise ValueError(f'{key} must be >=1')
    for key in ['tasks','models','objectives','masks','cutoffs','selectors','representations']:
        if not config[key]:raise ValueError(f'{key} cannot be empty')
    if any(x<0 or x>100.000001 for x in config['cutoffs']):raise ValueError('Cutoffs must be in [0,100.000001]')
    if not 0<config['sensitivity_target']<=1:raise ValueError('sensitivity_target must be in (0,1]')


_WORKER_DATASET=None


def _fit_report_task(task,config,output):
    predictions,selections=run_task(_WORKER_DATASET,task,config,output)
    return report_task(_WORKER_DATASET,task,predictions,selections,config,output)


def main(argv=None):
    args=parse_args(argv)
    config=json.loads(Path(args.config).read_text())
    for key in ['tasks','models','objectives','outer_repeats','outer_folds','inner_folds','bootstrap_samples']:
        value=getattr(args,key,None)
        if value is not None:config[key]=value
    config['plots']=not args.no_plots
    output=Path(args.output)
    if args.synthetic and output==Path('results/revision'):output=Path('results/revision-synthetic')
    if not args.synthetic and 'synthetic' in output.name.lower():
        raise ValueError('Use a separate output folder for clinical and synthetic runs')
    output.mkdir(parents=True,exist_ok=True)
    # Refuse to overwrite prior results or mix incompatible configurations/synthetic data.
    previous_manifest=None
    if (output/'run_manifest.json').exists() and not args.preflight:
        old=json.loads((output/'run_manifest.json').read_text())
        if args.resume:
            if old.get('status') not in ['interrupted','failed']:
                raise ValueError('Resume requires an explicitly interrupted/failed run, never an active or completed run')
            if old.get('synthetic')!=args.synthetic:
                raise ValueError('Resume requires identical synthetic flag')
            previous_manifest=old
        elif old.get('status') not in ['preflight_failed','preflight_passed']:
            raise ValueError(f'{output} already contains a run. Choose a new --output directory, or --resume an interrupted run.')
    elif args.resume:
        raise ValueError('Resume requires an existing run manifest')
    if args.synthetic:
        args.data,args.patterns=map(str,((output/'synthetic-input',output/'synthetic-input/patterns.txt') if args.resume else synthetic_fixture(output/'synthetic-input')))
        # A small complete integration path; full clinical search stays in the checked-in config.
        config.update(outer_folds=3,inner_folds=2,outer_repeats=1,top_k=1,frequency_counts=[1,3,8],
                      models=args.models or ['lr'],objectives=args.objectives or ['auc','sensitivity'],
                      masks=['All'],z_quantiles=[.5],selectors=['pca','anova'],representations=['magnitude','real_imag','ratio'],
                      cutoffs=[60,100.000001],normalizations=['freq'],imputations=['mean'],class_weights=[None],
                      bootstrap_samples=args.bootstrap_samples or 10,permutation_samples=10,
                      model_grids={'lr':{'C':[.1,1.]},'svm':{'C':[.1,1.],'kernel':['linear','rbf']},
                                   'rf':{'n_estimators':[20],'max_depth':[4],'max_features':['sqrt']}})
    validate_config(config)
    if previous_manifest is not None and previous_manifest.get('config')!=config:
        raise ValueError('Resume requires the identical resolved configuration')
    manifest=runtime_manifest(args,config)
    if previous_manifest is not None:
        parent_file=output/f'resume-parent-{len(list(output.glob("resume-parent-*.json")))+1:03d}.json'
        write_json(parent_file,previous_manifest)
        manifest['resume_history']=[*previous_manifest.get('resume_history',[]),dict(manifest=parent_file.name,sha256=digest(parent_file),
            note='Only fully checkpointed outer folds reused; configuration and input hashes must match. Source versions may differ and are recorded explicitly.')]
    # Archive exact scientific source for every execution, including resumed versions.
    archive=output/('source-at-start' if previous_manifest is None else f'source-at-resume-{len(manifest["resume_history"]):03d}')
    import shutil
    for name in manifest['source_hashes']:
        source=Path(__file__).resolve().parent/name;target=archive/name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    shutil.copytree(Path(__file__).resolve().parent/'configs',archive/'configs',dirs_exist_ok=True)
    write_json(output/'run_manifest.json',manifest);write_json(output/'resolved_config.json',config)
    try:
        if args.prepare_source and not args.synthetic and Path(args.source_mat).is_file():
            source_dir=Path(args.source_output)
            source_manifest=source_dir/'source_extraction.json'
            verified=False
            if source_manifest.is_file():
                previous=json.loads(source_manifest.read_text())
                hashes={str(path):digest(path) for path in sorted(Path(args.data).glob('*.npz'))}
                verified=(previous.get('mat_sha256')==digest(args.source_mat) and previous.get('npz_hashes')==hashes)
            if not verified:
                subprocess.run([sys.executable,str(Path(__file__).parent/'scripts/prepare_revision_source.py'),
                                '--mat',args.source_mat,'--data',args.data,'--output',args.source_output],check=True)
            args.metadata=args.metadata or str(source_dir/'clinical_metadata.csv')
            args.repeats=args.repeats or str(source_dir/'test_mean_repeats.npy')
            args.repeat_index=args.repeat_index or str(source_dir/'repeat_index.csv')
        if args.prepare_source and not args.synthetic and Path(args.source_mat).is_file():
            from scripts.audit_source_quality import audit
            audit(args.source_mat,args.data,args.source_output,output/'quality')
        needs_patterns=any(m!='All' for m in config['masks'])
        # All-only analyses can run without geometry; full default search requires the verified coordinate file.
        patterns=args.patterns if needs_patterns or Path(args.patterns).is_file() else None
        dataset=load_dataset(args.data,patterns_path=patterns,metadata_path=args.metadata,
                             label_map_path=args.label_map,frequencies=config.get('frequencies_hz'),
                             trust_object_arrays=args.trust_object_arrays,repeats_path=args.repeats,repeats_index_path=args.repeat_index)
        manifest['arguments']=vars(args)
        manifest['input_hashes']=dataset.input_hashes
        if previous_manifest is not None and previous_manifest.get('input_hashes')!=dataset.input_hashes:
            raise ValueError('Input hashes changed; cannot reuse any old outer-fold results')
        inventory=sorted((d.metadata.get('Name','unknown'),d.version) for d in importlib.metadata.distributions())
        (output/'environment-freeze.txt').write_text('\n'.join(f'{name}=={version}' for name,version in inventory)+'\n')
        manifest['input_summary']=dict(n_sites=len(dataset.table),n_patients=dataset.table.patient_id.nunique(),
                                      n_frequencies=len(dataset.frequencies),n_patterns=dataset.z.shape[2],
                                      complex_available=dataset.complex_available,unmapped_pathologies=dataset.table.loc[dataset.table.pathology.isna(),'raw_pathology'].value_counts().to_dict())
        # Validate all requested target populations before launching costly fits.
        from revision.pipeline import group_splits
        for task in config['tasks']:
            ix,y,names=dataset.task(task)
            if len(np.unique(y))!=len(names):raise ValueError(f'{task}: absent target classes; expected {names}')
            taskgroups=dataset.table.iloc[ix].patient_id.to_numpy()
            if config.get('external_cohort_column'):
                col=config['external_cohort_column']
                if col not in dataset.table:raise ValueError(f'External cohort column {col} is absent')
                ext=dataset.table.iloc[ix][col].astype(str).isin(config['external_cohort_values']).to_numpy()
                if set(taskgroups[ext]) & set(taskgroups[~ext]):raise ValueError('Development and external patients overlap')
                y=y[~ext];taskgroups=taskgroups[~ext]
            splits=group_splits(y,taskgroups,config['outer_folds'],config['seed'])
            for tr,_ in splits:
                group_splits(y[tr],taskgroups[tr],config['inner_folds'],config['seed']+1)
        quality_audit(dataset,config,output);repeatability_audit(dataset,output,config['bootstrap_samples'],config['seed'])
        statuses={
            'complex_feature_comparisons': 'available' if dataset.complex_available else 'unavailable: original arrays are real-valued',
            'repeatability':'available' if dataset.repeats is not None else 'unavailable: Zrepeats absent',
            'external_validation':'available' if config.get('external_cohort_column') else 'not configured: no independent cohort assumed',
            'pathology_registration_blinding_mixed_sites':'requires clinical records; no procedure inferred',
            'prospective_acquisition_and_hardware_validation':'requires new measurements; only computation and coordinate requirements can be measured here',
            'requested_representations':config['representations'],
        }
        write_json(output/'analysis_status.json',statuses)
        if args.preflight:
            manifest['status']='preflight_passed';write_json(output/'run_manifest.json',manifest)
            print(f'Preflight passed: {output}/run_manifest.json');return 0
        all_summaries=[];all_comparisons=[]
        start=time.perf_counter()
        execution_config={**config,'_resume':args.resume,
                          '_source_version':dict(started_utc=manifest['started_utc'],hashes=manifest['source_hashes']),
                          '_previous_source_version':dict(started_utc=previous_manifest['started_utc'],hashes=previous_manifest['source_hashes']) if previous_manifest else None}
        workers=min(int(config.get('task_workers',1)),len(config['tasks']))
        if workers>1:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor
            if 'fork' not in multiprocessing.get_all_start_methods():
                raise ValueError('task_workers>1 currently requires fork support; use task_workers=1 on this platform')
            # Raw immutable arrays are inherited read-only; only small task/config
            # arguments cross the pool. Each task has distinct folds/output files,
            # independent caches, identical prespecified seeds and no test reuse.
            global _WORKER_DATASET
            _WORKER_DATASET=dataset
            dataset.z.flags.writeable=False
            print(f'Running {workers} independent diagnostic tasks concurrently',flush=True)
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('fork')) as pool:
                futures=[pool.submit(_fit_report_task,task,execution_config,output) for task in config['tasks']]
                for future in futures:
                    summary,comparison=future.result()
                    all_summaries.append(summary);all_comparisons.append(comparison)
        else:
            for task in config['tasks']:
                predictions,selections=run_task(dataset,task,execution_config,output)
                summary,comparison=report_task(dataset,task,predictions,selections,config,output)
                all_summaries.append(summary);all_comparisons.append(comparison)
        summary=pd.concat(all_summaries,ignore_index=True);summary.to_csv(output/'all_metrics.csv',index=False)
        # Familiar table names with explicit nested evaluation provenance and class semantics.
        summary[summary.stream.str.startswith('baseline_')].to_csv(output/'BaselineResults.csv',index=False)
        summary[~summary.stream.str.startswith('baseline_')].to_csv(output/'FinalResults.csv',index=False)
        comparisons=pd.concat(all_comparisons,ignore_index=True)
        if len(comparisons):
            comparisons['p_holm_all_tasks_models_objectives']=holm(comparisons.p_value)
        comparisons.to_csv(output/'paired_comparisons_holm.csv',index=False)
        failures=[]
        for task in config['tasks']:
            failures.extend(json.loads((output/task/'candidate_failures.json').read_text()))
        statuses['candidate_failure_count']=len(failures)
        statuses['candidate_failures_artifacts']=[f'{task}/candidate_failures.json' for task in config['tasks']]
        statuses['completed_task_populations']=config['tasks']
        statuses['real_patient_reanalysis_completed']=not args.synthetic
        write_json(output/'analysis_status.json',statuses)
        manifest.update(status='completed',elapsed_seconds=time.perf_counter()-start,
                        finished_utc=datetime.now(timezone.utc).isoformat(),
                        claim_scope='SYNTHETIC SOFTWARE VALIDATION ONLY' if args.synthetic else 'Retrospective nested validation; conditional uncertainty, not prospective clinical validation')
        write_json(output/'run_manifest.json',manifest)
        print(f'Completed {len(config["tasks"])} tasks: {output}');return 0
    except KeyboardInterrupt:
        manifest.update(status='interrupted',error='Interrupted before completion; checkpoint files are not final results')
        write_json(output/'run_manifest.json',manifest)
        return 130
    except (ValueError,FileNotFoundError,RuntimeError,subprocess.CalledProcessError) as exc:
        manifest.update(status='preflight_failed' if args.preflight else 'failed',error=str(exc))
        write_json(output/'run_manifest.json',manifest)
        print(f'Revision workflow stopped: {exc}',file=sys.stderr)
        return 2


if __name__=='__main__':
    sys.exit(main())
