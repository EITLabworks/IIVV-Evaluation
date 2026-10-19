"""One authoritative outer patient loop: searches, scaling and calibration are nested."""
from dataclasses import replace
from pathlib import Path
import itertools
import json
import time
import traceback
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold
from .features import Candidate, FeatureBuilder, aligned_probabilities, make_model, patient_weights
from .metrics import scores, threshold_for_sensitivity, select_threshold


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def group_splits(y, groups, folds, seed):
    groups, y = np.asarray(groups), np.asarray(y)
    nclasses = len(np.unique(y))
    if folds < 2 or len(np.unique(groups)) < folds:
        raise ValueError(f'{folds} folds need at least {folds} distinct patients')
    for k in np.unique(y):
        if len(np.unique(groups[y == k])) < 2:
            raise ValueError(f'Class {k} appears in fewer than two patients; grouped validation cannot be fitted')
    # Retry deterministic stratifications if one training partition loses a class.
    for attempt in range(100):
        cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed + attempt)
        splits = list(cv.split(np.zeros(len(y)), y, groups))
        if all(len(np.unique(y[tr])) == nclasses for tr, te in splits):
            for tr, te in splits:
                assert not set(groups[tr]).intersection(groups[te])
            return splits
    raise ValueError('No patient-disjoint split with all target classes in each training fold; change fold count or obtain more patients')


def fit_predict(dataset, indices, y, candidate, train, test, seed, config):
    """Only training indices enter fit; quality failures in test are explicit abstentions."""
    groups = dataset.table.iloc[indices].patient_id.to_numpy()
    quality = dataset.table.iloc[indices].missing_percent.to_numpy(float)
    tr = np.asarray(train)[quality[train] < candidate.cutoff]
    te = np.asarray(test)
    valid_test = quality[te] < candidate.cutoff
    if set(y[tr]) != set(np.unique(y)):
        raise ValueError('Completeness cutoff removes a class from the training partition')
    nclasses = len(np.unique(y))
    # Cache train-fitted feature matrices within this OUTER fold only. The key
    # includes exact training/test site identities; no fold can share fitted statistics.
    from collections import OrderedDict
    import copy
    cache = config.setdefault('_feature_cache', OrderedDict())
    feature_key = (tuple(indices[tr]), tuple(indices[te]), tuple(valid_test), seed, candidate.mask, candidate.selector,
                   candidate.representation, candidate.normalization, candidate.imputation)
    cached = cache.pop(feature_key, None)
    start = time.perf_counter()
    if cached is None:
        full_candidate = replace(candidate, n_freq=0)
        full_builder = FeatureBuilder(full_candidate, dataset.frequencies, dataset.patterns, seed,
                                      config.get('max_pca_rows',100000))
        full_builder.fit(dataset.z[indices[tr]], y[tr], groups[tr])
        full_train = full_builder.transform(dataset.z[indices[tr]])
        before_test = time.perf_counter()
        full_test = full_builder.transform(dataset.z[indices[te[valid_test]]]) if valid_test.any() else None
        test_transform_time = time.perf_counter()-before_test
        cached = (full_builder, full_train, full_test, test_transform_time)
    full_builder, full_train, full_test, test_transform_time = cached
    cache[feature_key] = cached
    budget = config.get('feature_cache_bytes', 512*1024*1024)
    def cache_bytes():
        return sum(v[1].nbytes+(v[2].nbytes if v[2] is not None else 0) for v in cache.values())
    while cache and cache_bytes() > budget:
        cache.popitem(last=False)
    builder = copy.copy(full_builder)
    builder.candidate = candidate
    n = candidate.n_freq or len(full_builder.ranking_)
    builder.selected_ = full_builder.ranking_[:n]
    count = full_train.shape[1] if candidate.representation=='circuit' else full_train.shape[1]//len(full_builder.ranking_)*len(builder.selected_)
    xtrain = full_train[:,:count]
    clinical = None
    clinical_columns = [v for v in config['clinical_covariates'] if v in dataset.table and dataset.table.iloc[indices[tr]][v].notna().any()]
    if candidate.stage in {'covariate_only', 'covariate_combined'}:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        if not clinical_columns:
            raise ValueError('No usable clinical covariates in training patients')
        table = dataset.table.iloc[indices[tr]][clinical_columns]
        numeric = [col for col in clinical_columns if __import__('pandas').api.types.is_numeric_dtype(table[col])]
        categorical = [col for col in clinical_columns if col not in numeric]
        clinical = ColumnTransformer([
            ('numeric', make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler()), numeric),
            ('categorical', make_pipeline(SimpleImputer(strategy='most_frequent', keep_empty_features=True), OneHotEncoder(handle_unknown='ignore', sparse_output=False)), categorical)
        ], sparse_threshold=0)
        clinical_x = clinical.fit_transform(table)
        xtrain = clinical_x if candidate.stage == 'covariate_only' else np.column_stack([xtrain, clinical_x])
    model = make_model(candidate, seed)
    weights = patient_weights(groups[tr]) if candidate.patient_weighted else np.ones(len(tr))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model_start=time.perf_counter()
        model.fit(xtrain, y[tr], sample_weight=weights)
        model_fit_seconds=time.perf_counter()-model_start
    fitted_seconds = time.perf_counter() - start
    p = np.full((len(te), nclasses), np.nan)
    transform_seconds = predict_seconds = 0.
    if valid_test.any():
        start = time.perf_counter()
        xtest = full_test[:,:count]
        if clinical is not None:
            clinical_test = clinical.transform(dataset.table.iloc[indices[te[valid_test]]][clinical_columns])
            xtest = clinical_test if candidate.stage == 'covariate_only' else np.column_stack([xtest, clinical_test])
        transform_seconds = test_transform_time + time.perf_counter() - start
        start = time.perf_counter()
        p[valid_test] = aligned_probabilities(model, xtest, nclasses)
        predict_seconds = time.perf_counter() - start
    meta = builder.metadata()
    meta.update(train_patients=sorted(set(groups[tr])), eligible_train_sites=len(tr),
                test_patients=sorted(set(groups[te])), eligible_test_sites=int(valid_test.sum()),
                input_features=xtrain.shape[1], training_seconds=fitted_seconds,
                test_transform_seconds=transform_seconds, test_predict_seconds=predict_seconds,
                convergence_warnings=[str(w.message) for w in caught])
    meta['solver_training_dimensions'] = getattr(model, 'optimized_dimensions_', xtrain.shape[1])
    meta['preparation_and_fit_seconds']=fitted_seconds
    meta['model_fit_seconds']=model_fit_seconds
    meta['timing_scope']='Cached full-ranked feature preparation; preparation includes test transform on a cache miss; no acquisition timing'
    meta['clinical_covariates'] = clinical_columns if clinical is not None else []
    return p, meta, (builder, model, clinical)


def inner_evaluate(dataset, indices, y, candidate, splits, seed, config):
    predictions = np.full((len(indices), len(np.unique(y))), np.nan)
    metadata = []
    for fold, (tr, va) in enumerate(splits):
        p, meta, _ = fit_predict(dataset, indices, y, candidate, tr, va, seed + fold, config)
        predictions[va] = p
        metadata.append({key:meta[key] for key in ('train_patients','test_patients','frequency_indices_0based','input_features')})
    covered = np.isfinite(predictions).all(axis=1)
    if not covered.any() or len(np.unique(y[covered])) != len(np.unique(y)):
        raise ValueError('Candidate has no evaluable predictions for at least one target class')
    groups = dataset.table.iloc[indices].patient_id.to_numpy()
    # Equal-patient selection weights are calculated BEFORE removing abstentions.
    all_weights = patient_weights(groups)
    w = all_weights[covered]
    result, _, _ = scores(y[covered], predictions[covered], weights=w)
    coverage = w.sum() / all_weights.sum()
    values = dict(auc=result['macro_auc'], acc=result['accuracy'], f1=result['macro_f1'])
    if predictions.shape[1] == 2:
        threshold = threshold_for_sensitivity(y[covered], predictions[covered, 1],
                                             config['sensitivity_target'], w)
        threshold_pred = (predictions[covered, 1] >= threshold).astype(int)
        sensitivity_metrics = scores(y[covered], predictions[covered], threshold_pred, w)[0]
        values['sensitivity'] = sensitivity_metrics['specificity']
    else:
        # A sensitivity objective in multiclass optimizes dysplasia recall, NOT micro recall.
        dysplasia_index = list(config['_class_names']).index('High-grade dysplasia')
        _, cls, _ = scores(y[covered], predictions[covered], weights=w)
        values['sensitivity'] = cls.iloc[dysplasia_index].sensitivity
    # Abstentions cannot make a restrictive quality cutoff appear better for free.
    objective = {name: float(value * coverage) for name, value in values.items()}
    return dict(candidate=candidate, objectives=objective, metrics=result, coverage=coverage,
                predictions=predictions, metadata=metadata)


def screen_candidates(config, nfreq, complex_available):
    defaults = dict(cutoff=config['default_cutoff'], patient_weighted=config['patient_weighted_training'])
    base = Candidate(**defaults)
    candidates = [replace(base, stage='frequency', n_freq=n) for n in config.get('frequency_counts', list(range(1, nfreq + 1))) if 1 <= n <= nfreq]
    candidates += [replace(base, stage='mask', mask=mask) for mask in config['masks']]
    candidates += [replace(base, stage='z_threshold', mask=f'Zq:{q}') for q in config['z_quantiles']]
    for selector in config['selectors']:
        candidates.append(replace(base, stage='selector', selector=selector, n_freq=min(config['comparison_frequencies'], nfreq)))
    for rep in config['representations']:
        if rep in {'real_imag', 'magnitude_phase', 'circuit'} and not complex_available:
            continue
        if rep == 'circuit' and nfreq < 6:
            continue
        candidates.append(replace(base, stage='representation', representation=rep,
                                  n_freq=0 if rep == 'circuit' else min(config['comparison_frequencies'], nfreq)))
    candidates += [replace(base, stage='quality', cutoff=t) for t in config['cutoffs']]
    candidates += [replace(base, stage='normalization', normalization=n) for n in config['normalizations']]
    candidates += [replace(base, stage='imputation', imputation=i) for i in config['imputations']]
    candidates += [replace(base, stage='weighting', class_weight=w) for w in config['class_weights']]
    candidates += [replace(base, stage='missingness', representation='missingness', normalization='none', selector='all')]
    return list(dict.fromkeys(candidates))


def tune_candidates(screened, config, objective, nfreq):
    def top(stage, field, fallback):
        ranked = sorted([v for v in screened if v['candidate'].stage == stage],
                        key=lambda v: v['objectives'][objective], reverse=True)
        vals = list(dict.fromkeys(getattr(v['candidate'], field) for v in ranked))
        return vals[:config['top_k']] or fallback
    frequencies = top('frequency', 'n_freq', [0]); masks = top('mask', 'mask', ['All'])
    # Allow threshold masks to compete with geometric masks in the joint stage.
    z = sorted([v for v in screened if v['candidate'].stage == 'z_threshold'],
               key=lambda v: v['objectives'][objective], reverse=True)
    if z:
        masks = list(dict.fromkeys(masks + [z[0]['candidate'].mask]))
    settings = dict(selector=top('selector', 'selector', ['pca'])[0],
                    representation=top('representation', 'representation', ['magnitude'])[0],
                    cutoff=top('quality', 'cutoff', [config['default_cutoff']])[0],
                    normalization=top('normalization', 'normalization', ['freq'])[0],
                    imputation=top('imputation', 'imputation', ['mean'])[0],
                    class_weight=top('weighting', 'class_weight', [None])[0],
                    patient_weighted=config['patient_weighted_training'])
    candidates = []
    for model, grid in config['model_grids'].items():
        if model not in config['models']:
            continue
        for n, mask, params in itertools.product(frequencies, masks, ParameterGrid(grid)):
            candidates.append(Candidate(model=model, n_freq=0 if settings['representation'] == 'circuit' else n,
                                        mask=mask, params_json=json.dumps(params, sort_keys=True), **settings))
    return list(dict.fromkeys(candidates))


class OOFCalibrator:
    """Fixed multinomial/binary sigmoid mapping fitted on outer-training OOF scores."""
    def __init__(self, enabled=True):
        self.enabled = enabled

    def fit(self, y, p, groups):
        if self.enabled:
            self.model = LogisticRegression(C=1., max_iter=3000, random_state=0).fit(
                np.log(np.clip(p, 1e-8, 1)), y, sample_weight=patient_weights(groups))
        return self

    def transform(self, p):
        return self.model.predict_proba(np.log(np.clip(p, 1e-8, 1))) if self.enabled else p.copy()


def run_task(dataset, task, config, outdir):
    indices, y, classes = dataset.task(task)
    if not len(indices):
        raise ValueError(f'{task}: no sites with accepted explicit pathology labels')
    config = {**config, '_class_names': list(classes)}
    groups = dataset.table.iloc[indices].patient_id.to_numpy()
    taskdir = Path(outdir) / task; taskdir.mkdir(parents=True, exist_ok=True)
    dataset.table.iloc[indices].assign(class_index=y).to_csv(taskdir / 'task_inclusion.csv', index=False)
    predictions, selections, splits_log, inner_rows, failures = [], [], [], [], []
    completed_folds=set()
    checkpoint_names=['splits.json','selections.json','candidate_failures.json','outer_predictions.csv','inner_search.csv']
    if config.get('_resume') and all((taskdir/name).is_file() for name in checkpoint_names):
        old_splits=json.loads((taskdir/'splits.json').read_text())
        old_selections=json.loads((taskdir/'selections.json').read_text())
        old_predictions=pd.read_csv(taskdir/'outer_predictions.csv',dtype={'patient_id':str,'sample_id':str})
        old_inner=pd.read_csv(taskdir/'inner_search.csv')
        def key(row):return (row['repeat'],row['outer_fold'],row['evaluation'])
        selection_keys={key(row) for row in old_selections}
        prediction_keys={key(row) for row in old_predictions.to_dict('records')}
        inner_keys={key(row) for row in old_inner.to_dict('records')}
        for row in old_splits:
            group=key(row)
            if group not in selection_keys&prediction_keys&inner_keys:continue
            pred=old_predictions[(old_predictions.repeat==group[0])&(old_predictions.outer_fold==group[1])&(old_predictions.evaluation==group[2])]
            selected=[item for item in old_selections if key(item)==group]
            expected=set(item['stream'] for item in selected)
            if set(pred.stream)!=expected:continue
            if not pred.groupby('stream').sample_id.apply(lambda col:set(col)==set(row['test_sample_ids']) and not col.duplicated().any()).all():continue
            completed_folds.add(group)
        splits_log=[row for row in old_splits if key(row) in completed_folds]
        for row in splits_log:row.setdefault('source_version',config.get('_previous_source_version'))
        selections=[row for row in old_selections if key(row) in completed_folds]
        predictions=[row for row in old_predictions.to_dict('records') if key(row) in completed_folds]
        inner_rows=[row for row in old_inner.to_dict('records') if key(row) in completed_folds]
        completed_names={f'{evaluation}_repeat{repeat}_fold{fold}' for repeat,fold,evaluation in completed_folds}
        failures=[row for row in json.loads((taskdir/'candidate_failures.json').read_text()) if row['fold'] in completed_names]
        print(f'{task}: reusing {len(completed_folds)} fully checkpointed outer folds; prior source versions retained',flush=True)
    # External cohorts are never used in folds, preprocessing, or candidate selection.
    external = np.zeros(len(indices), bool)
    if config.get('external_cohort_column'):
        col = config['external_cohort_column']
        if col not in dataset.table:
            raise ValueError(f'External cohort column {col} is absent')
        external = dataset.table.iloc[indices][col].astype(str).isin(config['external_cohort_values']).to_numpy()
        if set(groups[external]).intersection(groups[~external]):
            raise ValueError('A patient occurs in both development and external cohorts')
    dev = np.flatnonzero(~external)
    outer = []
    for repeat in range(config['outer_repeats']):
        for fold, (tr, te) in enumerate(group_splits(y[dev], groups[dev], config['outer_folds'], config['seed'] + repeat * 1000)):
            outer.append((repeat, fold, dev[tr], dev[te], 'nested_cv'))
    if external.any():
        outer.append((0, 0, dev, np.flatnonzero(external), 'external'))
    for repeat, fold, tr, te, evaluation in outer:
        fold_name = f'{evaluation}_repeat{repeat}_fold{fold}'
        if (repeat,fold,evaluation) in completed_folds:continue
        print(f'{task}: {fold_name}, training patients={len(set(groups[tr]))}, test={len(set(groups[te]))}', flush=True)
        assert not set(groups[tr]).intersection(groups[te])
        train_indices, train_y = indices[tr], y[tr]
        seed = config['seed'] + repeat * 1000 + fold * 100
        inner = group_splits(train_y, groups[tr], config['inner_folds'], seed + 1)
        split_record = dict(task=task, repeat=repeat, outer_fold=fold, evaluation=evaluation,
                            train_sample_ids=dataset.table.iloc[indices[tr]].sample_id.tolist(),
                            test_sample_ids=dataset.table.iloc[indices[te]].sample_id.tolist(),
                            train_patients=sorted(set(groups[tr])), test_patients=sorted(set(groups[te])),
                            inner=[dict(train_patients=sorted(set(groups[tr][a])), validation_patients=sorted(set(groups[tr][b]))) for a,b in inner])
        split_record['source_version']=config.get('_source_version')
        splits_log.append(split_record)
        cache = {}
        def evaluate(candidate):
            if candidate not in cache:
                started=time.perf_counter()
                print(f'{task}/{fold_name}: inner {len(cache)+1}: {candidate.stage}/{candidate.model}, frequencies={candidate.n_freq or "all"}, mask={candidate.mask}',flush=True)
                try:
                    value = inner_evaluate(dataset, train_indices, train_y, candidate, inner, seed, config)
                    value['id'] = len(cache)
                    inner_rows.append({**candidate.dictionary(), **value['metrics'], **{f'objective_{k}': v for k,v in value['objectives'].items()},
                                       'coverage': value['coverage'], 'candidate_id': value['id'], 'repeat': repeat, 'outer_fold': fold, 'evaluation': evaluation})
                    cache[candidate] = value
                except (ValueError, FloatingPointError) as exc:
                    failures.append(dict(task=task, fold=fold_name, candidate=candidate.dictionary(), reason=str(exc)))
                    cache[candidate] = None
                print(f'{task}/{fold_name}: inner completed in {time.perf_counter()-started:.2f}s',flush=True)
            return cache[candidate]
        screened = [r for c in screen_candidates(config, len(dataset.frequencies), dataset.complex_available) if (r := evaluate(c)) is not None]
        if not screened:
            raise RuntimeError(f'{task}/{fold_name}: every screening candidate failed; inspect input labels and quality')
        # Baselines receive the SAME inner calibration and operating-point treatment.
        streams = {}
        for model in config['models']:
            baseline = Candidate(stage='baseline', model=model, cutoff=config['default_cutoff'],
                                 patient_weighted=config['patient_weighted_training'], selector='all')
            result = evaluate(baseline)
            if result is None:
                raise RuntimeError(f'Baseline {model} failed: cannot perform a paired evaluation')
            streams[f'baseline_{model}'] = (result, 'baseline', model)
        # Prespecified screening-family winners get unbiased outer scoring too.
        for stage, field in [('selector','selector'), ('representation','representation'), ('quality','cutoff'),
                             ('normalization','normalization'), ('imputation','imputation'), ('weighting','class_weight')]:
            options = [v for v in screened if v['candidate'].stage == stage]
            for value in options:
                label = str(getattr(value['candidate'], field))
                streams[f'ablation_{stage}_{label}'] = (value, 'ablation', 'svm')
        for value in screened:
            if value['candidate'].stage == 'missingness':
                streams['ablation_missingness'] = (value, 'ablation', 'svm')
        for objective in config['objectives']:
            tuned = [r for c in tune_candidates(screened, config, objective, len(dataset.frequencies)) if (r := evaluate(c)) is not None]
            if not tuned:
                raise RuntimeError('Every tuning candidate failed')
            for model in config['models']:
                options = [r for r in tuned if r['candidate'].model == model]
                if not options:
                    raise RuntimeError(f'No evaluable tuning candidate for {model}')
                best = max(options, key=lambda r: r['objectives'][objective])
                streams[f'{objective}_{model}'] = (best, objective, model)
            best = max(tuned, key=lambda r: r['objectives'][objective])
            streams[f'{objective}_selected'] = (best, objective, best['candidate'].model)
        # Prespecified confounding comparisons, fitted only on the current training patients.
        if any(v in dataset.table for v in config['clinical_covariates']):
            base = streams.get('auc_selected', next(iter(streams.values())))[0]['candidate']
            for stage in ('covariate_only', 'covariate_combined'):
                cov = replace(base, stage=stage, model='lr', params_json='{}')
                value = evaluate(cov)
                if value is not None:
                    streams[stage] = (value, stage, 'lr')
        # Multiple streams can refer to the same winner; reuse fit but never outer scores for selection.
        from collections import Counter
        remaining_uses=Counter(result['candidate'] for result,_,_ in streams.values())
        outer_cache = {}
        for stream, (result, objective, model_name) in streams.items():
            candidate = result['candidate']
            inner_p = result['predictions']; good = np.isfinite(inner_p).all(axis=1)
            calibrator = OOFCalibrator(config['calibration']).fit(train_y[good], inner_p[good], groups[tr][good])
            cal_train = calibrator.transform(inner_p[good])
            w = patient_weights(groups[tr])[good]
            sensitivity_threshold = .5
            operating = {}
            if len(classes) == 2:
                sensitivity_threshold = threshold_for_sensitivity(train_y[good], cal_train[:,1], config['sensitivity_target'], w)
                operating = {str(target): select_threshold(train_y[good], cal_train[:,1], target, 'specificity', w)
                             for target in config['specificity_targets']}
            if candidate not in outer_cache:
                outer_cache[candidate] = fit_predict(dataset, indices, y, candidate, tr, te, seed, config)
            raw, meta, fitted = outer_cache[candidate]
            covered = np.isfinite(raw).all(axis=1)
            calibrated = np.full_like(raw, np.nan)
            calibrated[covered] = calibrator.transform(raw[covered])
            decisions = np.full(len(te), -1, int)
            decisions[covered] = (calibrated[covered,1] >= sensitivity_threshold).astype(int) if objective == 'sensitivity' and len(classes)==2 else np.argmax(calibrated[covered], axis=1)
            selection = dict(task=task, stream=stream, objective=objective, model=model_name,
                             repeat=repeat, outer_fold=fold, evaluation=evaluation, candidate=candidate.dictionary(),
                             inner_selection_objective=result['objectives'].get(objective), calibration=config['calibration'],
                             sensitivity_threshold=str(sensitivity_threshold) if not np.isfinite(sensitivity_threshold) else sensitivity_threshold, specificity_thresholds={key:str(val) if not np.isfinite(val) else val for key,val in operating.items()}, **meta)
            selections.append(selection)
            for j, ix in enumerate(te):
                row = dict(task=task, stream=stream, objective=objective, model=model_name, evaluation=evaluation,
                           repeat=repeat, outer_fold=fold, sample_id=dataset.table.iloc[indices[ix]].sample_id,
                           patient_id=groups[ix], y_true=int(y[ix]), y_pred=int(decisions[j]), covered=bool(covered[j]),
                           sensitivity_threshold=sensitivity_threshold)
                for k, name in enumerate(classes):
                    row[f'p_{k}'] = calibrated[j,k]; row[f'raw_p_{k}'] = raw[j,k]
                for target, threshold in operating.items():
                    row[f'pred_at_specificity_{target}'] = int(calibrated[j,1] >= threshold) if covered[j] else -1
                predictions.append(row)
            if config.get('save_models'):
                import joblib
                folder = taskdir / 'models'; folder.mkdir(exist_ok=True)
                joblib.dump(dict(features=fitted[0], model=fitted[1], calibrator=calibrator,
                                 clinical_preprocessor=fitted[2], clinical_columns=meta['clinical_covariates'],
                                 classes=classes, sensitivity_threshold=sensitivity_threshold, cutoff=candidate.cutoff),
                            folder / f'{fold_name}_{stream}.joblib')
            remaining_uses[candidate]-=1
            if remaining_uses[candidate]==0:
                outer_cache.pop(candidate,None)
        # Checkpoint whole fold artifacts: no need to wait for a long run to audit progress.
        write_json(taskdir / 'splits.json', splits_log)
        write_json(taskdir / 'selections.json', selections)
        write_json(taskdir / 'candidate_failures.json', failures)
        pd.DataFrame(predictions).to_csv(taskdir / 'outer_predictions.csv', index=False)
        pd.DataFrame(inner_rows).to_csv(taskdir / 'inner_search.csv', index=False)
        config.pop('_feature_cache',None)
        # Do not retain all inner OOF arrays beyond this outer fold.
    write_json(taskdir / 'classes.json', dict(classes=list(classes), positive_class=classes[1] if len(classes)==2 else None,
                                            probability_columns={f'p_{k}': v for k,v in enumerate(classes)}))
    return pd.DataFrame(predictions), selections
