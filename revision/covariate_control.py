"""A matched EIS-only LR comparator for the clinical-plus-EIS LR pipeline."""

from dataclasses import replace
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from .features import Candidate, patient_weights
from .pipeline import (
    group_splits,
    inner_evaluate,
    fit_predict,
    OOFCalibrator,
    write_json,
)
from .metrics import threshold_for_sensitivity, select_threshold


def add_matched_control(dataset, task, config, outdir, predictions, selections):
    """Reuse frozen outer-training-selected settings; never select on test outcomes.

    Combined LR uses C=1 with the selected EIS feature/preprocessing rule. Its
    matched comparator uses exactly that same LR and rule without covariates.
    Existing folds are reused only when all their test rows are present.
    """
    source = [row for row in selections if row["stream"] == "covariate_combined"]
    if not source:
        return predictions, selections
    folder = Path(outdir) / task
    indices, y, classes = dataset.task(task)
    config = {**config, "_class_names": list(classes)}
    groups = dataset.table.iloc[indices].patient_id.to_numpy()
    samples = dataset.table.iloc[indices].sample_id.to_numpy()
    rows = predictions.to_dict("records")
    fitted = []
    source_version = dict(
        module="revision/covariate_control.py",
        sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    for reference in source:
        repeat, fold, evaluation = (
            reference["repeat"],
            reference["outer_fold"],
            reference["evaluation"],
        )
        split = next(
            row
            for row in json.loads((folder / "splits.json").read_text())
            if (row["repeat"], row["outer_fold"], row["evaluation"])
            == (repeat, fold, evaluation)
        )
        tr = np.flatnonzero(np.isin(samples, split["train_sample_ids"]))
        te = np.flatnonzero(np.isin(samples, split["test_sample_ids"]))
        existing = [
            row
            for row in rows
            if row["stream"] == "covariate_eis_only"
            and (row["repeat"], row["outer_fold"], row["evaluation"])
            == (repeat, fold, evaluation)
        ]
        if existing:
            if len(existing) != len(te) or {
                row["sample_id"] for row in existing
            } != set(samples[te]):
                raise ValueError(
                    "Incomplete matched-control checkpoint; use a fresh output or repair its explicit partial block"
                )
            continue
        candidate = replace(
            Candidate(**reference["candidate"]), stage="covariate_eis_only"
        )
        assert candidate.model == "lr" and candidate.params_json == "{}"
        assert not set(groups[tr]) & set(groups[te])
        seed = config["seed"] + repeat * 1000 + fold * 100
        inner = group_splits(y[tr], groups[tr], config["inner_folds"], seed + 1)
        print(
            f"{task}: matched EIS-only LR control repeat={repeat} fold={fold}",
            flush=True,
        )
        result = inner_evaluate(
            dataset, indices[tr], y[tr], candidate, inner, seed, config
        )
        inner_p = result["predictions"]
        good = np.isfinite(inner_p).all(axis=1)
        calibrator = OOFCalibrator(config["calibration"]).fit(
            y[tr][good], inner_p[good], groups[tr][good]
        )
        cal_train = calibrator.transform(inner_p[good])
        w = patient_weights(groups[tr])[good]
        threshold = 0.5
        operating = {}
        if len(classes) == 2:
            threshold = threshold_for_sensitivity(
                y[tr][good], cal_train[:, 1], config["sensitivity_target"], w
            )
            operating = {
                str(target): select_threshold(
                    y[tr][good], cal_train[:, 1], target, "specificity", w
                )
                for target in config["specificity_targets"]
            }
        raw, meta, _ = fit_predict(dataset, indices, y, candidate, tr, te, seed, config)
        covered = np.isfinite(raw).all(axis=1)
        cal = np.full_like(raw, np.nan)
        cal[covered] = calibrator.transform(raw[covered])
        pred = np.full(len(te), -1, int)
        pred[covered] = cal[covered].argmax(axis=1)
        selections.append(
            dict(
                task=task,
                stream="covariate_eis_only",
                objective="covariate_eis_only",
                model="lr",
                repeat=repeat,
                outer_fold=fold,
                evaluation=evaluation,
                candidate=candidate.dictionary(),
                inner_selection_objective=None,
                calibration=config["calibration"],
                sensitivity_threshold=(
                    threshold if np.isfinite(threshold) else str(threshold)
                ),
                specificity_thresholds={
                    key: value if np.isfinite(value) else str(value)
                    for key, value in operating.items()
                },
                source_version=source_version,
                **meta,
            )
        )
        for j, ix in enumerate(te):
            row = dict(
                task=task,
                stream="covariate_eis_only",
                objective="covariate_eis_only",
                model="lr",
                evaluation=evaluation,
                repeat=repeat,
                outer_fold=fold,
                sample_id=samples[ix],
                patient_id=groups[ix],
                y_true=int(y[ix]),
                y_pred=int(pred[j]),
                covered=bool(covered[j]),
                sensitivity_threshold=threshold,
            )
            for k in range(len(classes)):
                row[f"p_{k}"] = cal[j, k]
                row[f"raw_p_{k}"] = raw[j, k]
            for target, value in operating.items():
                row[f"pred_at_specificity_{target}"] = (
                    int(cal[j, 1] >= value) if covered[j] else -1
                )
            rows.append(row)
        fitted.append(dict(repeat=repeat, outer_fold=fold, evaluation=evaluation))
        config.pop("_feature_cache", None)
        # A complete extra stream/fold can safely be reused after an interruption.
        pd.DataFrame(rows).to_csv(folder / "outer_predictions.csv", index=False)
        write_json(folder / "selections.json", selections)
    write_json(
        folder / "matched_covariate_control.json",
        dict(
            status="completed",
            source_version=source_version,
            matched_eis_stream="covariate_eis_only",
            combined_stream="covariate_combined",
            clinical_stream="covariate_only",
            model="LR with C=1 and identical frozen EIS preprocessing/features and outer/inner patients",
            n_outer_folds=len(source),
            newly_fitted_folds=fitted,
            scope="Descriptive paired pipeline comparison; available covariates only, not causal identification",
        ),
    )
    return pd.DataFrame(rows), selections
