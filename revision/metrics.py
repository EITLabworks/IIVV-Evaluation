"""Named-class diagnostic metrics and patient-cluster uncertainty."""

import numpy as np
import pandas as pd
from .features import patient_weights

# Precision and PPV are deliberately both exported, but represent the same statistic.
CLASSIFICATION_METRICS = {
    "auc": "AUC",
    "f1": "F1",
    "precision": "precision",
    "sensitivity": "SN",
    "specificity": "SP",
    "ppv": "PPV",
    "npv": "NPV",
}


def classification_rows(summary, intervals):
    """Requested reporting metrics, with patient-bootstrap intervals when estimable."""
    ci = intervals.set_index("metric").to_dict("index")
    return [
        dict(
            metric=key,
            display_metric=label,
            estimate=summary.get(key, np.nan),
            **ci.get(key, {})
        )
        for key, label in CLASSIFICATION_METRICS.items()
    ]


def corrected_resampled_test(differences, train_patients, test_patients):
    """Nadeau--Bengio correction applied to paired patient-grouped outer CV folds.

    SE² = (1/n + mean(n_test_patients/n_train_patients)) * sample_variance(d).
    Student t reference uses n-1 degrees of freedom. This is an approximation
    (especially for a single k-fold partition and nonlinear diagnostic metrics),
    not an exact independent-fold test. All scheduled pairs must be estimable.
    References: Bouckaert & Frank (2004), sections 3.2--3.3,
    https://ml.cms.waikato.ac.nz/publications/2004/bouckaert-frank.pdf
    """
    from scipy.stats import t as student_t

    d, tr, te = [
        np.asarray(v, float) for v in (differences, train_patients, test_patients)
    ]
    if not (d.ndim == tr.ndim == te.ndim == 1 and len(d) == len(tr) == len(te)):
        raise ValueError("Paired differences and split patient counts must align")
    result = dict(
        n_scheduled_folds=len(d),
        n_valid_folds=int(np.isfinite(d).sum()),
        mean_difference=np.nan,
        sample_variance=np.nan,
        corrected_se=np.nan,
        mean_test_train_patient_ratio=np.nan,
        t_statistic=np.nan,
        degrees_of_freedom=len(d) - 1,
        p_value=np.nan,
        method="Nadeau-Bengio corrected resampled t approximation on paired outer folds",
        correction_unit="patients; mean scheduled test/train patient ratio",
        status="unavailable_incomplete_or_undefined_fold_pairs",
    )
    if (
        len(d) < 2
        or not np.isfinite(d).all()
        or not np.isfinite(tr).all()
        or not np.isfinite(te).all()
        or (tr <= 0).any()
        or (te <= 0).any()
    ):
        return result
    variance = float(np.var(d, ddof=1))
    correction = float(np.mean(te / tr))
    se = float(np.sqrt((1 / len(d) + correction) * variance))
    result.update(
        mean_difference=float(d.mean()),
        sample_variance=variance,
        corrected_se=se,
        mean_test_train_patient_ratio=correction,
    )
    if np.ptp(d) == 0:
        # Identical algorithms are a known null. Constant nonzero differences
        # provide no empirical variance estimate; do not invent significance.
        result.update(
            status=(
                "identical_fold_scores"
                if np.all(d == 0)
                else "unavailable_zero_between_fold_variance"
            ),
            t_statistic=0.0 if np.all(d == 0) else np.nan,
            p_value=1.0 if np.all(d == 0) else np.nan,
        )
        return result
    statistic = float(d.mean() / se)
    result.update(
        t_statistic=statistic,
        p_value=float(2 * student_t.sf(abs(statistic), len(d) - 1)),
        status="estimated_approximate",
    )
    return result


def ratio(a, b):
    return float(a / b) if b else float("nan")


def weighted_auc(y, probability, weights=None):
    """Tie-aware binary rank AUC, including sample weights."""
    y = np.asarray(y, dtype=bool)
    p = np.asarray(probability)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    positive = w[y].sum()
    negative = w[~y].sum()
    if positive <= 0 or negative <= 0:
        return np.nan
    order = np.argsort(p, kind="stable")
    values = p[order]
    yy = y[order]
    ww = w[order]
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    pos = np.add.reduceat(ww * yy, starts)
    neg = np.add.reduceat(ww * (~yy), starts)
    prior = np.cumsum(neg) - neg
    return float(np.sum(pos * (prior + 0.5 * neg)) / (positive * negative))


def score_arrays(y, p, pred=None, weights=None):
    """Numerical metrics without constructing pandas objects in resampling loops."""
    y, p = np.asarray(y, int), np.asarray(p, float)
    k = p.shape[1]
    pred = np.argmax(p, axis=1) if pred is None else np.asarray(pred, int)
    if not len(y):
        return {}, [], np.zeros((k, k))
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    cm = np.bincount(y * k + pred, weights=w, minlength=k * k).reshape(k, k)
    total = cm.sum()
    truth = np.eye(k)[y]
    perclass = []
    for c in range(k):
        tp = cm[c, c]
        fn = cm[c].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = total - tp - fn - fp
        perclass.append(
            dict(
                class_index=c,
                support=cm[c].sum(),
                tp=tp,
                fn=fn,
                fp=fp,
                tn=tn,
                sensitivity=ratio(tp, tp + fn),
                specificity=ratio(tn, tn + fp),
                precision=ratio(tp, tp + fp),
                ppv=ratio(tp, tp + fp),
                npv=ratio(tn, tn + fn),
                f1=ratio(2 * tp, 2 * tp + fp + fn),
                accuracy=ratio(tp + tn, total),
                auc=weighted_auc(y == c, p[:, c], w),
            )
        )
    accuracy = float(np.trace(cm) / total)
    summary = dict(
        accuracy=accuracy,
        balanced_accuracy=float(
            np.mean([row["sensitivity"] for row in perclass if row["support"] > 0])
        ),
        macro_f1=float(
            np.mean([row["f1"] if np.isfinite(row["f1"]) else 0.0 for row in perclass])
        ),
        micro_f1=accuracy,
        macro_auc=float(np.mean([row["auc"] for row in perclass])),
        micro_auc=weighted_auc(truth.ravel(), p.ravel(), np.repeat(w, k)),
        brier=float(np.average(((p - truth) ** 2).sum(1), weights=w)),
        log_loss=float(
            np.average(
                -np.log(
                    np.clip(
                        p[np.arange(len(y)), y],
                        np.finfo(p.dtype).eps,
                        1 - np.finfo(p.dtype).eps,
                    )
                ),
                weights=w,
            )
        ),
    )
    summary.update(auc=summary["macro_auc"], f1=summary["macro_f1"])
    for key in ("precision", "sensitivity", "specificity", "ppv", "npv"):
        summary["macro_" + key] = float(np.mean([row[key] for row in perclass]))
        summary[key] = summary["macro_" + key]
    if k == 2:
        summary.update(
            {
                key: val
                for key, val in perclass[1].items()
                if key not in {"class_index", "support"}
            }
        )
        summary["brier"] /= 2
    return summary, perclass, cm


def scores(y, p, pred=None, weights=None):
    """All calculations use explicit 0..K-1 columns; binary positive class is 1."""
    summary, perclass, cm = score_arrays(y, p, pred, weights)
    return summary, pd.DataFrame(perclass), cm


def threshold_for_sensitivity(y, p, target=0.95, weights=None):
    """Training-OOF threshold with maximal specificity subject to sensitivity target."""
    return select_threshold(y, p, target, kind="sensitivity", weights=weights)


def select_threshold(y, p, target, kind="sensitivity", weights=None):
    y, p = np.asarray(y), np.asarray(p)
    if len(np.unique(y)) < 2:
        raise ValueError(
            "Threshold selection needs both classes in training OOF predictions"
        )
    thresholds = np.unique(np.r_[-np.inf, p, np.inf])
    choices = []
    for threshold in thresholds:
        pred = p >= threshold
        positive, negative = y == 1, y == 0
        w = np.ones(len(y)) if weights is None else weights
        se = np.average(pred[positive], weights=w[positive])
        sp = np.average(~pred[negative], weights=w[negative])
        if (se if kind == "sensitivity" else sp) >= target - 1e-12:
            choices.append((sp if kind == "sensitivity" else se, threshold))
    # Deterministic tie-break: higher threshold for sensitivity target; lower for specificity target.
    return float(
        sorted(choices, key=lambda x: (x[0], x[1] if kind == "sensitivity" else -x[1]))[
            -1
        ][1]
    )


def prevalence_table(perclass, prevalences=(0.1, 0.2)):
    rows = []
    for row in perclass.to_dict("records"):
        se, sp = row["sensitivity"], row["specificity"]
        for pi in prevalences:
            rows.append(
                dict(
                    class_index=row["class_index"],
                    assumed_prevalence=pi,
                    ppv=ratio(se * pi, se * pi + (1 - sp) * (1 - pi)),
                    npv=ratio(sp * (1 - pi), sp * (1 - pi) + (1 - se) * pi),
                    sensitivity=se,
                    specificity=sp,
                )
            )
    return pd.DataFrame(rows)


def clinical_curves(y, p, classes, weights=None, bins=10):
    calibration, decision = [], []
    w = np.ones(len(y)) if weights is None else np.asarray(weights)
    for k, name in enumerate(classes):
        target = (np.asarray(y) == k).astype(float)
        idx = np.minimum((p[:, k] * bins).astype(int), bins - 1)
        for b in range(bins):
            keep = idx == b
            if keep.any():
                calibration.append(
                    dict(
                        class_name=name,
                        bin=b,
                        n=int(keep.sum()),
                        mean_probability=np.average(p[keep, k], weights=w[keep]),
                        observed_fraction=np.average(target[keep], weights=w[keep]),
                    )
                )
        prevalence = np.average(target, weights=w)
        for t in np.linspace(0.01, 0.99, 99):
            positive = p[:, k] >= t
            tp = np.sum(w * positive * target) / w.sum()
            fp = np.sum(w * positive * (1 - target)) / w.sum()
            decision.append(
                dict(
                    class_name=name,
                    threshold=t,
                    net_benefit=tp - fp * t / (1 - t),
                    treat_all=prevalence - (1 - prevalence) * t / (1 - t),
                    treat_none=0,
                )
            )
    return pd.DataFrame(calibration), pd.DataFrame(decision)


def cluster_indices(groups, rng):
    clusters = np.unique(groups)
    chosen = rng.choice(clusters, len(clusters), replace=True)
    # Return resampled cluster IDs too, so two copies of a patient are distinct bootstrap units.
    blocks = [np.flatnonzero(groups == group) for group in chosen]
    return np.concatenate(blocks), np.concatenate(
        [np.full(len(b), i) for i, b in enumerate(blocks)]
    )


def bootstrap_metrics(
    y,
    p,
    pred,
    groups,
    *,
    n_boot=1000,
    seed=0,
    patient_weighted=False,
    base_weights=None,
    return_perclass=False
):
    rng = np.random.default_rng(seed)
    draws = []
    class_draws = []
    for _ in range(n_boot):
        ix, bootstrap_groups = cluster_indices(np.asarray(groups), rng)
        w = (
            np.asarray(base_weights)[ix]
            if base_weights is not None
            else patient_weights(bootstrap_groups) if patient_weighted else None
        )
        result, perclass, _ = score_arrays(y[ix], p[ix], pred[ix], w)
        draws.append(result)
        if return_perclass:
            class_draws.extend(perclass)
    frame = pd.DataFrame(draws)
    rows = []
    for col in frame:
        v = frame[col].dropna()
        rows.append(
            dict(
                metric=col,
                ci_low=v.quantile(0.025) if len(v) else np.nan,
                ci_high=v.quantile(0.975) if len(v) else np.nan,
                valid_bootstraps=len(v),
                method="patient-cluster percentile; conditional on frozen outer predictions",
            )
        )
    summary = pd.DataFrame(rows)
    if not return_perclass:
        return summary
    classes = pd.DataFrame(class_draws)
    class_rows = []
    for class_index, part in classes.groupby("class_index"):
        for metric in part.columns.difference(["class_index"]):
            values = part[metric].dropna()
            class_rows.append(
                dict(
                    class_index=class_index,
                    metric=metric,
                    ci_low=values.quantile(0.025) if len(values) else np.nan,
                    ci_high=values.quantile(0.975) if len(values) else np.nan,
                    valid_bootstraps=len(values),
                    method="patient-cluster percentile; conditional on frozen outer predictions",
                )
            )
    return summary, pd.DataFrame(class_rows)


def paired_cluster_comparison(y, first, second, groups, *, n_resamples=1000, seed=0):
    """Conditional paired-prediction test, not a test on dependent CV fold means.

    Difference is macro AUC. Patient blocks of two paired sets of predictions
    are exchanged. This does not integrate uncertainty from refitting the model.
    """

    def statistic(labels, a, b):
        return float(
            np.mean([weighted_auc(labels == k, a[:, k]) for k in range(a.shape[1])])
            - np.mean([weighted_auc(labels == k, b[:, k]) for k in range(b.shape[1])])
        )

    observed = statistic(y, first, second)
    rng = np.random.default_rng(seed)
    null, differences = [], []
    groups = np.asarray(groups)
    unique = np.unique(groups)
    for _ in range(n_resamples):
        ix, _ = cluster_indices(groups, rng)
        differences.append(statistic(y[ix], first[ix], second[ix]))
        swap = np.isin(groups, unique[rng.integers(0, 2, len(unique)).astype(bool)])
        null.append(
            statistic(
                y,
                np.where(swap[:, None], second, first),
                np.where(swap[:, None], first, second),
            )
        )
    differences = np.asarray(differences)
    differences = differences[np.isfinite(differences)]
    null = np.asarray(null)
    null = null[np.isfinite(null)]
    return dict(
        delta_macro_auc=observed,
        ci_low=np.quantile(differences, 0.025) if len(differences) else np.nan,
        ci_high=np.quantile(differences, 0.975) if len(differences) else np.nan,
        p_value=(
            (1 + np.sum(np.abs(null) >= abs(observed))) / (1 + len(null))
            if len(null) and np.isfinite(observed)
            else np.nan
        ),
        n_paired_sites=len(y),
        n_patients=len(unique),
        valid_bootstraps=len(differences),
        method="paired patient-block prediction permutation; conditional on fitted outer models",
    )


def holm(pvalues):
    """Keep unavailable tests in the prespecified family (as unrejected tests)."""
    p = np.asarray(pvalues, float)
    result = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    if len(order):
        result[order] = np.minimum(
            1, np.maximum.accumulate(p[order] * (len(p) - np.arange(len(order))))
        )
    return result
