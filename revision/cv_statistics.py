"""Outer-fold metrics and approximate corrected paired tests; no model fitting."""

import itertools
from pathlib import Path
import numpy as np
import pandas as pd

from .features import patient_weights
from .metrics import CLASSIFICATION_METRICS, corrected_resampled_test, holm, scores


def export_cv_statistics(predictions, splits, classes, config, folder):
    """Use each outer test fold separately; never treat inner-search scores as tests.

    Pair tuned and baseline predictions on identical sites/patients in each fold.
    Diagnostic comparisons condition on their common quality coverage. The
    correction uses the scheduled patient split BEFORE quality exclusion.
    """
    from .analysis import patient_frame

    folder = Path(folder)
    k = len(classes)
    task = folder.name
    probability = [f"p_{j}" for j in range(k)]
    fold_rows, class_rows, differences, tests = [], [], [], []
    for split in splits:
        mask = np.ones(len(predictions), bool)
        for key in ("evaluation", "repeat", "outer_fold"):
            mask &= predictions[key].to_numpy() == split[key]
        part = predictions.loc[mask]
        identity = {key: split[key] for key in ("evaluation", "repeat", "outer_fold")}
        counts = dict(
            n_train_patients=len(split["train_patients"]),
            n_test_patients=len(split["test_patients"]),
            n_train_sites=len(split["train_sample_ids"]),
            n_test_sites=len(split["test_sample_ids"]),
        )
        for stream, group in part.groupby("stream"):
            endpoint, _ = patient_frame(group, k)
            for unit, frame in (
                ("site", group),
                ("patient_weighted_site", group),
                ("patient", endpoint),
            ):
                if frame.empty:
                    continue
                covered = frame.covered.to_numpy(bool)
                full_weights = (
                    patient_weights(frame.patient_id.to_numpy())
                    if unit == "patient_weighted_site"
                    else np.ones(len(frame))
                )
                summary, pc, _ = scores(
                    frame.loc[covered, "y_true"].to_numpy(int),
                    frame.loc[covered, probability].to_numpy(float),
                    frame.loc[covered, "y_pred"].to_numpy(int),
                    full_weights[covered],
                )
                meta = dict(
                    task=task,
                    stream=stream,
                    unit=unit,
                    **identity,
                    **counts,
                    n_sites=len(frame),
                    n_patients=frame.patient_id.nunique(),
                    n_covered=int(covered.sum()),
                    coverage=float(full_weights[covered].sum() / full_weights.sum()),
                )
                fold_rows.append(dict(**meta, **summary))
                for row in pc.to_dict("records"):
                    class_rows.append(
                        dict(**meta, **row, class_name=classes[row["class_index"]])
                    )
        if split["evaluation"] != "nested_cv":
            continue  # external cohorts are a separate evaluation, not extra CV folds
        for model, objective in itertools.product(
            config["models"], config["objectives"]
        ):
            first = part[part.stream == f"{objective}_{model}"]
            second = part[part.stream == f"baseline_{model}"]
            pairs = first.merge(
                second,
                on=["sample_id", "patient_id", "y_true"],
                suffixes=("_a", "_b"),
                validate="one_to_one",
            )
            expected = set(split["test_sample_ids"])
            complete = set(first.sample_id) == set(second.sample_id) == expected
            good = pairs.covered_a.to_numpy(bool) & pairs.covered_b.to_numpy(bool)
            full_weights = (
                patient_weights(pairs.patient_id.to_numpy())
                if len(pairs)
                else np.array([])
            )
            a, b = {}, {}
            if complete and good.any():
                for suffix, destination in (("a", a), ("b", b)):
                    values, _, _ = scores(
                        pairs.loc[good, "y_true"].to_numpy(int),
                        pairs.loc[good, [f"p_{j}_{suffix}" for j in range(k)]].to_numpy(
                            float
                        ),
                        pairs.loc[good, f"y_pred_{suffix}"].to_numpy(int),
                        full_weights[good],
                    )
                    destination.update(values)
            for metric, label in CLASSIFICATION_METRICS.items():
                x, y = a.get(metric, np.nan), b.get(metric, np.nan)
                differences.append(
                    dict(
                        task=task,
                        model=model,
                        objective=objective,
                        metric=metric,
                        display_metric=label,
                        **identity,
                        **counts,
                        unit="patient_weighted_site",
                        tuned_score=x,
                        baseline_score=y,
                        difference=x - y,
                        n_common_covered_sites=int(good.sum()),
                        n_common_covered_patients=pairs.loc[
                            good, "patient_id"
                        ].nunique(),
                        common_weighted_coverage=(
                            float(full_weights[good].sum() / full_weights.sum())
                            if len(pairs)
                            else 0.0
                        ),
                        conditional_on_common_quality_coverage=True,
                    )
                )
    folds = pd.DataFrame(fold_rows)
    # Keep every requested column even when all folds abstain for a stream.
    for key in CLASSIFICATION_METRICS:
        if key not in folds:
            folds[key] = np.nan
    folds.to_csv(folder / "outer_fold_metrics.csv", index=False)
    descriptive = []
    for (evaluation, stream, unit), group in folds.groupby(
        ["evaluation", "stream", "unit"]
    ):
        expected_folds = {
            (s["repeat"], s["outer_fold"])
            for s in splits
            if s["evaluation"] == evaluation
        }
        complete = set(zip(group["repeat"], group.outer_fold)) == expected_folds
        for metric, label in CLASSIFICATION_METRICS.items():
            values = group[metric].dropna()
            estimable = complete and len(values) == len(expected_folds)
            descriptive.append(
                dict(
                    task=task,
                    evaluation=evaluation,
                    stream=stream,
                    unit=unit,
                    metric=metric,
                    display_metric=label,
                    mean=values.mean() if estimable else np.nan,
                    sd=values.std(ddof=1) if estimable else np.nan,
                    n_valid_folds=len(values),
                    n_observed_folds=len(group),
                    n_scheduled_folds=len(expected_folds),
                    complete_metric=estimable,
                    note="Descriptive fold dispersion; not a standard error or confidence interval",
                )
            )
    pd.DataFrame(descriptive).to_csv(folder / "cv_metrics_summary.csv", index=False)
    pd.DataFrame(class_rows).to_csv(
        folder / "outer_fold_per_class_metrics.csv", index=False
    )
    differences = pd.DataFrame(differences)
    differences.to_csv(folder / "cv_comparison_fold_differences.csv", index=False)
    for (model, objective, metric), group in differences.groupby(
        ["model", "objective", "metric"]
    ):
        tests.append(
            dict(
                task=task,
                model=model,
                objective=objective,
                metric=metric,
                display_metric=CLASSIFICATION_METRICS[metric],
                alias_of="precision" if metric == "ppv" else "",
                unit="patient_weighted_site",
                conditional_on_common_quality_coverage=True,
                reporting_scope=(
                    "binary positive class" if k == 2 else "macro one-vs-rest"
                ),
                **corrected_resampled_test(
                    group.difference, group.n_train_patients, group.n_test_patients
                ),
            )
        )
    result = pd.DataFrame(tests)
    result.to_csv(folder / "corrected_resampled_tests.csv", index=False)
    return result


def combine_corrected_tests(root, tasks):
    """Correct all prespecified tests together, retaining undefined tests in the family.

    Precision=PPV is tested only once; its adjusted values are copied to the PPV
    reporting alias. Both within-metric and across-all-unique-metrics Holm values
    are exported; the latter is the conservative default for multiple metrics.
    """
    root = Path(root)
    frame = pd.concat(
        [pd.read_csv(root / task / "corrected_resampled_tests.csv") for task in tasks],
        ignore_index=True,
    )
    unique = frame.metric != "ppv"
    frame["p_holm_all_unique_metrics"] = np.nan
    frame.loc[unique, "p_holm_all_unique_metrics"] = holm(frame.loc[unique, "p_value"])
    frame["p_holm_within_metric"] = np.nan
    for _, group in frame.loc[unique].groupby("metric"):
        frame.loc[group.index, "p_holm_within_metric"] = holm(group.p_value)
    for index in frame.index[~unique]:
        alias = frame.loc[index]
        source = frame[
            (frame.task == alias.task)
            & (frame.model == alias.model)
            & (frame.objective == alias.objective)
            & (frame.metric == "precision")
        ].iloc[0]
        for column in ("p_holm_all_unique_metrics", "p_holm_within_metric"):
            frame.loc[index, column] = source[column]
    frame["n_unique_hypotheses"] = int(unique.sum())
    frame.to_csv(root / "corrected_resampled_tests_holm.csv", index=False)
    return frame
