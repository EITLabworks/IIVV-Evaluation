"""Descriptive CV populations from frozen splits; never used to fit/select models."""

import json
from pathlib import Path
import numpy as np
import pandas as pd

DEMOGRAPHIC_FIELDS = (
    "age",
    "sex",
    "smoking",
    "alcohol",
    "prior_radiation",
    "prior_surgery",
    "location",
    "inflammation",
    "treatment_history",
    "contact_pressure",
    "contact_angle",
)
NUMERIC_FIELDS = {"age", "contact_pressure", "contact_angle"}


def export_cv_demographics(folder, dataset, task, config):
    folder = Path(folder)
    indices, y, classes = dataset.task(task)
    table = dataset.table.iloc[indices].copy()
    table["task_class"] = [classes[v] for v in y]
    summaries, pathology, categorical, numeric, patients, availability = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for split in json.loads((folder / "splits.json").read_text()):
        roles = [
            (-1, "outer_train", split["train_patients"], split.get("train_sample_ids")),
            (-1, "outer_test", split["test_patients"], split.get("test_sample_ids")),
        ]
        for fold, inner in enumerate(split["inner"]):
            roles.extend(
                [
                    (
                        fold,
                        "inner_train",
                        inner["train_patients"],
                        inner.get("train_sample_ids"),
                    ),
                    (
                        fold,
                        "inner_validation",
                        inner["validation_patients"],
                        inner.get("validation_sample_ids"),
                    ),
                ]
            )
        for inner_fold, role, group_ids, sample_ids in roles:
            part = table[table.patient_id.isin(group_ids)]
            if sample_ids is not None:
                part = part[part.sample_id.isin(sample_ids)]
            identity = dict(
                task=task,
                evaluation=split["evaluation"],
                repeat=split["repeat"],
                outer_fold=split["outer_fold"],
                inner_fold=inner_fold,
                partition=role,
            )
            cutoffs = [
                None,
                *sorted(set(config["cutoffs"] + [config["default_cutoff"]])),
            ]
            for cutoff in cutoffs:
                retained = (
                    part if cutoff is None else part[part.missing_percent < cutoff]
                )
                meta = dict(
                    **identity,
                    population="unfiltered" if cutoff is None else "quality_retained",
                    cutoff=np.nan if cutoff is None else cutoff,
                )
                sizes = retained.groupby("patient_id").size()
                mixed = retained.groupby("patient_id").task_class.nunique()
                summaries.append(
                    dict(
                        **meta,
                        n_samples=len(retained),
                        n_patients=len(sizes),
                        n_samples_before_quality=len(part),
                        n_patients_before_quality=part.patient_id.nunique(),
                        n_quality_excluded_samples=len(part) - len(retained),
                        n_quality_excluded_patients=part.patient_id.nunique()
                        - len(sizes),
                        n_mixed_task_class_patients=int((mixed > 1).sum()),
                        samples_per_patient_min=sizes.min(),
                        samples_per_patient_median=sizes.median(),
                        samples_per_patient_max=sizes.max(),
                    )
                )
                if "raw_pathology" in retained:
                    for (label, raw), group in retained.groupby(
                        ["task_class", "raw_pathology"], dropna=False
                    ):
                        pathology.append(
                            dict(
                                **meta,
                                class_name=label,
                                raw_pathology=raw,
                                n_samples=len(group),
                                n_patients=group.patient_id.nunique(),
                            )
                        )
                # All-class and per-task-class summaries; patients can occur in
                # multiple tissue/class categories and must not be summed across them.
                strata = [("All classes", retained)] + [
                    (label, retained[retained.task_class == label]) for label in classes
                ]
                for label, group in strata:
                    for field in DEMOGRAPHIC_FIELDS:
                        values = (
                            group[field]
                            if field in group
                            else pd.Series(np.nan, index=group.index)
                        )
                        valid = values.notna() & values.astype(str).str.strip().ne("")
                        availability.append(
                            dict(
                                **meta,
                                class_name=label,
                                variable=field,
                                available_in_source=field in table,
                                n_samples=len(group),
                                n_patients=group.patient_id.nunique(),
                                n_recorded_samples=int(valid.sum()),
                                n_recorded_patients=group.loc[
                                    valid, "patient_id"
                                ].nunique(),
                            )
                        )
                        if field in NUMERIC_FIELDS:
                            numbers = pd.to_numeric(
                                values.where(valid), errors="coerce"
                            )
                            numbers = numbers.where(np.isfinite(numbers))
                            patient_values = (
                                pd.DataFrame(
                                    {"patient_id": group.patient_id, "value": numbers}
                                )
                                .groupby("patient_id")
                                .value
                            )
                            conflicts = patient_values.nunique() > 1
                            patient_numbers = (
                                patient_values.first().loc[~conflicts].dropna()
                            )
                            for unit, data in (
                                ("sample", numbers.dropna()),
                                ("consistent_patient", patient_numbers),
                            ):
                                numeric.append(
                                    dict(
                                        **meta,
                                        class_name=label,
                                        variable=field,
                                        unit=unit,
                                        n_observed=len(data),
                                        mean=data.mean(),
                                        sd=data.std(ddof=1),
                                        median=data.median(),
                                        q25=data.quantile(0.25),
                                        q75=data.quantile(0.75),
                                        minimum=data.min(),
                                        maximum=data.max(),
                                        n_patients_with_conflicting_values=int(
                                            conflicts.sum()
                                        ),
                                        n_non_numeric_recorded_samples=int(
                                            (valid & numbers.isna()).sum()
                                        ),
                                    )
                                )
                        else:
                            categories = values.where(valid, "<missing>").astype(str)
                            for category, members in group.groupby(
                                categories, dropna=False
                            ):
                                categorical.append(
                                    dict(
                                        **meta,
                                        class_name=label,
                                        variable=field,
                                        value=category,
                                        n_samples=len(members),
                                        n_patients=members.patient_id.nunique(),
                                        sample_denominator=len(group),
                                        patient_denominator=group.patient_id.nunique(),
                                    )
                                )
                for patient_id, group in retained.groupby("patient_id"):
                    row = dict(
                        **meta,
                        patient_id=patient_id,
                        n_samples=len(group),
                        task_classes=" | ".join(sorted(set(group.task_class))),
                        n_task_classes=group.task_class.nunique(),
                        raw_pathologies=" | ".join(
                            sorted(
                                set(
                                    group.get(
                                        "raw_pathology", pd.Series(dtype=str)
                                    ).dropna()
                                )
                            )
                        ),
                    )
                    for field in DEMOGRAPHIC_FIELDS:
                        values = (
                            group[field].dropna().astype(str)
                            if field in group
                            else pd.Series(dtype=str)
                        )
                        row[field] = " | ".join(sorted(set(values)))
                    patients.append(row)
    for name, rows in (
        ("cv_population_summary.csv", summaries),
        ("cv_pathology_breakdown.csv", pathology),
        ("cv_demographics_categorical.csv", categorical),
        ("cv_demographics_numeric.csv", numeric),
        ("cv_demographics_availability.csv", availability),
        ("cv_patient_groups.csv", patients),
    ):
        pd.DataFrame(rows).to_csv(folder / name, index=False)
    (folder / "cv_demographics_notes.json").write_text(
        json.dumps(
            dict(
                counts="Samples are measurement sites; patients deduplicated within each displayed stratum",
                overlap="Patients with several tissue classes/locations can appear in multiple strata; do not sum patient counts",
                numeric="No demographic imputation; patient statistics exclude patients with conflicting recorded values",
                unavailable="Missing fields stay unavailable; observed information is never inferred from pathology",
                privacy="cv_patient_groups.csv contains patient identifiers, as do the existing split/inclusion artifacts",
                role="Descriptive only, downstream of saved partitions; never an input to feature/model selection",
            ),
            indent=2,
        )
        + "\n"
    )
