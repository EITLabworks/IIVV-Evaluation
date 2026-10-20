"""Compatibility loaders for inspection only; scaling belongs inside training folds."""

from pathlib import Path
import numpy as np
from revision.data import load_dataset


def load_data(path="data/", Cutpats_threshold=60, norm=None, print_info=True):
    """Return unnormalized magnitudes/NaNs in the historical tuple format.

    Cohort-wide normalization is deliberately rejected. New evaluations must use
    run_revision.sh, which fits scaling and imputation on each training partition.
    Do not pass these raw arrays directly to historical ungrouped model helpers.
    """
    if norm not in (None, "none"):
        raise ValueError(
            "Cohort-wide normalization is disabled. Use revision.features.TrainScaler inside patient-grouped CV, or run_revision.sh."
        )
    dataset = load_dataset(path)
    t = dataset.table
    keep = t.missing_percent.to_numpy() < Cutpats_threshold
    original_files = sorted(Path(path).glob("*.npz"))
    extra = {
        key: []
        for key in (
            "py_idx",
            "labels_healthy_vs_cancer",
            "labels_cancer_vs_lr_hr",
            "labels_MC",
        )
    }
    for i, file in enumerate(original_files):
        with np.load(file, allow_pickle=False) as a:
            for key in extra:
                extra[key].append(
                    a[key].item() if key in a else i if key == "py_idx" else "None"
                )
    if print_info:
        print(
            f"Loaded {keep.sum()} raw sites; no normalization or imputation performed."
        )
    return (
        np.asarray(extra["py_idx"])[keep],
        t.patient_id.to_numpy()[keep],
        t.site_id.to_numpy()[keep],
        np.abs(dataset.z[keep]),
        t.missing_percent.to_numpy()[keep],
        t.raw_pathology.to_numpy()[keep],
        *(
            np.asarray(extra[key])[keep]
            for key in (
                "labels_healthy_vs_cancer",
                "labels_cancer_vs_lr_hr",
                "labels_MC",
            )
        ),
    )


def load_data_no_norm(path="data/", Cutpats_threshold=60, print_info=True):
    result = load_data(path, Cutpats_threshold, norm=None, print_info=print_info)
    return result[:5]
