#!/usr/bin/env python3
"""Audit acquisition-quality averages against supplied test means, without inference."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np
import pandas as pd
from scripts.prepare_revision_source import unpack, complex_array
from revision.data import digest
from revision.pipeline import write_json
from revision.features import finite_mean


def audit(mat, data, source_output, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    repeats = np.load(Path(source_output) / "test_mean_repeats.npy", mmap_mode="r")
    index = pd.read_csv(Path(source_output) / "repeat_index.csv")
    lookup = {}
    for path in sorted(Path(data).glob("*.npz")):
        with np.load(path, allow_pickle=False) as item:
            lookup[int(item["py_idx"])] = path
    rows = []
    with h5py.File(mat) as f:
        group = f["filtered_cases"]
        for j, row in index.iterrows():
            i = int(row.py_idx)
            full = f[group["Full_data"][i, 0]]
            quality = np.array(
                [float(unpack(f, f[full[t, 1]])) for t in range(full.shape[0])]
            )
            site = repeats[j, :, :, : full.shape[0]]
            with np.load(lookup[i], allow_pickle=False) as item:
                reported = float(item["Avg_Cutpats"])
                z = item["Zdata"]
            reconstructed = finite_mean(site, axis=2)
            same = np.allclose(z, reconstructed, equal_nan=True, rtol=1e-8, atol=1e-8)
            rows.append(
                dict(
                    sample_id=row.sample_id,
                    py_idx=i,
                    n_test_means=full.shape[0],
                    reported_quality_percent=reported,
                    mean_test_quality_percent=float(quality.mean()),
                    quality_mean_abs_difference=abs(reported - quality.mean()),
                    exported_equals_nanmean_of_test_means=bool(same),
                    observed_missing_percent=100 * (~np.isfinite(z).all(axis=0)).mean(),
                    mean_test_missing_percent=float(
                        100 * (~np.isfinite(site).all(axis=0)).mean()
                    ),
                )
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "source_quality_audit.csv", index=False)
    info = dict(
        n_sites=len(frame),
        all_reported_quality_equals_mean_of_test_quality=bool(
            (frame.quality_mean_abs_difference < 1e-8).all()
        ),
        max_quality_mean_abs_difference=float(frame.quality_mean_abs_difference.max()),
        n_exported_spectra_equal_nanmean_of_test_means=int(
            frame.exported_equals_nanmean_of_test_means.sum()
        ),
        interpretation="Reported Avg_Cutpats is checked against the arithmetic mean of the supplied per-test rejection percentages. Missingness of the final averaged spectrum is a different quantity. This does not reconstruct within-burst filtering or contact controls.",
        mat_sha256=digest(mat),
        script_sha256=digest(__file__),
    )
    write_json(output / "source_quality_audit.json", info)
    return info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mat", default="source/Python-Labeled-CaseData-17-Sep-2025.mat")
    p.add_argument("--data", default="data")
    p.add_argument("--source-output", default="results/revision-source")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    print(audit(a.mat, a.data, a.source_output, a.output))


if __name__ == "__main__":
    main()
