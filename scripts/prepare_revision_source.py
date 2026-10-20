#!/usr/bin/env python3
"""Verify MATLAB-to-NPZ alignment and extract available metadata/test-mean repeats.

MATLAB v7.3 Full_data contains repositioned TEST MEANS, not individual bursts.
No attempt is made to decode MATLAB's opaque string-object serialization.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from revision.data import scalar, digest
from revision.pipeline import write_json


def unpack(f, node):
    for _ in range(8):
        if (
            isinstance(node, h5py.Dataset)
            and node.dtype == h5py.ref_dtype
            and node.size == 1
        ):
            node = f[node[...].flat[0]]
        else:
            break
    matlab_class = node.attrs.get("MATLAB_class", b"")
    if matlab_class == b"string":
        return None  # proprietary MCOS object payload is not a Unicode code-point array
    value = node[...]
    if matlab_class == b"char":
        return "".join(chr(int(c)) for c in value.ravel()).strip()
    return scalar(value) if value.size == 1 else value


def complex_array(node):
    a = node[...]
    return (
        a["real"] + 1j * a["imag"]
        if a.dtype.fields and {"real", "imag"} <= set(a.dtype.fields)
        else a
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mat", default="source/Python-Labeled-CaseData-17-Sep-2025.mat")
    p.add_argument("--data", default="data")
    p.add_argument("--output", default="results/revision-source")
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path(args.data).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(args.data)
    metadata = []
    index = []
    with h5py.File(args.mat) as f:
        group = f["filtered_cases"]
        first = complex_array(f[group["Zdata"][0, 0]])
        source_fields = sorted(group.keys())
        # Verify every coordinate array, cohort/case pair, and source index before writing.
        for path in paths:
            with np.load(path, allow_pickle=False) as a:
                i = int(a["py_idx"])
                bx_case = tuple(map(int, a["Bx_Case"]))
                original = (
                    int(unpack(f, f[group["Bx"][i, 0]])),
                    int(unpack(f, f[group["Case"][i, 0]])),
                )
                if original != bx_case:
                    raise ValueError(f"{path}: cohort/case misalignment")
                if not np.allclose(
                    a["Zdata"], complex_array(f[group["Zdata"][i, 0]]), equal_nan=True
                ):
                    raise ValueError(
                        f"{path}: MATLAB Zdata differs from exported array; no row alignment assumed"
                    )
                patient = str(scalar(a["patient_no"]))
                site = str(scalar(a["patient_sample"]))
                row = dict(patient_id=patient, site_id=site)
                for key, column in [
                    ("PriorRad", "prior_radiation"),
                    ("PriorSurgery", "prior_surgery"),
                ]:
                    value = unpack(f, f[group[key][i, 0]])
                    row[column] = (
                        value if isinstance(value, (str, int, float)) else None
                    )
                metadata.append(row)
                index.append(dict(sample_id=f"{patient}:{site}", py_idx=i))
        counts = [f[group["Full_data"][v["py_idx"], 0]].shape[0] for v in index]
        max_repeats = max(counts)
        repeats = np.lib.format.open_memmap(
            out / "test_mean_repeats.npy",
            mode="w+",
            dtype=np.complex128,
            shape=(len(paths), *first.shape, max_repeats),
        )
        repeats[:] = np.nan
        for row, item in enumerate(index):
            full = f[group["Full_data"][item["py_idx"], 0]]
            if len(full.shape) != 2 or full.shape[1] != 2:
                raise ValueError(
                    f"Unexpected Full_data shape {full.shape}; do not infer burst/test organization"
                )
            item["n_test_means"] = full.shape[0]
            for test in range(full.shape[0]):
                array = complex_array(f[full[test, 0]])
                if array.shape != first.shape:
                    raise ValueError("Inconsistent repeated spectrum coordinates")
                repeats[row, :, :, test] = array
        repeats.flush()
    pd.DataFrame(metadata).to_csv(out / "clinical_metadata.csv", index=False)
    pd.DataFrame(index).to_csv(out / "repeat_index.csv", index=False)
    write_json(
        out / "source_extraction.json",
        dict(
            mat_sha256=digest(args.mat),
            n_sites=len(paths),
            source_fields=source_fields,
            npz_hashes={str(path): digest(path) for path in paths},
            sidecar_hashes={
                name: digest(out / name)
                for name in [
                    "clinical_metadata.csv",
                    "repeat_index.csv",
                    "test_mean_repeats.npy",
                ]
            },
            repeat_units="two to four repositioned test-mean spectra; burst-level spectra unavailable",
            test_count_distribution={
                str(n): counts.count(n) for n in sorted(set(counts))
            },
            unavailable_metadata=[
                "age",
                "sex",
                "smoking",
                "alcohol",
                "pathology blinding",
                "histology coverage",
                "probe pressure",
            ],
            undecoded_opaque_matlab_string_fields=["HPV", "Tgrade"],
            all_exported_zdata_and_cohort_case_pairs_verified=True,
            metadata="PriorRad and PriorSurgery only; no treatment or risk factors inferred",
        ),
    )
    print(
        f"Verified {len(paths)} sites and extracted metadata/available test means to {out}"
    )


if __name__ == "__main__":
    main()
