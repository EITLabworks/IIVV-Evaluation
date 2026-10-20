"""Raw inputs and explicit pathology semantics; no learned transforms in loaders."""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

CLASS_NAMES = {
    "healthy_vs_cancer": ("Healthy", "Cancer"),
    "cancer_vs_lr_hr": ("Cancer", "High-grade dysplasia", "Non-malignant"),
    "MC": ("Healthy", "Cancer", "High-grade dysplasia", "Non-malignant"),
    "highrisk_vs_lowrisk": ("Low risk", "High risk"),
}
# These are the explicit original conversion-notebook mappings, now centralized.
PATHOLOGY = {
    "healthy": "Healthy",
    "normal": "Non-malignant",
    "benign": "Non-malignant",
    "mild dysplasia": "Non-malignant",
    "moderate dysplasia": "High-grade dysplasia",
    "severe dysplasia": "High-grade dysplasia",
    "high grade dysplasia": "High-grade dysplasia",
    "high-grade dysplasia": "High-grade dysplasia",
    "high-grade dysplasia /cis": "High-grade dysplasia",
    "high-grade dysplasia/ cis": "High-grade dysplasia",
    "cis": "High-grade dysplasia",
    "carcinoma in situ": "High-grade dysplasia",
    "invasive carcinoma": "Cancer",
    "oscc": "Cancer",
}


def scalar(value):
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"Expected one scalar, got shape {arr.shape}")
    item = arr.reshape(-1)[0]
    return (
        item.decode("utf-8")
        if isinstance(item, bytes)
        else item.item() if hasattr(item, "item") else item
    )


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class Dataset:
    z: np.ndarray  # sites, frequencies, IIVV coordinates; NaNs retained
    table: pd.DataFrame  # stable sample_id / patient_id / site_id
    frequencies: np.ndarray
    patterns: np.ndarray | None
    complex_available: bool
    input_hashes: dict
    repeats: np.ndarray | None = None  # sites, frequencies, patterns, replicates

    def task(self, name):
        if name not in CLASS_NAMES:
            raise ValueError(f"Unknown task {name}")
        labels = self.table["pathology"].copy()
        if name == "highrisk_vs_lowrisk":
            labels = labels.map(
                {
                    "Healthy": "Low risk",
                    "Non-malignant": "Low risk",
                    "Cancer": "High risk",
                    "High-grade dysplasia": "High risk",
                }
            )
        keep = labels.isin(CLASS_NAMES[name]).to_numpy()
        indices = np.flatnonzero(keep)
        y = np.array([CLASS_NAMES[name].index(v) for v in labels[keep]], dtype=int)
        return indices, y, CLASS_NAMES[name]


def load_dataset(
    path,
    *,
    patterns_path=None,
    metadata_path=None,
    frequencies=None,
    label_map_path=None,
    trust_object_arrays=False,
    repeats_path=None,
    repeats_index_path=None,
):
    """Read the repository's one-site-per-NPZ format, preserving complex values/NaNs.

    Pickled object arrays are refused unless explicitly enabled for trusted input.
    Pattern/frequency order is a coordinate contract, never inferred from array size.
    """
    files = sorted(Path(path).glob("*.npz"))
    if not files:
        raise FileNotFoundError(
            f"No site NPZ files in {path}. See code-revisions.md for the input schema."
        )
    mapping = dict(PATHOLOGY)
    if label_map_path:
        mapping.update(json.loads(Path(label_map_path).read_text()))
    spectra, rows, bursts, hashes = [], [], [], {}
    common_freq = (
        np.asarray(frequencies, dtype=float) if frequencies is not None else None
    )
    complex_flags = []
    for file in files:
        with np.load(file, allow_pickle=trust_object_arrays) as a:
            for key in (
                "Zdata",
                "patient_no",
                "patient_sample",
                "LesionAssessment",
                "Avg_Cutpats",
            ):
                if key not in a:
                    raise ValueError(f"{file}: required key {key} is missing")
            z = np.asarray(a["Zdata"])
            if z.ndim != 2:
                raise ValueError(
                    f"{file}: Zdata must have shape (frequencies, patterns), got {z.shape}"
                )
            f = (
                np.asarray(a["frequencies_hz"], dtype=float).ravel()
                if "frequencies_hz" in a
                else common_freq
            )
            if f is None:
                if z.shape[0] != 31:
                    raise ValueError(
                        f"{file}: non-31-frequency arrays require frequencies_hz or a config frequency list"
                    )
                f = np.geomspace(100, 100000, 31)
            if (
                len(f) != z.shape[0]
                or not np.all(np.isfinite(f))
                or np.any(f <= 0)
                or np.any(np.diff(f) <= 0)
            ):
                raise ValueError(
                    f"{file}: frequencies must match axis 0 and be strictly increasing and positive"
                )
            if common_freq is None:
                common_freq = f
            if not np.array_equal(f, common_freq):
                raise ValueError(f"{file}: frequency coordinates differ between sites")
            raw = str(scalar(a["LesionAssessment"])).strip().lower()
            pathology = mapping.get(raw)
            legacy = (
                str(scalar(a["labels_MC"])).strip()
                if pathology and "labels_MC" in a
                else ""
            )
            # Reject a disagreement, rather than silently regrouping moderate dysplasia.
            if pathology and legacy in CLASS_NAMES["MC"] and legacy != pathology:
                raise ValueError(
                    f"{file}: raw pathology {raw!r} maps to {pathology!r}, conflicting with labels_MC={legacy!r}"
                )
            patient = str(scalar(a["patient_no"]))
            site = str(scalar(a["patient_sample"]))
            cut = float(scalar(a["Avg_Cutpats"]))
            if not np.isfinite(cut) or not 0 <= cut <= 100:
                raise ValueError(f"{file}: Avg_Cutpats must be finite and in [0,100]")
            row = dict(
                sample_id=f"{patient}:{site}",
                patient_id=patient,
                site_id=site,
                pathology=pathology,
                raw_pathology=raw,
                missing_percent=cut,
                observed_missing_pattern_percent=100
                * (~np.isfinite(z)).any(axis=0).mean(),
                source_file=str(file),
            )
            aliases = {
                "LesionLoc": "location",
                "age": "age",
                "sex": "sex",
                "smoking": "smoking",
                "alcohol": "alcohol",
                "prior_radiation": "prior_radiation",
                "inflammation": "inflammation",
                "treatment_history": "treatment_history",
                "contact_pressure": "contact_pressure",
                "contact_angle": "contact_angle",
                "external_cohort": "external_cohort",
            }
            for key, dest in aliases.items():
                if key in a:
                    row[dest] = scalar(a[key])
            spectra.append(
                z.astype(np.complex128 if np.iscomplexobj(z) else np.float64)
            )
            complex_flags.append(np.iscomplexobj(z))
            bursts.append(np.asarray(a["Zrepeats"]) if "Zrepeats" in a else None)
            rows.append(row)
        hashes[str(file)] = digest(file)
    if len({z.shape for z in spectra}) != 1:
        raise ValueError(
            "All sites must share the same frequency/IIVV coordinate shape"
        )
    table = pd.DataFrame(rows)
    if table["sample_id"].duplicated().any():
        raise ValueError(
            "Duplicate patient_no/patient_sample pairs; use unique stable site identifiers"
        )
    if metadata_path:
        meta = pd.read_csv(metadata_path, dtype={"patient_id": str, "site_id": str})
        keys = ["patient_id", "site_id"]
        if not set(keys) <= set(meta) or meta.duplicated(keys).any():
            raise ValueError("Metadata requires unique patient_id,site_id keys")
        protected = {
            "pathology",
            "raw_pathology",
            "missing_percent",
            "source_file",
            "sample_id",
        }
        if protected.intersection(meta.columns):
            raise ValueError(
                "Metadata must not overwrite outcome, quality, or identifier fields; use an explicit label map"
            )
        table = table.merge(
            meta, on=keys, how="left", validate="one_to_one", suffixes=("", "_metadata")
        )
        for col in meta.columns.difference(keys):
            if col + "_metadata" in table:
                table[col] = table.pop(col + "_metadata").combine_first(table[col])
        hashes[str(metadata_path)] = digest(metadata_path)
    patterns = None
    if patterns_path:
        if not Path(patterns_path).is_file():
            raise FileNotFoundError(
                f"Missing IIVV coordinates: {patterns_path}; never infer their order from feature counts"
            )
        patterns = np.loadtxt(patterns_path, dtype=int, ndmin=2)
        if patterns.shape != (spectra[0].shape[1], 4):
            raise ValueError(
                f"IIVV coordinate shape {patterns.shape} differs from ({spectra[0].shape[1]},4)"
            )
        if len(np.unique(patterns, axis=0)) != len(patterns):
            raise ValueError("IIVV coordinate rows must be unique")
        hashes[str(patterns_path)] = digest(patterns_path)
    if label_map_path:
        hashes[str(label_map_path)] = digest(label_map_path)
    repeats = None
    if any(v is not None for v in bursts):
        if not all(v is not None for v in bursts):
            raise ValueError(
                "Zrepeats must be supplied for every site or in a separate complete acquisition dataset"
            )
        repeats = np.stack(bursts)
        if repeats.shape[:3] != (len(rows), *spectra[0].shape) or repeats.ndim != 4:
            raise ValueError(
                "Zrepeats must have shape (frequencies, patterns, repeats) for each site"
            )
    if repeats_path:
        if repeats_index_path is None:
            raise ValueError(
                "An explicit repeats_index_path is required with a repeat array"
            )
        repeat_index = pd.read_csv(repeats_index_path, dtype={"sample_id": str})
        if repeat_index.sample_id.tolist() != table.sample_id.tolist():
            raise ValueError("Repeat-index sample order differs from the loaded sites")
        repeats = np.load(repeats_path, mmap_mode="r", allow_pickle=False)
        if repeats.ndim != 4 or repeats.shape[:3] != (len(table), *spectra[0].shape):
            raise ValueError(
                "External repeats must have shape (sites,frequencies,patterns,repeats)"
            )
        hashes[str(repeats_path)] = digest(repeats_path)
        hashes[str(repeats_index_path)] = digest(repeats_index_path)
    return Dataset(
        np.stack(spectra),
        table,
        common_freq,
        patterns,
        all(complex_flags),
        hashes,
        repeats,
    )
