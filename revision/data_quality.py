"""Descriptive cohort exploration; these fitted summaries never enter model selection."""

import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from .plot_style import apply_style, class_color

apply_style()

from .data import digest, TASK3_CAPTION, TASK_DEFINITIONS, task_labels
from .demographics import export_cv_demographics
from .features import _geometry, finite_mean, patient_weights
from .pipeline import group_splits, make_split_record, task_outer_splits, write_json


def quality_protocol(config):
    """Only settings that govern descriptive summaries and planned patient splits."""
    protocol = {
        key: config.get(key)
        for key in (
            "schema_version",
            "task_protocol",
            "tasks",
            "seed",
            "outer_folds",
            "inner_folds",
            "outer_repeats",
            "cutoffs",
            "default_cutoff",
            "masks",
            "bootstrap_samples",
            "clinical_covariates",
            "external_cohort_column",
            "external_cohort_values",
            "data_quality",
        )
    }
    protocol["task_definitions"] = {
        task: TASK_DEFINITIONS[task] for task in config["tasks"]
    }
    return protocol


def weighted_projection(x, groups):
    """Patient-balanced mean imputation, standardization and two-component SVD."""
    x = np.asarray(x, float)
    if x.ndim != 2 or min(x.shape) < 2:
        raise ValueError(
            "A two-dimensional PCA requires at least two sites and frequencies"
        )
    weights = patient_weights(np.asarray(groups))
    weights = weights / weights.sum()
    finite = np.isfinite(x)
    observed_weight = (finite * weights[:, None]).sum(0)
    means = np.divide(
        (np.where(finite, x, 0) * weights[:, None]).sum(0),
        observed_weight,
        out=np.zeros(x.shape[1]),
        where=observed_weight > 0,
    )
    filled = np.where(finite, x, means)
    scale = np.sqrt(((filled - means) ** 2 * weights[:, None]).sum(0))
    scaled = (filled - means) / np.where(scale > 0, scale, 1)
    _, singular, components = np.linalg.svd(
        scaled * np.sqrt(weights[:, None]), full_matrices=False
    )
    if not np.any(singular > 0):
        raise ValueError(
            "PCA is undefined: all frequency summaries are constant or missing"
        )
    # Fix arbitrary SVD signs so saved figures are deterministic.
    for row in components:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    return (
        scaled @ components[:2].T,
        components[:2],
        singular[:2] ** 2 / np.sum(singular**2),
        observed_weight,
    )


def save_figure(fig, destination, labels=()):
    if "Low-grade dysplasia" in set(labels):
        fig.set_layout_engine(None)
        fig.text(0.5, 0.01, TASK3_CAPTION, ha="center", va="bottom", fontsize=7)
        fig.tight_layout(rect=(0, 0.10, 1, 1))
    for suffix in (".pdf", ".png"):
        fig.savefig(Path(destination).with_suffix(suffix), dpi=160)
    plt.close(fig)


def save_projection(x, table, frequencies, destination, labels, title):
    destination = Path(destination)
    coordinates, loadings, variance, observed = weighted_projection(x, table.patient_id)
    frame = table[["sample_id", "patient_id", "pathology"]].copy()
    frame["class_name"] = np.asarray(labels)
    frame[["PC1", "PC2"]] = coordinates
    frame.to_csv(destination.with_suffix(".csv"), index=False)
    pd.DataFrame(
        {
            "frequency_hz": frequencies,
            "PC1_loading": loadings[0],
            "PC2_loading": loadings[1],
            "observed_patient_weight": observed,
        }
    ).to_csv(destination.with_name(destination.name + "-loadings.csv"), index=False)
    write_json(
        destination.with_suffix(".json"),
        dict(
            explained_variance_ratio=variance.tolist(),
            n_sites=len(table),
            n_patients=table.patient_id.nunique(),
            representation="log1p(mean finite magnitude across IIVV coordinates / 1 ohm), one row per site",
            scaling="Patient-balanced mean imputation and weighted standardization before weighted SVD",
            scope="Whole-cohort descriptive projection only; never imported into cross-validation",
        ),
    )
    fig, ax = plt.subplots(figsize=(6.5, 4.5), layout="constrained")
    for label in pd.unique(frame.class_name):
        part = frame[frame.class_name == label]
        ax.scatter(
            part.PC1, part.PC2, label=label, color=class_color(label), s=20, alpha=0.65
        )
    ax.set(
        xlabel=f"PC1 ({variance[0]:.1%})",
        ylabel=f"PC2 ({variance[1]:.1%})",
        title=title,
    )
    ax.legend(fontsize=8)
    save_figure(fig, destination, labels)


def explore(dataset, config, output):
    """Export quality, MI and PCA evidence without fitting a diagnostic classifier."""
    folder = Path(output) / "exploration"
    folder.mkdir(parents=True, exist_ok=True)
    table, z = dataset.table, dataset.z
    table.groupby(["pathology", "raw_pathology"], dropna=False).agg(
        n_sites=("sample_id", "size"), n_patients=("patient_id", "nunique")
    ).to_csv(folder / "cohort_pathology_counts.csv")
    finite = np.isfinite(z)
    magnitude = np.where(finite, np.abs(z), np.nan)
    summary = finite_mean(magnitude, axis=2)
    transformed = np.log1p(summary)
    pd.DataFrame({"frequency_hz": dataset.frequencies}).to_csv(
        folder / "frequencies_hz.csv", index=False
    )
    site = table.copy()
    site["nonfinite_coordinate_percent"] = 100 * (~finite).mean(axis=(1, 2))
    site["median_magnitude_ohm"] = np.nanmedian(magnitude, axis=(1, 2))
    site.to_csv(folder / "site_quality.csv", index=False)
    by_frequency = pd.DataFrame(
        {
            "frequency_hz": dataset.frequencies,
            "nonfinite_percent": 100 * (~finite).mean(axis=(0, 2)),
            "median_magnitude_ohm": np.nanmedian(magnitude, axis=(0, 2)),
            "magnitude_q25_ohm": np.nanquantile(magnitude, 0.25, axis=(0, 2)),
            "magnitude_q75_ohm": np.nanquantile(magnitude, 0.75, axis=(0, 2)),
        }
    )
    by_frequency.to_csv(folder / "frequency_quality.csv", index=False)
    patterns = pd.DataFrame(
        {
            "pattern_index_0based": np.arange(z.shape[2]),
            "nonfinite_percent": 100 * (~finite).mean(axis=(0, 1)),
        }
    )
    if dataset.patterns is not None:
        patterns[["I1", "I2", "V1", "V2"]] = dataset.patterns
    patterns.to_csv(folder / "pattern_quality.csv", index=False)
    rows = []
    for mask in config["masks"]:
        if mask == "All":
            keep = np.ones(z.shape[2], bool)
        elif dataset.patterns is not None:
            coordinates = np.ascontiguousarray(dataset.patterns, dtype=np.int64)
            keep = _geometry(mask, coordinates.shape, coordinates.tobytes())
        else:
            continue
        for j, frequency in enumerate(dataset.frequencies):
            rows.append(
                dict(
                    mask=mask,
                    frequency_hz=frequency,
                    n_patterns=int(keep.sum()),
                    nonfinite_percent=(
                        100 * (~finite[:, j, keep]).mean() if keep.any() else np.nan
                    ),
                )
            )
    mask_quality = pd.DataFrame(rows)
    mask_quality.to_csv(folder / "mask_frequency_quality.csv", index=False)
    heat = mask_quality.pivot(
        index="mask", columns="frequency_hz", values="nonfinite_percent"
    )
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    image = ax.imshow(heat, aspect="auto", vmin=0, vmax=100)
    ax.grid(False)
    ax.set(
        yticks=range(len(heat)),
        yticklabels=heat.index,
        xlabel="Ordered frequency index",
        title="Missingness by geometric IIVV mask",
    )
    fig.colorbar(image, ax=ax, label="Non-finite coordinates (%)")
    save_figure(fig, folder / "mask-frequency-quality")
    retention = []
    for task in config["tasks"]:
        print(
            f"QUALITY: {task}: retention, PCA/MI and planned patient partitions",
            flush=True,
        )
        indices, y, classes = dataset.task(task)
        part = table.iloc[indices].copy()
        part["class_name"] = [classes[k] for k in y]
        part.to_csv(folder / f"{task}-inclusion.csv", index=False)
        label_audit = table.copy()
        label_audit["task_class"] = task_labels(table, task)
        label_audit["included"] = label_audit.task_class.isin(classes)
        label_audit.to_csv(folder / f"{task}-label-audit.csv", index=False)
        for cutoff in config["data_quality"]["audit_cutoffs"]:
            for label, group in part.groupby("class_name"):
                retained = group[group.missing_percent < cutoff]
                retention.append(
                    dict(
                        task=task,
                        class_name=label,
                        cutoff=cutoff,
                        total_sites=len(group),
                        retained_sites=len(retained),
                        total_patients=group.patient_id.nunique(),
                        retained_patients=retained.patient_id.nunique(),
                    )
                )
        # MI has no sample-weight argument. Repeated one-site-per-patient draws
        # show how much its descriptive ranking depends on repeated sites.
        x = transformed[indices]
        observed = np.isfinite(x).any(0)
        if observed.sum() < 2:
            raise ValueError(
                f"{task}: fewer than two observed frequency summaries for PCA/MI"
            )
        medians = np.nanmedian(x[:, observed], axis=0)
        filled = np.where(np.isfinite(x[:, observed]), x[:, observed], medians)
        mi = np.full(x.shape[1], np.nan)
        mi[observed] = mutual_info_classif(filled, y, random_state=config["seed"])
        rng = np.random.default_rng(config["seed"])
        patient_sites = list(part.groupby("patient_id", sort=True).indices.values())
        draws = []
        for draw in range(config["data_quality"]["mi_patient_draws"]):
            chosen = np.array([rng.choice(sites) for sites in patient_sites])
            if (
                len(np.unique(y[chosen])) != len(classes)
                or min(np.bincount(y[chosen])) < 2
            ):
                continue
            values = mutual_info_classif(
                filled[chosen], y[chosen], random_state=config["seed"] + draw
            )
            for j, value in zip(np.flatnonzero(observed), values):
                draws.append(
                    dict(
                        draw=draw,
                        frequency_hz=dataset.frequencies[j],
                        mutual_information=value,
                    )
                )
        draw_frame = pd.DataFrame(
            draws, columns=["draw", "frequency_hz", "mutual_information"]
        )
        draw_frame.to_csv(folder / f"{task}-mi-patient-draws.csv", index=False)
        write_json(
            folder / f"{task}-mi-analysis.json",
            dict(
                requested_patient_draws=config["data_quality"]["mi_patient_draws"],
                valid_patient_draws=int(draw_frame.draw.nunique()),
                excluded_draws="Missing class or fewer than two patient draws for a class",
                scope="Exploratory frequency summaries; no inference or classifier input",
            ),
        )
        ranks = pd.DataFrame(
            {"frequency_hz": dataset.frequencies, "mutual_information_site": mi}
        )
        if not draw_frame.empty:
            descriptive = draw_frame.groupby("frequency_hz").mutual_information.agg(
                ["mean", "std", "count"]
            )
            ranks = ranks.join(
                descriptive.add_prefix("patient_draw_"), on="frequency_hz"
            )
        ranks["rank_site"] = ranks.mutual_information_site.rank(
            ascending=False, method="min"
        )
        ranks.to_csv(folder / f"{task}-mutual-information.csv", index=False)
        ordered = ranks.sort_values("rank_site")
        fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
        ax.bar(range(len(ordered)), ordered.mutual_information_site, label="All sites")
        if "patient_draw_mean" in ordered:
            ax.plot(
                range(len(ordered)),
                ordered.patient_draw_mean,
                "o",
                markersize=3,
                label="Mean of one-site-per-patient draws",
            )
        ax.set(
            xticks=range(len(ordered)),
            xticklabels=[f"{value:g}" for value in ordered.frequency_hz],
            xlabel="Frequency (Hz), ordered by exploratory site MI",
            ylabel="Mutual information",
            title=TASK_DEFINITIONS[task]["title"],
        )
        plt.setp(ax.get_xticklabels(), rotation=70, ha="right")
        ax.legend(fontsize=8)
        save_figure(fig, folder / f"{task}-mutual-information", classes)
        save_projection(
            x,
            part,
            dataset.frequencies,
            folder / f"{task}-pca",
            part.class_name,
            "Unsupervised descriptive PCA",
        )
        if config["data_quality"]["mi_guided_pca"]:
            best = np.flatnonzero(observed)[
                np.argsort(-mi[observed], kind="stable")[
                    : config["data_quality"]["mi_top_frequencies"]
                ]
            ]
            save_projection(
                x[:, best],
                part,
                dataset.frequencies[best],
                folder / f"{task}-mi-pca",
                part.class_name,
                "Exploratory PCA after label-informed MI selection",
            )
        splits = []
        for repeat, fold, tr, te, evaluation in task_outer_splits(
            dataset, task, config
        ):
            groups = part.patient_id.to_numpy()
            seed = config["seed"] + repeat * 1000 + fold * 100
            inner = group_splits(y[tr], groups[tr], config["inner_folds"], seed + 1)
            splits.append(
                make_split_record(
                    dataset, task, indices, tr, te, inner, repeat, fold, evaluation
                )
            )
        task_folder = Path(output) / "planned-splits" / task
        task_folder.mkdir(parents=True, exist_ok=True)
        write_json(task_folder / "splits.json", splits)
        export_cv_demographics(task_folder, dataset, task, config)
    retention = pd.DataFrame(retention)
    retention.to_csv(folder / "task_quality_retention.csv", index=False)
    for task in config["tasks"]:
        part = retention[retention.task == task]
        fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
        for label, group in part.groupby("class_name"):
            ax.plot(
                group.cutoff,
                group.retained_sites / group.total_sites,
                "o-",
                label=label,
                color=class_color(label),
            )
        ax.set(
            xlabel="Strict acquisition-rejection cutoff (%)",
            ylabel="Retained fraction of sites",
            ylim=(0, 1.05),
            title=TASK_DEFINITIONS[task]["title"],
        )
        ax.legend(fontsize=8)
        save_figure(fig, folder / f"{task}-quality-retention", part.class_name)
    overlap = table.groupby("patient_id").agg(
        n_sites=("sample_id", "size"),
        n_pathologies=("pathology", "nunique"),
        n_raw_pathologies=("raw_pathology", "nunique"),
    )
    overlap.to_csv(folder / "patient_pathology_overlap.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].scatter(
        site.missing_percent, site.nonfinite_coordinate_percent, s=15, alpha=0.65
    )
    axes[0].set(
        xlabel="Mean acquisition rejection (%)",
        ylabel="Non-finite averaged coordinates (%)",
    )
    axes[1].semilogx(by_frequency.frequency_hz, by_frequency.median_magnitude_ohm)
    axes[1].fill_between(
        by_frequency.frequency_hz,
        by_frequency.magnitude_q25_ohm,
        by_frequency.magnitude_q75_ohm,
        alpha=0.25,
    )
    axes[1].set(xlabel="Frequency (Hz)", ylabel="Magnitude median and IQR (ohm)")
    save_figure(fig, folder / "quality-overview")
    icc_path = Path(output) / "repeatability/icc_and_inter_site_variability.csv"
    if icc_path.is_file():
        icc = pd.read_csv(icc_path)
        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        positions = np.arange(len(icc))
        ax.vlines(
            positions,
            icc.icc_ci_low,
            icc.icc_ci_high,
            colors=[class_color(v) for v in icc.class_name],
        )
        ax.scatter(
            positions,
            icc.icc_2_1_absolute,
            color=[class_color(v) for v in icc.class_name],
        )
        ax.set(
            xticks=positions,
            xticklabels=icc.class_name,
            ylabel="ICC(2,1), absolute agreement",
            title="Saved test means: ICC and patient-bootstrap 95% intervals",
        )
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
        save_figure(fig, folder / "saved-test-mean-icc")
    write_json(
        folder / "interpretation.json",
        dict(
            PCA="Descriptive whole-cohort patient-balanced projection, not cross-validated discrimination. No projection, loading, MI rank or imputation estimate is consumed by tuning.",
            MI="Exploratory site-level MI with repeated one-site-per-patient sensitivity summaries; no inferential p-values or confidence intervals. These draws are not a patient endpoint.",
            acquisition="Acquisition rejection percentages and non-finite averaged spectrum coordinates are distinct endpoints.",
            overlap="Multiple pathologies in a patient do not establish mixed histology within one probe footprint.",
            audit_cutoffs="Descriptive retention only; do not change the classifier search using these plots without acknowledging cohort-informed design.",
        ),
    )


def freeze_quality(output, dataset, config, source_hashes):
    root = Path(output)
    # Preserve the code/environment that actually produced the descriptive evidence.
    execution = root / "run_manifest.json"
    provenance = json.loads(execution.read_text()) if execution.is_file() else {}
    write_json(
        root / "quality" / "execution-provenance.json",
        {
            key: provenance.get(key)
            for key in (
                "started_utc",
                "source_hashes",
                "versions",
                "python",
                "git_revision",
                "cpu_affinity",
                "thread_environment",
                "config",
                "input_summary",
            )
        },
    )
    if (root / "source-at-start").is_dir():
        shutil.copytree(
            root / "source-at-start",
            root / "quality/source-at-start",
            dirs_exist_ok=True,
        )
    if (root / "environment-freeze.txt").is_file():
        shutil.copyfile(
            root / "environment-freeze.txt", root / "quality/environment-freeze.txt"
        )
    folders = ("quality", "repeatability", "exploration", "planned-splits")
    artifacts = {
        str(p.relative_to(root)): digest(p)
        for name in folders
        for p in sorted((root / name).rglob("*"))
        if p.is_file()
    }
    write_json(
        root / "quality_manifest.json",
        dict(
            status="completed",
            synthetic=bool(provenance.get("synthetic", False)),
            input_hashes=dataset.input_hashes,
            source_input_hashes=source_hashes,
            protocol=quality_protocol(config),
            artifact_hashes=artifacts,
        ),
    )


def verify_quality(root, config, input_hashes=None, source_hashes=None):
    root = Path(root)
    manifest = json.loads((root / "quality_manifest.json").read_text())
    if manifest["status"] != "completed" or manifest["protocol"] != quality_protocol(
        config
    ):
        raise ValueError(
            "Quality analysis protocol differs; run a fresh quality analysis first"
        )
    if input_hashes is not None and manifest["input_hashes"] != input_hashes:
        raise ValueError("Quality analysis uses different input data")
    if input_hashes is not None and manifest["source_input_hashes"] != source_hashes:
        raise ValueError("Quality analysis uses different MATLAB/source provenance")
    required = [
        "quality/clinical_metadata_status.json",
        "exploration/interpretation.json",
    ]
    required += [f"planned-splits/{task}/splits.json" for task in config["tasks"]]
    if not set(required) <= set(manifest["artifact_hashes"]):
        raise ValueError("Quality analysis lacks required artifacts")
    for name, expected in manifest["artifact_hashes"].items():
        path = Path(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not (root / path).is_file()
            or digest(root / path) != expected
        ):
            raise ValueError(
                f"Quality analysis artifact missing, changed or unsafe: {name}"
            )
    return manifest
