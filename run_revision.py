#!/usr/bin/env python3
"""Describe EIS data, then run training-only nested selection for three tasks."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import traceback

# Writable cache and noninteractive plotting before importing the scientific stack.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "iivv-matplotlib")
)
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import pandas as pd
from revision.data import (
    load_dataset,
    digest,
    CLASS_NAMES,
    TASK_PROTOCOL,
    TASK_DEFINITIONS,
)
from revision.pipeline import write_json, run_task
from revision.analysis import quality_audit, repeatability_audit, report_task
from revision.metrics import holm


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--prepare-source",
        action="store_true",
        help="Verify/extract optional MATLAB source metadata and test means before fitting",
    )
    p.add_argument(
        "--source-mat", default="source/Python-Labeled-CaseData-17-Sep-2025.mat"
    )
    p.add_argument("--source-output", default="results/revision-source-v3")
    p.add_argument(
        "--data",
        default="data",
        help="One NPZ per site, preserving complex values and NaNs",
    )
    p.add_argument(
        "--patterns",
        default="source/IIVV_patterns.txt",
        help="Verified ordered IIVV coordinates",
    )
    p.add_argument(
        "--metadata", help="Optional site-keyed clinical CSV (patient_id,site_id)"
    )
    p.add_argument(
        "--repeats", help="Optional memory-mapped raw or test-mean repeat array"
    )
    p.add_argument(
        "--repeat-index", help="CSV with explicit repeat-array sample_id row order"
    )
    p.add_argument(
        "--label-map", help="Optional explicit raw-pathology-to-canonical-class JSON"
    )
    p.add_argument(
        "--config", default=str(Path(__file__).parent / "configs/revision.json")
    )
    p.add_argument(
        "--output",
        default="results/revision",
        help="Dedicated new output directory; existing result plots stay untouched",
    )
    p.add_argument("--tasks", nargs="+", choices=list(CLASS_NAMES))
    p.add_argument("--models", nargs="+", choices=["svm", "rf", "lr"])
    p.add_argument(
        "--objectives", nargs="+", choices=["auc", "acc", "f1", "sensitivity"]
    )
    p.add_argument("--outer-repeats", type=int)
    p.add_argument("--outer-folds", type=int)
    p.add_argument("--inner-folds", type=int)
    p.add_argument("--bootstrap-samples", type=int)
    p.add_argument(
        "--fold-workers",
        type=int,
        help="Parallel outer folds; 0 uses task-level execution",
    )
    p.add_argument(
        "--rf-jobs", type=int, help="Tree-training threads per random forest"
    )
    p.add_argument(
        "--blas-threads", type=int, help="BLAS/OpenMP threads per fold worker"
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume fully checkpointed folds with unchanged configuration/input hashes; source versions are archived",
    )
    p.add_argument(
        "--preflight",
        "--data-quality-only",
        dest="preflight",
        action="store_true",
        help="Run descriptive data-quality, PCA/MI and planned-fold analyses without diagnostic classifiers",
    )
    p.add_argument(
        "--quality-results",
        help="Completed matching quality-analysis directory, required for clinical tuning",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip figure creation (all numeric reports are still computed)",
    )
    p.add_argument(
        "--trust-object-arrays",
        action="store_true",
        help="Allow pickle only for locally trusted historical NPZ object arrays",
    )
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate an explicitly labeled smoke-test fixture; NEVER clinical results",
    )
    return p.parse_args(argv)


def synthetic_fixture(folder):
    """Deterministic integration fixture with repeated sites and deliberately missing data."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(311)
    frequencies = np.geomspace(100, 100000, 8)
    labels = ["healthy", "invasive carcinoma", "moderate dysplasia", "benign"]
    # Synthetic coordinates have a declared order; not a reconstruction of the study.
    patterns = np.array(
        [
            [26, 27, 1, 2],
            [26, 28, 1, 3],
            [27, 29, 2, 4],
            [28, 30, 3, 5],
            [29, 31, 4, 6],
            [30, 32, 5, 7],
        ]
    )
    np.savetxt(folder / "patterns.txt", patterns, fmt="%d")
    for patient in range(32):
        for site in range(2):
            k = patient % 4
            z = (200 + 60 * k + 20 * site) / (
                1 + (1j * frequencies[:, None] / (1000 + 200 * k)) ** 0.8
            )
            z = (
                np.broadcast_to(z, (8, 6)).copy()
                + rng.normal(0, 8, (8, 6))
                + 1j * rng.normal(0, 3, (8, 6))
            )
            missing = np.zeros(6, bool)
            if patient % 5 == 0:
                missing[0] = True
            z[:, missing] = np.nan
            repeats = z[..., None] * (1 + rng.normal(0, 0.02, (8, 6, 9)))
            np.savez(
                folder / f"sample_{patient:03d}_{site}.npz",
                Zdata=z,
                patient_no=str(patient),
                patient_sample=str(site),
                LesionAssessment=labels[k],
                Avg_Cutpats=100 * missing.mean(),
                frequencies_hz=frequencies,
                LesionLoc=["tongue", "cheek"][site],
                age=40 + patient,
                sex=["female", "male"][patient % 2],
                Zrepeats=repeats,
            )
    return folder, folder / "patterns.txt"


def runtime_manifest(args, config):
    versions = {}
    for name in [
        "numpy",
        "scipy",
        "scikit-learn",
        "pandas",
        "matplotlib",
        "Pillow",
        "joblib",
        "threadpoolctl",
        "h5py",
    ]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    root = Path(__file__).resolve().parent
    source_paths = [
        Path(__file__),
        *sorted((root / "revision").glob("*.py")),
        *sorted((root / "scripts").glob("*.py")),
        root / "run_all.sh",
        root / "scripts/parameter_explorer.html",
        root / "PROTOCOL.md",
        root / "environment.yml",
        root / "setup_environment.sh",
    ]
    try:
        git_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_revision = None
    return dict(
        status="running",
        reporting_protocol_version=1,
        started_utc=datetime.now(timezone.utc).isoformat(),
        arguments=vars(args),
        config=config,
        task_definitions={task: TASK_DEFINITIONS[task] for task in config["tasks"]},
        synthetic=args.synthetic,
        python=sys.version,
        platform=platform.platform(),
        processor=platform.processor(),
        cpu_count=os.cpu_count(),
        versions=versions,
        git_revision=git_revision,
        cpu_affinity=(
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None
        ),
        thread_environment={
            name: os.environ.get(name)
            for name in [
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "CPUSET",
            ]
        },
        source_hashes={str(p.relative_to(root)): digest(p) for p in source_paths},
        notes=[
            "CPU scikit-learn is the reference implementation; no GPU calibration differences.",
            "Inner searches select configurations; outer outcomes are used only for assessment.",
            "Patient bootstrap/permutation inference is conditional on fitted outer predictions, not full model refitting.",
            "CV significance uses approximate corrected resampled paired outer-fold tests; patient-bootstrap intervals remain separate.",
            "Clinical sensitivity/specificity targets are analysis assumptions, not validated safety requirements.",
        ],
    )


def validate_config(config):
    if (
        config.get("schema_version") != 3
        or config.get("task_protocol") != TASK_PROTOCOL
    ):
        raise ValueError(
            "Expected schema_version 3 and the confirmed three-task protocol; use a fresh output directory"
        )
    if config.get("fine_tuning"):
        raise ValueError(
            "Use the fresh search; prior-winner fine-tuning has been retired"
        )
    if not config.get("final_selection"):
        raise ValueError(
            "The current protocol requires the fresh staged selection settings"
        )
    quality = config.get("data_quality", {})
    if (
        not isinstance(quality.get("mi_patient_draws"), int)
        or quality["mi_patient_draws"] < 1
        or not isinstance(quality.get("mi_top_frequencies"), int)
        or quality["mi_top_frequencies"] < 2
        or not isinstance(quality.get("mi_guided_pca"), bool)
        or not quality.get("audit_cutoffs")
        or any(not 0 <= value <= 100.000001 for value in quality["audit_cutoffs"])
    ):
        raise ValueError("Invalid data_quality settings")
    if config.get("final_selection"):
        for key in ("svm_max_iter", "lr_max_iter"):
            value = config["final_selection"].get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"Invalid final-selection {key}")
    interval = config.get("progress_interval_seconds", 0)
    if not isinstance(interval, int) or interval < 0:
        raise ValueError("progress_interval_seconds must be a nonnegative integer")
    if config.get("fold_schedule", "task_order") not in {"task_order", "round_robin"}:
        raise ValueError("Unknown fold_schedule")
    counts = config.get("frequency_counts")
    if counts is not None and (
        not counts
        or len(set(counts)) != len(counts)
        or any(not isinstance(n, int) or n < 1 for n in counts)
    ):
        raise ValueError("frequency_counts must contain unique positive integers")
    if config.get("cache_svm_training_gram") and not set(
        config["model_grids"].get("svm", {}).get("kernel", ["rbf"])
    ) <= {"linear", "rbf"}:
        raise ValueError("SVM Gram caching requires linear/RBF kernels")
    for key in ["outer_folds", "inner_folds"]:
        if config[key] < 2:
            raise ValueError(f"{key} must be >=2")
    if int(config.get("task_workers", 1)) < 1:
        raise ValueError("task_workers must be positive")
    if int(config.get("fold_workers", 0)) < 0:
        raise ValueError("fold_workers must be >=0")
    for key in ("rf_jobs", "blas_threads"):
        if int(config.get(key, 1)) < 1:
            raise ValueError(f"{key} must be positive")
    for key in ["outer_repeats", "bootstrap_samples", "permutation_samples", "top_k"]:
        if config[key] < 1:
            raise ValueError(f"{key} must be >=1")
    for key in [
        "tasks",
        "models",
        "objectives",
        "masks",
        "cutoffs",
        "selectors",
        "representations",
    ]:
        if not config[key]:
            raise ValueError(f"{key} cannot be empty")
    if any(x < 0 or x > 100.000001 for x in config["cutoffs"]):
        raise ValueError("Cutoffs must be in [0,100.000001]")
    if not 0 < config["sensitivity_target"] <= 1:
        raise ValueError("sensitivity_target must be in (0,1]")
    from revision.features import MASKS

    choices = dict(
        tasks=set(CLASS_NAMES),
        models={"svm", "rf", "lr"},
        objectives={"auc", "acc", "f1", "sensitivity"},
        masks=set(MASKS),
        selectors={"pca", "mutual_info"},
        representations={
            "magnitude",
            "magnitude_phase",
        },
        normalizations={"freq", "global", "none"},
        imputations={"mean", "median", "zero"},
    )
    for name, allowed in choices.items():
        values = config[name]
        if not values or len(set(values)) != len(values) or not set(values) <= allowed:
            raise ValueError(f"Invalid, duplicate or empty {name}: {values}")
    screen = config.get("screen_defaults", {})
    allowed_screen = {"cutoff", "normalization", "imputation", "class_weight"}
    if not isinstance(screen, dict) or not set(screen) <= allowed_screen:
        raise ValueError("Invalid screen_defaults fields")
    for field, collection in {
        "cutoff": "cutoffs",
        "normalization": "normalizations",
        "imputation": "imputations",
        "class_weight": "class_weights",
    }.items():
        if field in screen and screen[field] not in config[collection]:
            raise ValueError(f"screen_defaults.{field} must occur in {collection}")
    if any(not 0 < q < 1 for q in config["z_quantiles"]):
        raise ValueError("z_quantiles must be in (0,1)")
    if any(
        not 0 < t < 1 for t in config["specificity_targets"] + config["prevalences"]
    ):
        raise ValueError("Specificity targets and prevalences must be in (0,1)")
    if config.get("external_cohort_column") and not config.get(
        "external_cohort_values"
    ):
        raise ValueError(
            "External cohort column requires explicit external_cohort_values"
        )
    for model in config["models"]:
        if not config["model_grids"].get(model):
            raise ValueError(f"Missing parameter grid for {model}")


def _fit_report_task(job, config):
    from revision.parallel import _DATASET

    task, output = job
    predictions, selections = run_task(_DATASET, task, config, output)
    from revision.covariate_control import add_matched_control

    predictions, selections = add_matched_control(
        _DATASET, task, config, output, predictions, selections
    )
    return report_task(_DATASET, task, predictions, selections, config, output)


def main(argv=None):
    args = parse_args(argv)
    if args.resume and args.preflight:
        raise ValueError(
            "--resume and --preflight cannot be combined; preflight uses a separate output directory"
        )
    if not args.synthetic and not args.preflight and not args.quality_results:
        raise ValueError(
            "Clinical tuning requires --quality-results from the completed data-quality stage"
        )
    config = json.loads(Path(args.config).read_text())
    for key in [
        "tasks",
        "models",
        "objectives",
        "outer_repeats",
        "outer_folds",
        "inner_folds",
        "bootstrap_samples",
        "fold_workers",
        "rf_jobs",
        "blas_threads",
    ]:
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    config["plots"] = not args.no_plots
    output = Path(args.output)
    if args.synthetic and output == Path("results/revision"):
        output = Path("results/revision-synthetic")
    if not args.synthetic and "synthetic" in output.name.lower():
        raise ValueError("Use a separate output folder for clinical and synthetic runs")
    output.mkdir(parents=True, exist_ok=True)
    # Refuse to overwrite prior results or mix incompatible configurations/synthetic data.
    previous_manifest = None
    if (output / "run_manifest.json").exists():
        old = json.loads((output / "run_manifest.json").read_text())
        if args.resume:
            if old.get("status") not in ["interrupted", "failed"]:
                raise ValueError(
                    "Resume requires an explicitly interrupted/failed run, never an active or completed run"
                )
            if old.get("synthetic") != args.synthetic:
                raise ValueError("Resume requires identical synthetic flag")
            previous_manifest = old
        else:
            raise ValueError(
                f"{output} already contains a run. Choose a new --output directory, or --resume an interrupted run."
            )
    elif args.resume:
        raise ValueError("Resume requires an existing run manifest")
    if args.synthetic:
        args.data, args.patterns = map(
            str,
            (
                (output / "synthetic-input", output / "synthetic-input/patterns.txt")
                if args.resume
                else synthetic_fixture(output / "synthetic-input")
            ),
        )
        # A small complete integration path; full clinical search stays in the checked-in config.
        config.update(
            outer_folds=3,
            inner_folds=2,
            outer_repeats=1,
            fold_workers=args.fold_workers if args.fold_workers is not None else 0,
            task_workers=1,
            top_k=1,
            frequency_counts=[1, 3, 8],
            models=args.models or ["lr"],
            objectives=args.objectives or ["auc", "sensitivity"],
            masks=["All"],
            z_quantiles=[0.5],
            selectors=["pca", "mutual_info"],
            representations=["magnitude", "magnitude_phase"],
            cutoffs=[60, 100.000001],
            normalizations=["freq"],
            imputations=["mean"],
            class_weights=[None],
            bootstrap_samples=args.bootstrap_samples or 10,
            permutation_samples=10,
            model_grids={
                "lr": {"C": [0.1, 1.0]},
                "svm": {"C": [0.1, 1.0], "kernel": ["linear", "rbf"]},
                "rf": {
                    "n_estimators": [20],
                    "max_depth": [4],
                    "max_features": ["sqrt"],
                },
            },
        )
        if config.get("final_selection"):
            config["screen_defaults"] = dict(
                cutoff=100.000001,
                normalization="freq",
                imputation="mean",
                class_weight=None,
            )
            config["model_grids"]["svm"]["max_iter"] = [
                config["final_selection"]["svm_max_iter"]
            ]
    validate_config(config)
    if config.get("progress_interval_seconds"):
        print(
            "PIPELINE 1/6: verify inputs, source metadata and patient partitions",
            flush=True,
        )
    if previous_manifest is not None and previous_manifest.get("config") != config:
        raise ValueError("Resume requires the identical resolved configuration")
    if previous_manifest is not None and previous_manifest.get("task_definitions") != {
        task: TASK_DEFINITIONS[task] for task in config["tasks"]
    }:
        raise ValueError("Task definitions changed; use a fresh run")
    if (
        previous_manifest is not None
        and previous_manifest.get("reporting_protocol_version") != 1
    ):
        raise ValueError(
            "Older checkpoints lack training-scope/statistical provenance; use a fresh output directory"
        )
    manifest = runtime_manifest(args, config)
    if previous_manifest is not None:
        # A failed resume must keep the inventory governing existing checkpoints.
        # Otherwise a second attempt could accept changed inputs from the failed
        # attempt and silently mix them with old folds.
        for field in ["input_hashes", "source_input_hashes"]:
            if field in previous_manifest:
                manifest[field] = previous_manifest[field]
        parent_file = (
            output
            / f'resume-parent-{len(list(output.glob("resume-parent-*.json")))+1:03d}.json'
        )
        write_json(parent_file, previous_manifest)
        manifest["resume_history"] = [
            *previous_manifest.get("resume_history", []),
            dict(
                manifest=parent_file.name,
                sha256=digest(parent_file),
                note="Only fully checkpointed outer folds reused; configuration and input hashes must match. Source versions may differ and are recorded explicitly.",
            ),
        ]
    # Archive exact scientific source for every execution, including resumed versions.
    archive = output / (
        "source-at-start"
        if previous_manifest is None
        else f'source-at-resume-{len(manifest["resume_history"]):03d}'
    )
    import shutil

    for name in manifest["source_hashes"]:
        source = Path(__file__).resolve().parent / name
        target = archive / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    shutil.copytree(
        Path(__file__).resolve().parent / "configs",
        archive / "configs",
        dirs_exist_ok=True,
    )
    write_json(output / "run_manifest.json", manifest)
    write_json(output / "resolved_config.json", config)
    try:
        if (
            args.prepare_source
            and not args.synthetic
            and not Path(args.source_mat).is_file()
        ):
            raise FileNotFoundError(
                f"--prepare-source requires the MATLAB source: {args.source_mat}. "
                "Supply --source-mat or use run_revision.py without --prepare-source for an explicitly reduced-input analysis."
            )
        if (
            args.prepare_source
            and not args.synthetic
            and Path(args.source_mat).is_file()
        ):
            source_dir = Path(args.source_output)
            source_manifest = source_dir / "source_extraction.json"
            verified = False
            if source_manifest.is_file():
                previous = json.loads(source_manifest.read_text())
                hashes = {
                    str(path): digest(path)
                    for path in sorted(Path(args.data).glob("*.npz"))
                }
                sidecars = previous.get("sidecar_hashes", {})
                verified = (
                    previous.get("mat_sha256") == digest(args.source_mat)
                    and previous.get("script_sha256")
                    == digest(
                        Path(__file__).parent / "scripts/prepare_revision_source.py"
                    )
                    and previous.get("npz_hashes") == hashes
                    and set(sidecars)
                    == {
                        "clinical_metadata.csv",
                        "repeat_index.csv",
                        "test_mean_repeats.npy",
                    }
                    and all(
                        (source_dir / name).is_file()
                        and digest(source_dir / name) == value
                        for name, value in sidecars.items()
                    )
                )
            if not verified:
                subprocess.run(
                    [
                        sys.executable,
                        str(
                            Path(__file__).parent / "scripts/prepare_revision_source.py"
                        ),
                        "--mat",
                        args.source_mat,
                        "--data",
                        args.data,
                        "--output",
                        args.source_output,
                    ],
                    check=True,
                )
            args.metadata = args.metadata or str(source_dir / "clinical_metadata.csv")
            args.repeats = args.repeats or str(source_dir / "test_mean_repeats.npy")
            args.repeat_index = args.repeat_index or str(
                source_dir / "repeat_index.csv"
            )
        if (
            args.prepare_source
            and not args.synthetic
            and Path(args.source_mat).is_file()
        ):
            from scripts.audit_source_quality import audit

            audit(args.source_mat, args.data, args.source_output, output / "quality")
        needs_patterns = any(m != "All" for m in config["masks"])
        # All-only analyses can run without geometry; full default search requires the verified coordinate file.
        patterns = (
            args.patterns if needs_patterns or Path(args.patterns).is_file() else None
        )
        dataset = load_dataset(
            args.data,
            patterns_path=patterns,
            metadata_path=args.metadata,
            label_map_path=args.label_map,
            frequencies=config.get("frequencies_hz"),
            trust_object_arrays=args.trust_object_arrays,
            repeats_path=args.repeats,
            repeats_index_path=args.repeat_index,
        )
        manifest["arguments"] = vars(args)
        source_hashes = None
        if args.prepare_source and not args.synthetic:
            source_hashes = {
                str(Path(args.source_mat)): digest(args.source_mat),
                str(source_manifest): digest(source_manifest),
            }
            if (
                previous_manifest is not None
                and previous_manifest.get("source_input_hashes") != source_hashes
            ):
                raise ValueError(
                    "Source input inventory changed or was not recorded by the older runner; use a fresh output directory"
                )
        if (
            previous_manifest is not None
            and previous_manifest.get("input_hashes") != dataset.input_hashes
        ):
            raise ValueError(
                "Input hashes changed; cannot reuse any old outer-fold results"
            )
        manifest["input_hashes"] = dataset.input_hashes
        if source_hashes is not None:
            manifest["source_input_hashes"] = source_hashes
        inventory = sorted(
            (d.metadata.get("Name", "unknown"), d.version)
            for d in importlib.metadata.distributions()
        )
        (output / "environment-freeze.txt").write_text(
            "\n".join(f"{name}=={version}" for name, version in inventory) + "\n"
        )
        manifest["input_summary"] = dict(
            n_sites=len(dataset.table),
            n_patients=dataset.table.patient_id.nunique(),
            n_frequencies=len(dataset.frequencies),
            n_patterns=dataset.z.shape[2],
            complex_available=dataset.complex_available,
            unmapped_pathologies=dataset.table.loc[
                dataset.table.pathology.isna(), "raw_pathology"
            ]
            .value_counts()
            .to_dict(),
        )
        # Persist the verified inventory before long fits, including interrupted runs.
        write_json(output / "run_manifest.json", manifest)
        # Validate all requested target populations before launching costly fits.
        from revision.pipeline import group_splits

        for task in config["tasks"]:
            ix, y, names = dataset.task(task)
            if len(np.unique(y)) != len(names):
                raise ValueError(f"{task}: absent target classes; expected {names}")
            taskgroups = dataset.table.iloc[ix].patient_id.to_numpy()
            if config.get("external_cohort_column"):
                col = config["external_cohort_column"]
                if col not in dataset.table:
                    raise ValueError(f"External cohort column {col} is absent")
                ext = (
                    dataset.table.iloc[ix][col]
                    .astype(str)
                    .isin(config["external_cohort_values"])
                    .to_numpy()
                )
                if set(taskgroups[ext]) & set(taskgroups[~ext]):
                    raise ValueError("Development and external patients overlap")
                y = y[~ext]
                taskgroups = taskgroups[~ext]
            for repeat in range(config["outer_repeats"]):
                splits = group_splits(
                    y, taskgroups, config["outer_folds"], config["seed"] + repeat * 1000
                )
                for fold, (tr, _) in enumerate(splits):
                    group_splits(
                        y[tr],
                        taskgroups[tr],
                        config["inner_folds"],
                        config["seed"] + repeat * 1000 + fold * 100 + 1,
                    )
            if config.get("external_cohort_column"):
                group_splits(y, taskgroups, config["inner_folds"], config["seed"] + 1)
        if config.get("progress_interval_seconds"):
            print(
                "PIPELINE 2/6: calculate or verify frozen data-quality evidence",
                flush=True,
            )
        from revision.data_quality import explore, freeze_quality, verify_quality

        if args.preflight or args.synthetic:
            quality_audit(dataset, config, output)
            repeatability_audit(
                dataset, output, config["bootstrap_samples"], config["seed"]
            )
            if args.preflight:
                explore(dataset, config, output)
                freeze_quality(output, dataset, config, source_hashes)
        else:
            if not args.quality_results:
                raise ValueError(
                    "Clinical tuning requires --quality-results from ./run_quality.sh"
                )
            quality_root = Path(args.quality_results)
            quality = verify_quality(
                quality_root, config, dataset.input_hashes, source_hashes
            )
            if quality.get("synthetic"):
                raise ValueError(
                    "Synthetic quality evidence cannot be used for clinical tuning"
                )
            quality_hash = digest(quality_root / "quality_manifest.json")
            if (
                previous_manifest is not None
                and previous_manifest.get("quality_manifest_sha256") != quality_hash
            ):
                raise ValueError("Resume requires the original frozen quality report")
            for name in ("quality", "repeatability", "exploration", "planned-splits"):
                shutil.copytree(quality_root / name, output / name, dirs_exist_ok=True)
            shutil.copyfile(
                quality_root / "quality_manifest.json", output / "quality_manifest.json"
            )
            manifest["quality_manifest_sha256"] = digest(
                output / "quality_manifest.json"
            )
        statuses = {
            "complex_feature_comparisons": (
                "available"
                if dataset.complex_available
                else "unavailable: original arrays are real-valued"
            ),
            "repeatability": (
                "available"
                if dataset.repeats is not None
                else "unavailable: Zrepeats absent"
            ),
            "external_validation": (
                "available"
                if config.get("external_cohort_column")
                else "not configured: no independent cohort assumed"
            ),
            "pathology_registration_blinding_mixed_sites": "requires clinical records; no procedure inferred",
            "prospective_acquisition_and_hardware_validation": "requires new measurements; only computation and coordinate requirements can be measured here",
            "requested_representations": config["representations"],
        }
        write_json(output / "analysis_status.json", statuses)
        if args.preflight:
            manifest["status"] = "quality_completed"
            write_json(output / "run_manifest.json", manifest)
            print(f"DATA QUALITY READY: {output}/quality_manifest.json")
            return 0
        all_summaries = []
        all_comparisons = []
        start = time.perf_counter()
        if config.get("progress_interval_seconds"):
            print(
                "PIPELINE 3/6: training-only screening -> joint tuning -> inner OOF calibration -> frozen outer testing",
                flush=True,
            )
        execution_config = {
            **config,
            "_resume": args.resume,
            "_source_version": dict(
                started_utc=manifest["started_utc"], hashes=manifest["source_hashes"]
            ),
            "_previous_source_version": (
                dict(
                    started_utc=previous_manifest["started_utc"],
                    hashes=previous_manifest["source_hashes"],
                )
                if previous_manifest
                else None
            ),
        }
        workers = min(int(config.get("task_workers", 1)), len(config["tasks"]))
        if config.get("fold_workers", 0) > 0:
            from revision.parallel import run_parallel

            all_summaries, all_comparisons = run_parallel(
                dataset, execution_config, output
            )
        elif workers > 1:
            from revision.parallel import _execute

            print(
                f"Running {workers} independent diagnostic tasks concurrently",
                flush=True,
            )
            for summary, comparison in _execute(
                _fit_report_task,
                [(task, str(output)) for task in config["tasks"]],
                workers,
                dataset,
                execution_config,
            ):
                all_summaries.append(summary)
                all_comparisons.append(comparison)
        else:
            for task in config["tasks"]:
                predictions, selections = run_task(
                    dataset, task, execution_config, output
                )
                from revision.covariate_control import add_matched_control

                predictions, selections = add_matched_control(
                    dataset, task, config, output, predictions, selections
                )
                summary, comparison = report_task(
                    dataset, task, predictions, selections, config, output
                )
                all_summaries.append(summary)
                all_comparisons.append(comparison)
        if config.get("progress_interval_seconds"):
            print(
                "PIPELINE 4/6: combine task metrics and global corrected CV/Holm statistics",
                flush=True,
            )
        summary = pd.concat(all_summaries, ignore_index=True)
        summary.to_csv(output / "all_metrics.csv", index=False)
        pd.concat(
            [
                pd.read_csv(output / task / "classification_outputs.csv")
                for task in config["tasks"]
            ],
            ignore_index=True,
        ).to_csv(output / "classification_outputs.csv", index=False)
        from revision.cv_statistics import combine_corrected_tests

        combine_corrected_tests(output, config["tasks"])
        # Familiar table names with explicit nested evaluation provenance and class semantics.
        summary[summary.stream.str.startswith("baseline_")].to_csv(
            output / "BaselineResults.csv", index=False
        )
        summary[~summary.stream.str.startswith("baseline_")].to_csv(
            output / "FinalResults.csv", index=False
        )
        comparisons = pd.concat(all_comparisons, ignore_index=True)
        if len(comparisons):
            comparisons["p_holm_all_tasks_models_objectives"] = holm(
                comparisons.p_value
            )
        comparisons.to_csv(output / "paired_comparisons_holm.csv", index=False)
        failures = []
        for task in config["tasks"]:
            failures.extend(
                json.loads((output / task / "candidate_failures.json").read_text())
            )
        statuses["candidate_failure_count"] = len(failures)
        statuses["candidate_failures_artifacts"] = [
            f"{task}/candidate_failures.json" for task in config["tasks"]
        ]
        statuses["unavailable_stream_count"] = sum(
            len(json.loads((output / task / "unavailable_streams.json").read_text()))
            for task in config["tasks"]
        )
        partial_tasks = []
        statuses["completed_task_populations"] = [
            t for t in config["tasks"] if t not in partial_tasks
        ]
        statuses["partial_task_populations"] = partial_tasks
        statuses["cv_statistics"] = {
            "test": "approximate Nadeau-Bengio corrected resampled paired outer-fold t test",
            "artifact": "corrected_resampled_tests_holm.csv",
            "multiple_testing": "Holm across all tasks/models/objectives and six unique metrics; PPV aliases precision",
            "confidence_intervals": "whole-patient bootstrap of frozen outer predictions",
            "supplementary": "paired_comparisons_holm.csv retains conditional patient-block prediction permutations",
        }
        statuses["real_patient_reanalysis_completed"] = (
            not args.synthetic and not partial_tasks
        )
        write_json(output / "analysis_status.json", statuses)
        manifest["matched_covariate_control_completed"] = all(
            (output / task / "matched_covariate_control.json").is_file()
            and json.loads(
                (output / task / "matched_covariate_control.json").read_text()
            ).get("status")
            == "completed"
            for task in config["tasks"]
        )
        if config.get("final_selection"):
            from revision.final_selection import annotate

            if config.get("progress_interval_seconds"):
                print(
                    "PIPELINE 5/6: export selected parameters, search budget and reviewer summary",
                    flush=True,
                )
            annotate(output, config)
            manifest["final_selection_protocol_sha256"] = digest(
                output / "final_selection_protocol.json"
            )
        manifest.update(
            status="completed_partial" if partial_tasks else "completed",
            elapsed_seconds=time.perf_counter() - start,
            finished_utc=datetime.now(timezone.utc).isoformat(),
            claim_scope=(
                "SYNTHETIC SOFTWARE VALIDATION ONLY"
                if args.synthetic
                else "Adaptive reduced search on the original cohort; not independent confirmation"
            ),
        )
        write_json(output / "run_manifest.json", manifest)
        print(
            f'Completed {len(config["tasks"])} task computations; status={manifest["status"]}: {output}'
        )
        return 0
    except KeyboardInterrupt:
        manifest.update(
            status="interrupted",
            error="Interrupted before completion; checkpoint files are not final results",
        )
        write_json(output / "run_manifest.json", manifest)
        return 130
    except Exception as exc:
        manifest.update(
            status="preflight_failed" if args.preflight else "failed", error=str(exc)
        )
        write_json(output / "run_manifest.json", manifest)
        (output / "failure_traceback.txt").write_text(traceback.format_exc())
        print(f"Revision workflow stopped: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
