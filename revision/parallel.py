"""Independent outer folds, isolated checkpoints and deterministic task assembly."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import signal
import time

import pandas as pd
from threadpoolctl import threadpool_limits

from .data import digest
from .pipeline import run_task, task_outer_splits, write_json

_DATASET = None
_THREAD_LIMIT = None
_JSON_PARTS = ("splits.json", "selections.json", "candidate_failures.json")
_CSV_PARTS = ("outer_predictions.csv", "inner_search.csv")
_IDENTICAL = ("task_inclusion.csv", "task_label_audit.csv", "classes.json")


def worker_count(requested, jobs, threads=1):
    """Respect affinity and common scheduler/cgroup CPU quotas, not host CPU count."""
    cpus = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count() or 1
    )
    for variable in ("SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS"):
        value = os.environ.get(variable, "")
        if value.isdigit() and int(value) > 0:
            cpus = min(cpus, int(value))
    root = Path("/sys/fs/cgroup")
    quotas = {root / "cpu.max"}
    membership = Path("/proc/self/cgroup")
    if membership.is_file():
        for line in membership.read_text().splitlines():
            if line.startswith("0::"):
                folder = root / line.split("::", 1)[1].lstrip("/")
                quotas.update(
                    parent / "cpu.max"
                    for parent in [folder, *folder.parents]
                    if parent == root or root in parent.parents
                )
    for quota in quotas:
        if quota.is_file():
            values = quota.read_text().split()
            if len(values) == 2 and values[0] != "max":
                cpus = min(cpus, max(1, int(values[0]) // int(values[1])))
    return max(1, min(int(requested), int(jobs), max(1, cpus // int(threads))))


def _initialize(dataset, blas_threads):
    global _DATASET, _THREAD_LIMIT
    _DATASET = dataset
    _DATASET.z.flags.writeable = False
    if _DATASET.repeats is not None:
        _DATASET.repeats.flags.writeable = False
    _THREAD_LIMIT = threadpool_limits(limits=blas_threads)
    if mp.current_process().name != "MainProcess":
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # parent stops the whole pool


def _execute(function, jobs, workers, dataset, config):
    """Fail fast and stop running children when any job fails or Ctrl-C arrives."""
    if workers == 1:
        with threadpool_limits(limits=config.get("blas_threads", 1)):
            global _DATASET
            _DATASET = dataset
            return [function(job, config) for job in jobs]
    if "fork" not in mp.get_all_start_methods():
        raise ValueError(
            "Parallel outer folds require fork; set fold_workers=1 on this platform"
        )
    previous_children = {child.pid for child in mp.active_children()}
    pool = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("fork"),
        initializer=_initialize,
        initargs=(dataset, config.get("blas_threads", 1)),
    )
    futures = []
    try:
        futures = [pool.submit(function, job, config) for job in jobs]
        results = [future.result() for future in as_completed(futures)]
    except BaseException:
        for future in futures:
            future.cancel()
        # Public multiprocessing API: only this coordinator creates child processes.
        for child in mp.active_children():
            if child.pid not in previous_children:
                child.terminate()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
        return results


def fold_directory(root, task, key):
    repeat, fold, evaluation = key
    return (
        Path(root)
        / "fold-checkpoints"
        / task
        / f"{evaluation}_repeat{repeat}_fold{fold}"
    )


def _check_complete(folder, task, key):
    marker = folder / "complete.json"
    if not marker.is_file():
        return False
    value = json.loads(marker.read_text())
    if value.get("task") != task or value.get("fold_key") != list(key):
        raise ValueError(f"Incorrect fold checkpoint identity: {folder}")
    inventory = value.get("artifacts", {})
    if not set((*_JSON_PARTS, *_CSV_PARTS, *_IDENTICAL)) <= set(inventory):
        raise ValueError(f"Incomplete fold checkpoint inventory: {folder}")
    for name, expected in inventory.items():
        path = folder / name
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Changed completed fold checkpoint: {path}")
    return True


def _fit_fold(job, config):
    task, key, root = job
    folder = fold_directory(root, task, key)
    if config.get("_resume") and _check_complete(folder, task, key):
        print(f"{task}/{folder.name}: checkpoint reused", flush=True)
        return task, key
    started = time.perf_counter()
    fold_config = {**config, "_fold_key": key}
    predictions, selections = run_task(
        _DATASET, task, fold_config, root, taskdir=folder
    )
    from .covariate_control import add_matched_control

    add_matched_control(
        _DATASET, task, fold_config, root, predictions, selections, taskdir=folder
    )
    names = [*_JSON_PARTS, *_CSV_PARTS, *_IDENTICAL]
    if (folder / "matched_covariate_control.json").is_file():
        names.append("matched_covariate_control.json")
    names += [
        str(path.relative_to(folder))
        for path in sorted((folder / "models").glob("*.joblib"))
    ]
    write_json(
        folder / "complete.json",
        dict(
            task=task,
            fold_key=list(key),
            elapsed_seconds=time.perf_counter() - started,
            artifacts={name: digest(folder / name) for name in names},
            source_version=config.get("_source_version"),
        ),
    )
    print(
        f"{task}/{folder.name}: checkpoint ready ({(time.perf_counter()-started)/3600:.2f} h)",
        flush=True,
    )
    return task, key


def _write_csv(path, frame):
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def merge_task(root, task, keys):
    """Only the parent writes task aggregates; fold directories remain resumable."""
    target = Path(root) / task
    target.mkdir(parents=True, exist_ok=True)
    folders = [fold_directory(root, task, key) for key in keys]
    for key, folder in zip(keys, folders):
        if not _check_complete(folder, task, key):
            raise ValueError(f"Missing completed outer fold: {folder}")
    for name in _IDENTICAL:
        expected = digest(folders[0] / name)
        if any(digest(folder / name) != expected for folder in folders[1:]):
            raise ValueError(f"Fold input/class inventories disagree: {task}/{name}")
        temporary = (target / name).with_suffix(Path(name).suffix + ".tmp")
        shutil.copyfile(folders[0] / name, temporary)
        temporary.replace(target / name)
    for name in _JSON_PARTS:
        write_json(
            target / name,
            [
                row
                for folder in folders
                for row in json.loads((folder / name).read_text())
            ],
        )
    for name in _CSV_PARTS:
        frames = [
            pd.read_csv(
                folder / name,
                float_precision="round_trip",
                dtype={"patient_id": str, "sample_id": str},
            )
            for folder in folders
        ]
        _write_csv(target / name, pd.concat(frames, ignore_index=True))
    controls = [
        json.loads((folder / "matched_covariate_control.json").read_text())
        for folder in folders
        if (folder / "matched_covariate_control.json").is_file()
    ]
    if controls:
        write_json(
            target / "matched_covariate_control.json",
            {
                **controls[0],
                "n_outer_folds": sum(row["n_outer_folds"] for row in controls),
                "newly_fitted_folds": [
                    item for row in controls for item in row["newly_fitted_folds"]
                ],
                "fold_control_provenance": [row["source_version"] for row in controls],
            },
        )
    for folder in folders:
        for source in (folder / "models").glob("*.joblib"):
            (target / "models").mkdir(exist_ok=True)
            shutil.copyfile(source, target / "models" / source.name)


def _report(job, config):
    task, root = job
    from .analysis import report_task

    folder = Path(root) / task
    predictions = pd.read_csv(
        folder / "outer_predictions.csv",
        float_precision="round_trip",
        dtype={"patient_id": str, "sample_id": str},
    )
    selections = json.loads((folder / "selections.json").read_text())
    report_task(_DATASET, task, predictions, selections, config, root)
    print(f"{task}: reports completed", flush=True)
    return task


def run_parallel(dataset, config, root):
    schedule = {
        task: [
            (repeat, fold, evaluation)
            for repeat, fold, _, _, evaluation in task_outer_splits(
                dataset, task, config
            )
        ]
        for task in config["tasks"]
    }
    jobs = [(task, key, str(root)) for task, keys in schedule.items() for key in keys]
    workers = worker_count(
        config.get("fold_workers", 1),
        len(jobs),
        max(config.get("rf_jobs", 4), config.get("blas_threads", 1)),
    )
    write_json(
        Path(root) / "parallel_execution.json",
        dict(
            mode="independent outer folds; fork-shared raw inputs; isolated fitted state",
            fold_workers=workers,
            requested_fold_workers=config.get("fold_workers", 1),
            rf_threads_per_fit=config.get("rf_jobs", 4),
            blas_threads=config.get("blas_threads", 1),
            feature_cache_bytes_per_fold=config.get("feature_cache_bytes"),
            fold_jobs=len(jobs),
            fold_order=[dict(task=task, key=key) for task, key, _ in jobs],
        ),
    )
    print(
        f"Running {len(jobs)} outer folds on {workers} workers; RF threads={config.get('rf_jobs', 4)}",
        flush=True,
    )
    _execute(_fit_fold, jobs, workers, dataset, config)
    for task, keys in schedule.items():
        merge_task(root, task, keys)
    # Reports start only after fitting workers exit, releasing their feature caches.
    report_workers = worker_count(
        config.get("task_workers", 1), len(schedule), config.get("blas_threads", 1)
    )
    _execute(
        _report,
        [(task, str(root)) for task in schedule],
        report_workers,
        dataset,
        config,
    )
    return (
        [pd.read_csv(Path(root) / task / "metrics_summary.csv") for task in schedule],
        [
            pd.read_csv(Path(root) / task / "prediction_permutation_comparisons.csv")
            for task in schedule
        ],
    )
