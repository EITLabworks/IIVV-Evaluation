"""Opt-in diagnostic progress; never used for candidate selection."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time


def update_progress(config, phase, **fields):
    path = config.get("_progress_path")
    if not config.get("progress_interval_seconds") or path is None:
        return
    from .pipeline import write_json

    state = config.setdefault("_progress_state", {})
    if state.get("phase") != phase:
        state["phase_started_epoch"] = time.time()
    state.update(fields, phase=phase, pid=os.getpid())
    state["updated_epoch"] = time.time()
    state["updated_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(path, state)


def heartbeat(jobs, completed, total, *, since_epoch=0, label="folds"):
    """Read worker-owned snapshots; queued jobs have no snapshot yet."""
    active = []
    for job in jobs:
        if len(job) != 3:  # task-report jobs use their existing log lines
            continue
        task, key, root = job
        repeat, fold, evaluation = key
        path = (
            Path(root)
            / "fold-checkpoints"
            / task
            / f"{evaluation}_repeat{repeat}_fold{fold}"
            / "progress.json"
        )
        if not path.is_file():
            continue
        state = json.loads(path.read_text())
        if (
            state.get("phase") == "complete"
            or state.get("updated_epoch", 0) < since_epoch
        ):
            continue
        elapsed = max(0, time.time() - state.get("phase_started_epoch", time.time()))
        candidate = state.get("candidate", {})
        model = candidate.get("model", "-")
        selector = candidate.get("selector", "-")
        active.append(
            f"{task}/r{repeat}/f{fold}: {state['phase']} "
            f"candidate={state.get('candidates_completed', 0)}/{state.get('candidates_planned', '?')} "
            f"inner={state.get('inner_fold', '-')}, {elapsed:.0f}s "
            f"{model}/{selector}/n={candidate.get('n_freq') or 'all'}/{candidate.get('mask', '-')}"
        )
    print(
        f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] "
        f"PROGRESS {label}={completed}/{total}; " + " | ".join(active),
        flush=True,
    )
