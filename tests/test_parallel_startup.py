"""Process-startup checks only: no model fitting, predictions or evaluation."""

import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from revision.data import Dataset
from revision.parallel import _execute


def inspect_shared_input(job, config):
    from revision.parallel import _DATASET

    return dict(
        index=job,
        value=float(_DATASET.z[job].real.sum()),
        pid=os.getpid(),
        start_method=mp.get_start_method(),
        mapped=isinstance(_DATASET.z, np.memmap),
        readonly=not _DATASET.z.flags.writeable,
        repeats_readonly=not _DATASET.repeats.flags.writeable,
        backing_file=str(getattr(_DATASET.z, "filename", "")),
    )


def test_fresh_workers_share_readonly_input_after_parent_numerical_work():
    z = np.arange(24).reshape(4, 2, 3).astype(complex)
    dataset = Dataset(
        z,
        pd.DataFrame({"patient_id": ["a", "b", "c", "d"]}),
        np.array([100, 1000]),
        None,
        True,
        {},
        z[..., None],
    )
    original = z.copy()
    config = {"blas_threads": 2}
    # Exercise the serial-then-process sequence that previously used unsafe fork.
    serial = _execute(inspect_shared_input, [0, 1], 1, dataset, config)
    with threadpool_limits(limits=2):
        np.eye(16) @ np.eye(16)
    parallel = _execute(inspect_shared_input, [0, 1], 2, dataset, config)
    assert sorted(row["value"] for row in parallel) == [row["value"] for row in serial]
    assert all(
        row["pid"] != os.getpid() and row["start_method"] == "spawn" for row in parallel
    )
    assert all(
        row["mapped"] and row["readonly"] and row["repeats_readonly"]
        for row in parallel
    )
    assert len({row["backing_file"] for row in parallel}) == 1
    assert all(not Path(row["backing_file"]).exists() for row in parallel)
    np.testing.assert_array_equal(dataset.z, original)
    assert dataset.z.flags.writeable
