# Code changes for the three-task revision

The only active protocol is `configs/revision.json` (schema 3). [PROTOCOL.md](PROTOCOL.md)
records class definitions, the retained 5 × 4 grouped nested CV protocol, exact
search settings, each omitted parameter and the limits on interpretation.

The data-quality stage precedes tuning and hashes source inputs, artifacts and
planned splits. Global descriptive PCA/MI projections are never consumed by
classifier selection. The same partition-inventory helper serves quality and
training. Each training fold independently learns imputation, scaling, frequency
ranking and data-dependent masks; outer-training OOF predictions determine
calibration and operating thresholds.

The former four-class task and prior-winner refinement mode are retired. Task 2
alone uses risk labels. Task 3 is lesion categorization and excludes clinically
healthy sites. Canonical pathology and raw labels are retained; legacy NPZ
`labels_MC` is checked as source metadata, not treated as an active task.

Reports preserve prior metrics, patient-bootstrap intervals, corrected paired CV
tests, demographic splits, covariate controls, parameter identities, failures,
coverage and timing. Rejected/nonconverged streams remain visibly unavailable.
Training-feature/SVM-Gram caches stay within exact partitions. The launcher
respects CPU affinity, emits progress and verifies complete output.

Retired files are listed in [CLEANUP-REMOVED.txt](CLEANUP-REMOVED.txt). Research
inputs/results and original manuscript/reviewer material are preserved. Use
[todo.md](todo.md) for execution. New evaluations and scientific regression checks
have not been run locally; the server launcher runs the prepared checks.

## Input schema

`data/` contains one trusted NPZ per measurement site. Required keys are
`Zdata` (frequency × IIVV, complex values and non-finite entries preserved),
`patient_no`, `patient_sample`, `LesionAssessment` (raw diagnosis), and
`Avg_Cutpats` (finite acquisition rejection percentage in [0,100]).
`frequencies_hz` is optional only for the declared 31-bin 100–100000 Hz grid;
other shapes require explicit coordinates. `source/IIVV_patterns.txt` contains
the corresponding ordered integer injection/voltage quadruples. Coordinate
order is never inferred from a feature count.

Optional NPZ clinical scalars or a `patient_id,site_id` keyed metadata CSV supply
available covariates. Metadata cannot replace outcome, quality or identity fields.
Optional `labels_MC` is checked against canonical raw pathology solely for legacy
export consistency; it does not create a four-class task. Pickled object arrays
are refused unless explicitly trusted. Duplicate site identities are rejected.

Repeats may be provided as `Zrepeats` (frequency × IIVV × repeat) in every NPZ,
or as a separate site × frequency × IIVV × repeat NPY with a mandatory sample-ID
index CSV. The analysis preserves the supplied repeat units and does not invent
burst-level recordings. The server launcher verifies/extracts MATLAB metadata
and available test means into `results/revision-source-v3`; older sidecars remain
in place. Extraction cache acceptance requires matching MATLAB, NPZ, sidecar and
producer-script hashes. Every run archives input hashes and its exact code/config.

Worker processes start with `spawn` to avoid inheriting native thread-pool locks
after serial numerical work. Each pool shares one temporary, read-only memory-mapped
dataset. This adds startup I/O and requires temporary space for the uncompressed
input arrays; `TMPDIR` can point to a fast scratch directory. Checkpoints, seeds and
training/test separation are unchanged. Regression logs show each test name and
automatically dump Python stacks after a test exceeds five minutes.
