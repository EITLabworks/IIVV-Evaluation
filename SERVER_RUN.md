# Revision evaluation on the server

Protocol chosen by the author on 20 October 2026: **5 outer × 4 inner
patient folds, one outer partition**, seed 1729, four tasks, three classifiers,
four objectives. Use `configs/revision.json`. This produces **20 outer
folds** and 48 tuned-versus-baseline comparisons per metric. Corrected CV tests
use Holm adjustment over **288 unique hypotheses** (six distinct metrics;
precision and PPV are aliases). Whole-patient bootstrap intervals are computed
separately from frozen outer predictions, combined once per site.

No project code, tests, fitting, result imports or LaTeX builds were executed
during this preparation. Run the following stages yourself, in order. Stop
on failure and retain the output/logs for review.

## 1. Transfer and environment

Copy this working tree, including its uncommitted changes, `data/`, `source/`
and `manuscript/` to the server. Those three directories are ignored by Git;
a clone alone does not contain the study inputs or paper. Preserve the complete
manuscript figure/template/bibliography trees. Keep earlier results separately.

Create/reuse and activate the Conda Python 3.11 environment from the repository root
(Bash or Zsh):

```sh
source ./setup_environment.sh
mkdir -p logs
```

For a fresh environment, the manual equivalent is `conda env create --file
environment.yml`, then `conda activate iivv-revision`. The YAML includes notebooks
and tests, sets noninteractive plotting and limits BLAS/OpenMP/MKL to one thread.
The helper reuses existing environments; update packages explicitly with
`conda env update --name iivv-revision --file environment.yml --prune` when needed.

The reference workflow uses CPU scikit-learn. `fold_workers=20` runs up to 20
independent outer folds, with `rf_jobs=4`, `blas_threads=1` and an 8 GiB feature
cache per fold. This targets the supplied 128-core/2 TiB server: up to 80 RF
threads and 160 GiB of feature caches, plus raw inputs and temporary arrays.
A planning reservation is 80 CPU cores and 384 GiB RAM; this is not a measured
peak. Concurrency is capped by CPU affinity, common scheduler allocations and
cgroup CPU quotas. On smaller jobs, reduce `fold_workers` and/or
`feature_cache_bytes` for the allocated RAM. No GPU is required.

Fold workers share immutable raw arrays through fork; fitted state remains
isolated per fold. After fitting workers exit, `task_workers=4` builds reports.
No two workers write the same checkpoint. The parent assembles completed folds
in the original deterministic order. `parallel_execution.json` records the
actual workers/schedule; `fold-checkpoints/` retains hash-verified complete folds
for resume. Keep this directory when transferring results or running checks.
Ctrl-C or a worker exception stops the worker pool; only complete folds are reused.
For platforms without fork, set both `fold_workers` and `task_workers` to 1.

The expanded search includes all 31 frequency counts, 13 geometry masks,
five impedance quantiles, four selectors, five representations, four quality
cutoffs, and the full model grids. It retains up to four frequency counts and
four geometry masks plus an impedance mask for joint tuning. Hundreds of thousands of inner
fits are possible before cache/duplicate reuse. A trustworthy wall-time estimate
needs server timing; do not reduce the protocol based on observed outer scores.

## 2. Software checks and input audits

```sh
make revision-tests > logs/revision-tests.log 2>&1
make revision-preflight > logs/revision-preflight.log 2>&1
```

Run the second command only if the tests pass. Preflight verifies/extracts the
MATLAB data and available metadata/repositioned test means, audits quality and
repeatability, and checks every requested patient partition. It does not fit
classifiers, but it does compute audit statistics and repeatability bootstraps.
Its manifest must say `preflight_passed`. Inspect:

- `results/revision-expanded-preflight/analysis_status.json`
- `results/revision-expanded-preflight/quality/clinical_metadata_status.json`
- `results/revision-expanded-preflight/quality/source_quality_audit.json`
- `results/revision-expanded-preflight/repeatability/status.json`

The source file must exist; `run_all.sh` now fails if it is absent. Sidecar
hashes are checked before cached extraction is reused. Do not run preflight and
evaluation simultaneously against the shared source-extraction directory.

## 3. Full evaluation

Start in a persistent terminal or your server's batch scheduler:

```sh
make revision-evaluation > logs/revision-expanded.log 2>&1
```

Default output: `results/revision-expanded`. Use a fresh directory for this
revised protocol; the old `revision-clinical-final` checkpoints have different
fold settings and source history. The earlier ten-repeat `revision-expanded`
run also has incompatible settings: changing the config does not change an
already-running process, and its checkpoints cannot be resumed with one repeat.
For the replacement run, use `REVISION_OUTPUT=results/revision-parallel`
consistently for evaluation, resume and check. If rerunning preflight, use
`PREFLIGHT_OUTPUT=results/revision-parallel-preflight` for a fresh audit.
Reporting notebooks accept `IIVV_RESULTS=results/revision-parallel`.
Runs made before the training-provenance/statistics changes require a fresh output,
even with the same fold configuration; older checkpoints cannot be resumed by this code.
The runner writes complete-fold checkpoints,
exact splits, fitted coordinate identities, failed-candidate details, source
snapshots, input hashes and environment inventory. It generates numerical
reports and figures automatically, including matched EIS-only/clinical/combined
controls. Obsolete numbered fitting entry points have been removed.
The optimized profile changes execution settings, so use a fresh directory rather
than resuming earlier task-only runs. Example:

```sh
make revision-evaluation REVISION_OUTPUT=results/revision-parallel > logs/revision-parallel.log 2>&1
```

The log reports fold start/completion and every 250 candidates, rather than two
lines per candidate. Set `verbose_candidates: true` in the config for detailed
candidate timings. Every candidate's scores/failures remain archived regardless
of log verbosity.

After an interruption explicitly recorded as `interrupted` or `failed`:

```sh
make revision-resume > logs/revision-expanded-resume.log 2>&1
```

Resume requires identical inputs and configuration. A `running` manifest after
an uncatchable scheduler kill needs investigation before its status is changed.
Do not resume this full experiment using the older clinical profile. For a
different fresh output name, supply `REVISION_OUTPUT=results/NAME` consistently
to evaluation, resume and check targets.

### Inspect a copied snapshot while the server continues

Use `Revision-Snapshot.ipynb` for the transferred `quality/`, `repeatability/`,
manifest/execution files and completed `fold-checkpoints/` folders. On the local host:

```sh
source ./setup_environment.sh
IIVV_SNAPSHOT=results/revision-server-snapshot jupyter lab Revision-Snapshot.ipynb
```

Run cells in order. Completion hashes, patient boundaries and fitting provenance
are checked before fold-derived tables are used. Saved cohort, quality and
test-mean repeatability audits can already support reviewer responses. Derived
CSVs and PDF/PNG figures go to `results/revision-snapshot-analysis/<snapshot-hash>/`.
There is no fitting, threshold tuning, bootstrap recomputation, aggregate classifier
performance or significance testing on partial folds. Copying only the quickest
folds is not a valid overall performance estimate. Recopy new completed folds and
rerun to refresh the evidence. If the run was resumed, also copy its
`resume-parent-*.json` manifests for checkpoint source-provenance verification.
Keep the final-result notebook completion guards and the original run status.

### Diagnose slow folds without fitting

Copy `scripts/diagnose_revision.py` to the server checkout without restarting the
run or changing its seed/configuration. This standard-library tool only reads
logs and hash-checks saved complete-fold artifacts; it imports no evaluation
modules, fits no models, and writes no results:

```sh
python scripts/diagnose_revision.py \
  --results-dir results/revision-expanded \
  --log logs/revision-expanded.log \
  --folds MC:0:2 healthy_vs_cancer:0:4
```

The final output lists measured complete-fold wall times (minimum/median/maximum),
last logged candidate counts, run age and thread settings. Saved outer-fit times
and warnings describe those configurations, not the inner search. Fits reused
across objectives are counted once. Incomplete-fold start times and timed candidate
progress are absent, so remaining time cannot be reliably calculated. A copied
log's modification time is not a fold-progress timestamp.

Inspect the current candidate without another fit by finding active worker PIDs:

```sh
ps -u "$USER" -o pid,ppid,stat,pcpu,etime,time,comm --sort=-pcpu
```

If `py-spy` is available, replace `12345` with an active worker PID, then take two
stack dumps a few minutes apart, or inspect its live profile:

```sh
py-spy dump --pid 12345 --locals
py-spy top --pid 12345 --native
```

`run_task`/`inner_evaluate`/`fit_predict` frames can expose the task, fold and
candidate. `_libsvm` suggests SVM optimization, `_lbfgsb` LR, tree-building functions
RF, and preprocessing/least-squares frames feature preparation. Repeated samples
in one function suggest an expensive step, not proof of a deadlock. Attaching
does not launch training, but stack collection can briefly pause the process.
Linux may require administrator permission to attach. See the
[py-spy documentation](https://github.com/benfred/py-spy#dump).

SVM currently has no explicit iteration cap; classifier LR has `max_iter=5000`.
Inner convergence warnings are caught but not retained in the inner-search
export. `candidate_failures.json` records exceptions, not convergence warnings;
an empty failure list therefore does not rule out slow/nonconverged fits.
Changing the global seed changes patient partitions and prevents combining those
folds with the existing 18; changing solver limits changes the protocol.

Optional server checks of this diagnostic tool only (no model imports/fits):

```sh
python -m unittest discover -s tests -p test_diagnose_revision.py
```

## 4. Verify and return the results

```sh
make revision-check > logs/revision-expanded-check.log 2>&1
```

Success requires `run_manifest.json` to say `completed`, with `synthetic: false`.
The checker verifies input hashes, all 20 outer folds, patient separation,
inner partitions, coverage, class/probability order, matched controls and the
complete Holm family. Candidate failures and unavailable streams must still be
reviewed; a completed run does not establish clinical adequacy.

Return the logs and the complete `results/revision-expanded/` tree for our next
review, subject to the study's existing data-access conditions. Especially useful:

- `run_manifest.json`, `resolved_config.json`, `analysis_status.json`
- `parallel_execution.json`, complete `fold-checkpoints/` tree
- `classification_outputs.csv`: AUC/F1/precision/SN/SP/PPV/NPV with patient-bootstrap intervals
- `corrected_resampled_tests_holm.csv`: paired CV tests and full-family adjusted p values
- `all_metrics.csv`, `paired_comparisons_holm.csv` (the latter is supplementary conditional prediction permutation)
- task-level `per_class_metrics.csv`, `split_population_counts.csv`,
  `candidate_failures.json`, `unavailable_streams.json`
- task-level `outer_fold_metrics.csv`, `cv_metrics_summary.csv`, `cv_comparison_fold_differences.csv`
- task-level `cv_population_summary.csv`, `cv_pathology_breakdown.csv`,
  `cv_demographics_*.csv`, `cv_patient_groups.csv`, `task_label_audit.csv`
- `quality/`, `repeatability/`, and generated plots

Corrected tests use SE² = (1/n_folds + mean(n_test_patients/n_train_patients)) ×
the sample variance of paired fold differences, with n_folds−1 t degrees of freedom.
The patient-grouped/nonlinear-metric application is approximate; five folds provide
limited information about variation between training partitions. Undefined folds or
constant nonzero differences produce unavailable p values, retained in the family.
The method follows [Bouckaert & Frank, sections 3.2–3.3](https://ml.cms.waikato.ac.nz/publications/2004/bouckaert-frank.pdf).
Confidence intervals use 1,000 whole-patient bootstrap draws, not independent-fold
standard errors. Tests condition on common quality coverage. Precision/PPV share
one hypothesis; the CSV also includes within-metric Holm values.

## 5. Paper builds and later numerical imports

TeX Live with `latexmk`, BibTeX, and the paper's packages is required. These
commands build the current drafts without running an evaluation or importing
new numerical results:

```sh
make revision revision-clean response
# Optional: build the unchanged original paper too
make paper
```

`manuscript/Paper/Revision-1-clean.tex` is a small entry point which reads
`Revision-1.tex`, renders additions normally and suppresses deletions, including
nested edits. Its PDF is `Revision-1-clean.pdf`. Both manuscript sources are
preserved. The tracked `templates/Revision-1-clean.tex` recreates the entry
point when needed. The clean paper contains the current draft's content; it is
not a claim that pending revision results are already incorporated.

**Defer the following until we have reviewed the completed run together:**

```sh
python scripts/update_manuscript_results.py --results results/revision-expanded
python scripts/update_abstract_results.py --results results/revision-expanded
python scripts/update_response_results.py --results results/revision-expanded
python scripts/create_rebuttal_screenshots.py
make revision-clean response
python scripts/validate_revision.py --results results/revision-expanded --require-completed
```

The importers reject incomplete/synthetic evidence and now describe the actual
expanded configuration. Existing generated manuscript blocks have preservation
guards: if an audit differs from the earlier draft, stop and review the marked
replacement with us instead of overwriting the existing block.

## Source evidence checked without running project code

The MATLAB v7.3 file's HDF5 headers list these 14 `filtered_cases` fields:
`Avg_Cutpats`, `Bx`, `Case`, `Category`, `Full_data`, `HPV`,
`LesionAssessment`, `LesionLoc`, `PriorRad`, `PriorSurgery`, `Sample`,
`Tgrade`, `Zdata`, `encoder`. The two lesion-info CSVs contain location,
pathology and label encodings; they add no contact or pathology protocol records.

`Full_data` contains test-level cells; an inspected cell is 3 × 2, with a
complex spectrum of shape 31 × 7728 and a rejection percentage per test.
The historical conversion notebook, preserved in Git history, describes these as
three tests averaged into `Zdata`/`Avg_Cutpats`; it supplies no within-test burst arrays.
Existing extraction handles all available two-to-four test means.

No named source fields document sectioning, footprint registration error,
mixed-histology adjudication, pathology blinding, pressure/angle, age, sex or
smoking/alcohol. Those reviewer subcomments require author records or an explicit
limitation. `HPV` and `Tgrade` are present as MATLAB `string` objects backed by
MCOS serialization; their integer payloads must not be interpreted as text.
They are not asserted absent or silently used as predictors. A native MATLAB
export would be needed to inspect their actual clinical values reliably.

## Optional notebooks

Activate `iivv-revision` and use its Python kernel; notebook packages are included
in `environment.yml`. `Revision-Evaluation.ipynb` exposes the canonical Make stages with
all execution switches disabled by default. `mat-to-npz.ipynb` reads the preflight
source/quality/repeatability audits. `Figures-and-Tables.ipynb` reads verified
completed results for every reviewer analysis. These notebooks contain no saved
results and do not implement a second fitting pipeline. The reporting notebooks
do not modify manuscripts or result files.

Reporting commands are now `python -m revision.report_cli check|evaluation|tuning`;
the numbered/print compatibility scripts and interim configs were removed.
