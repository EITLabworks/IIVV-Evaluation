# Revision evaluation on the server

Protocol chosen by the author on 20 October 2026: **5 outer × 4 inner
patient folds, 10 outer repeats**, seed 1729, four tasks, three classifiers,
four objectives. Use `configs/revision.json`. This produces **200 outer
folds** and a prespecified **48-comparison** Holm family. Repeated predictions
are combined once per site before patient-cluster inference.

No project code, tests, fitting, result imports or LaTeX builds were executed
during this preparation. Run the following stages yourself, in order. Stop
on failure and retain the output/logs for review.

## 1. Transfer and environment

Copy this working tree, including its uncommitted changes, `data/`, `source/`
and `manuscript/` to the server. Those three directories are ignored by Git;
a clone alone does not contain the study inputs or paper. Preserve the complete
manuscript figure/template/bibliography trees. Keep earlier results separately.

Use Python 3.11 in a fresh environment. From the repository root:

```sh
python3.11 -m venv .venv-revision
. .venv-revision/bin/activate
python -m pip install -r requirements-revision.txt
mkdir -p logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg
```

The reference workflow uses CPU scikit-learn. The profile runs two tasks
concurrently, with four threads per RF fit and a 2 GiB feature cache per task.
Reserve at least 8 CPU cores; 64 GiB RAM is a conservative planning request,
not a measured peak-memory requirement. The raw repeat array and temporary
transform matrices also consume memory. No GPU is required.

The expanded search includes all 31 frequency counts, 13 geometry masks,
five impedance quantiles, six selectors, five representations, five quality
cutoffs, and the full model grids. It retains up to four frequency counts and
four geometry masks plus an impedance mask for joint tuning. Millions of inner
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
fold settings and source history. The runner writes complete-fold checkpoints,
exact splits, fitted coordinate identities, failed-candidate details, source
snapshots, input hashes and environment inventory. It generates numerical
reports and figures automatically, including matched EIS-only/clinical/combined
controls. Do not run the legacy numbered fitting scripts in sequence.

After an interruption explicitly recorded as `interrupted` or `failed`:

```sh
make revision-resume > logs/revision-expanded-resume.log 2>&1
```

Resume requires identical inputs and configuration. A `running` manifest after
an uncatchable scheduler kill needs investigation before its status is changed.
Do not resume this full experiment using the older clinical profile. For a
different fresh output name, supply `REVISION_OUTPUT=results/NAME` consistently
to evaluation, resume and check targets.

## 4. Verify and return the results

```sh
make revision-check > logs/revision-expanded-check.log 2>&1
```

Success requires `run_manifest.json` to say `completed`, with `synthetic: false`.
The checker verifies input hashes, all 200 outer folds, patient separation,
inner partitions, coverage, class/probability order, matched controls and the
complete Holm family. Candidate failures and unavailable streams must still be
reviewed; a completed run does not establish clinical adequacy.

Return the logs and the complete `results/revision-expanded/` tree for our next
review, subject to the study's existing data-access conditions. Especially useful:

- `run_manifest.json`, `resolved_config.json`, `analysis_status.json`
- `all_metrics.csv`, `paired_comparisons_holm.csv`
- task-level `per_class_metrics.csv`, `split_population_counts.csv`,
  `candidate_failures.json`, `unavailable_streams.json`
- `quality/`, `repeatability/`, and generated plots

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
The conversion notebook explicitly describes these as three tests averaged
into `Zdata`/`Avg_Cutpats`. It does not supply separate within-test burst arrays.
Existing extraction handles all available two-to-four test means.

No named source fields document sectioning, footprint registration error,
mixed-histology adjudication, pathology blinding, pressure/angle, age, sex or
smoking/alcohol. Those reviewer subcomments require author records or an explicit
limitation. `HPV` and `Tgrade` are present as MATLAB `string` objects backed by
MCOS serialization; their integer payloads must not be interpreted as text.
They are not asserted absent or silently used as predictors. A native MATLAB
export would be needed to inspect their actual clinical values reliably.
