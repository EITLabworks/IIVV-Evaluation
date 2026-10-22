# IIVV Evaluation — reviewer revision

Patient-grouped nested evaluation of in vivo oral-lesion EIS. The author-selected
protocol uses **5 outer × 4 inner folds, one outer partition**, seed 1729, four diagnostic
tasks, three classifiers and four selection objectives. All learned preprocessing,
selection, calibration and thresholds stay inside outer-training patients.

Classification outputs report **AUC, F1, precision, SN, SP, PPV and NPV**, with
whole-patient bootstrap confidence intervals. Binary summaries use the positive
class; multiclass summaries use macro one-versus-rest metrics and separate class
tables. Precision equals PPV. Corrected paired outer-CV tests and detailed
patient/pathology/demographic split reports accompany the outputs.

Follow [SERVER_RUN.md](SERVER_RUN.md) for prerequisites, resources, ordered server
execution and result handoff. The computational implementation and reviewer mapping
are documented in [code-revisions.md](code-revisions.md).

## Server run

With Conda installed, create/reuse and activate the Python 3.11 environment
in your current Bash or Zsh shell:

```sh
source ./setup_environment.sh
```

The helper uses [environment.yml](environment.yml), which includes scientific,
test and notebook dependencies. Equivalent manual commands for a new environment:

```sh
conda env create --file environment.yml
conda activate iivv-revision
```

For an existing environment, the helper activates it without changing packages.
After editing the YAML, update explicitly with
`conda env update --name iivv-revision --file environment.yml --prune`.

Transfer the private `data/`, `source/` and `manuscript/` trees separately from Git.
Then run sequentially:

```sh
make revision-tests
make revision-preflight
make revision-evaluation
make revision-check
```

Results go to `results/revision-expanded`. An interrupted/failed run with identical
inputs and settings can use `make revision-resume`. Earlier results and source
snapshots remain separate. The runner never imports results into manuscripts.

The server profile runs up to **20 independent outer folds**, each with four RF
threads, then assembles task reports. CPU affinity, scheduler allocations and
cgroup CPU quotas limit concurrency automatically. Each fold has an 8 GiB feature
cache and an isolated, hash-verified checkpoint under `fold-checkpoints/`.
Use a fresh output after updating from the earlier task-only execution:

```sh
make revision-evaluation REVISION_OUTPUT=results/revision-parallel > logs/revision-parallel.log 2>&1
make revision-check REVISION_OUTPUT=results/revision-parallel
```

Run the check after evaluation succeeds. Set `IIVV_RESULTS=results/revision-parallel`
for reporting notebooks. Include `fold-checkpoints/` when copying results.

## Notebooks

JupyterLab and ipykernel are included in `iivv-revision`. After activation,
launch `jupyter lab` and select its Python kernel.
All committed notebooks have empty outputs and unexecuted cells.

| Notebook | Purpose |
|---|---|
| [Revision-Evaluation.ipynb](Revision-Evaluation.ipynb) | Canonical test/preflight/evaluation/check commands, with execution switches disabled by default |
| [Revision-Snapshot.ipynb](Revision-Snapshot.ipynb) | Ongoing-run snapshots: saved cohort/quality/repeatability evidence and hash-verified completed-fold partition audits; no fitting or aggregate classifier inference |
| [mat-to-npz.ipynb](mat-to-npz.ipynb) | Saved source/export alignment, centralized labels, metadata availability, quality, anatomy and test-mean repeatability audits |
| [Figures-and-Tables.ipynb](Figures-and-Tables.ipynb) | Completed outer results for all reviewer analyses, clinical metrics, dysplasia errors, ablations, uncertainty, matched controls, identities, timing and figures |

The reporting notebooks read existing artifacts without fitting models or
changing primary result files. Run preflight before the source notebook; run evaluation and
completion checks before the figure notebook. The historical conversion code and
notebook outputs remain available in Git history. Patient identifiers may appear
when notebooks are executed; keep those copies under the study's access conditions.

For copied outputs from an ongoing server run:

```sh
source ./setup_environment.sh
IIVV_SNAPSHOT=results/revision-server-snapshot jupyter lab Revision-Snapshot.ipynb
```

Run its cells in order. It verifies completion-marker hashes and writes derived CSV,
PNG and PDF files under `results/revision-snapshot-analysis/<snapshot-hash>/`.
Cohort audits cover their recorded source population; completed-fold tables remain
preliminary. No final AUC/F1/SN/SP/PPV/NPV, CV tests, bootstrap recomputation or
manuscript import is performed. Existing completed-run notebook guards remain intact.

## Result inspection

```sh
python -m revision.report_cli check --verify-inputs
python -m revision.report_cli evaluation
python -m revision.report_cli tuning --only freq --top 10
```

These commands default to `results/revision-expanded`; override `--results-dir`
when inspecting a different directory. Inner tuning scores are descriptive.

## Manuscripts

```sh
make revision revision-clean response
```

The clean PDF is `manuscript/Paper/Revision-1-clean.pdf`. Its tracked entry-point
template is in the local `templates/` directory; both original and marked manuscript sources are
preserved. Defer numerical manuscript/rebuttal imports until review of the completed
server outputs. See SERVER_RUN.md for those later commands.

Paper/response sources, templates, `manuscript.mk` and document-generation scripts
are local-only and ignored by Git. The shared Makefile loads `manuscript.mk` when
present; copy these local files separately if building documents on another host.

## Repository layout

- `run_all.sh`, `run_revision.sh`, `run_revision.py`: canonical execution.
- `configs/revision.json`: final expanded protocol.
- `environment.yml`, `setup_environment.sh`: Conda dependencies and activation.
- `revision/`: active data, geometry, fitting, metrics, reporting and verification.
- `scripts/`: source audits and guarded manuscript/rebuttal preparation.
- `tests/`: scientific regression and integration checks.

Obsolete numbered entry points, compatibility adapters, unused toolbox modules,
GPU dependency stack and interim profiles were removed. The live geometry module was
moved unchanged to `revision/geometry.py`. Historical figures/results and private
inputs were preserved. No evaluation, tests, notebooks or builds were executed
during cleanup; only static dependency/diff and notebook-file checks were made.

For citation metadata, see [CITATION.cff](CITATION.cff).
