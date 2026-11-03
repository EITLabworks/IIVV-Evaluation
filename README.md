# IIVV evaluation: three-task revision

The workflow describes the data first, then runs patient-grouped nested selection
for Healthy vs Cancer, binary High risk vs Low risk, and three-class lesion
categorization. Task 3 retains the previous lesion-only population; clinically
healthy sites are excluded. Its lower-grade class includes normal/benign tissue
and mild dysplasia. Exact definitions and search limits are in [PROTOCOL.md](PROTOCOL.md).

Start with the step-by-step [todo.md](todo.md). On the server:

```bash
conda env create -f environment.yml
source ./setup_environment.sh
./run_all.sh --data-quality-only --output results/revision-quality-v3
```

Review the saved quality evidence, then run every task/model/objective:

```bash
./run_all.sh --quality-results results/revision-quality-v3 --output results/revision-three-tasks-v3
```

The default affinity is logical CPUs 224–255. Override with `CPUSET=0-23` for
24 logical CPUs. Worker limits respect affinity and scheduler quotas. The launcher
runs regression checks on the server before tuning, keeps progress logs and
checkpoints, creates all reports and verifies the completed output. Use a fresh
output after scientific changes. Resume an interrupted/failed run by repeating its
command with `--resume`, identical inputs/configuration and the original output.

## Notebooks

All notebooks are English, unexecuted and read saved artifacts without training.

| Notebook | Purpose | Environment variable |
|---|---|---|
| [Data-Quality.ipynb](Data-Quality.ipynb) | Cohort, source quality, PCA/MI, missingness, retention, repeatability and planned CV demographics | `IIVV_QUALITY` |
| [Revision-Evaluation.ipynb](Revision-Evaluation.ipynb) | Exact quality/tuning/check commands and protocol | none |
| [Revision-Snapshot.ipynb](Revision-Snapshot.ipynb) | Hash-verified completed folds and saved progress from an ongoing run | `IIVV_RESULTS` |
| [Figures-and-Tables.ipynb](Figures-and-Tables.ipynb) | Final metrics, intervals, corrected tests, controls, selected parameters and figures | `IIVV_RESULTS` |
| `manuscript/Reviewer-Evidence.ipynb` | Reviewer/concern evidence and response drafts | both |

```bash
IIVV_QUALITY=results/revision-quality-v3 jupyter lab Data-Quality.ipynb
IIVV_RESULTS=results/revision-three-tasks-v3 jupyter lab Figures-and-Tables.ipynb
```

`review_selected_parameters.csv` contains each outer fold's training-selected
configuration. Task `inner_search.csv` files contain evaluated candidates;
`selections.json` includes full frequency/IIVV identities and postprocessing
provenance. There is no test-selected universal winning configuration. The
self-contained `parameter-explorer.html` supports interactive parameter filtering.

Reports retain AUC, accuracy, F1, precision, SN, SP, PPV, NPV, abstention/coverage,
per-class and patient endpoints, whole-patient bootstrap intervals, corrected
resampled CV tests, Holm correction, matched available-covariate controls and
measured timings. Global PCA/MI views are descriptive and never feed tuning.

## Repository and documents

- `run_all.sh`, `run_revision.py`: one quality/tuning entry point and orchestration.
- `configs/revision.json`: the only active scientific configuration.
- `revision/`: fitting, features, patient splits, quality, inference and verification.
- `scripts/`: MATLAB/source audits and the offline parameter explorer.
- `tests/`: prepared scientific and artifact-integrity regression checks.
- `environment.yml`, `setup_environment.sh`: Conda environment and activation.
- [CLEANUP-REMOVED.txt](CLEANUP-REMOVED.txt): retired entry points/configurations.

Raw inputs, old result directories and original reviewer comments are preserved.
Old four-task results are not compatible with this protocol. Clinical data and
local paper/response/build files remain ignored by Git. `main.tex` is the untouched
original. Revised sources use `\edit{}`/`\edits{}`; `make manuscript` builds the
original, marked/clean revision and response when local document files are present.
No paper numbers are imported automatically. See [CITATION.cff](CITATION.cff).
