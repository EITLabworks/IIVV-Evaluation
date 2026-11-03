# Revision workflow

Run scientific computations on the server. The notebooks read saved artifacts;
they do not train models or rerun PCA, MI, or bootstrap analyses. Completion
checks recalculate saved metric/test consistency.

## 1. Transfer the updated code and preserve old results

- [ ] Commit/push the prepared changes locally, then update the server checkout
  while no old process is using files that will be replaced. Alternatively create
  a separate checkout for this protocol. Do not mix results from the old code.
- [ ] Keep `data/`, `source/` and existing `results/` as research inputs/provenance.
  They are ignored by Git and are not removed by this cleanup.
- [ ] Confirm the MATLAB file, NPZ files and ordered IIVV pattern file are present.
- [ ] Reserve the CPUs assigned to the job. Defaults below use **224–255**;
  replace `CPUSET=224-255` with `CPUSET=0-23` for your 24-CPU allocation.

## 2. Create or update the environment

On the server, in the new repository checkout:

```bash
conda env create -f environment.yml
source ./setup_environment.sh
```

For an existing environment, replace the creation command with:

```bash
conda env update -f environment.yml --prune
source ./setup_environment.sh
```

- [ ] Select `iivv-revision` for later notebooks. All packages come from
  `environment.yml`; no requirements text file is used.

## 3. Calculate data-quality and exploratory evidence first

Start a persistent session, then run the commands inside it:

```bash
tmux new -s iivv-quality
cd /home/jt292/IIVV-Evaluation
source ./setup_environment.sh
CPUSET=224-255 ./run_all.sh --data-quality-only --output results/revision-quality-v3
```

Detach with Ctrl-b, d. Reattach using `tmux attach -t iivv-quality`.

- [ ] Wait for `DATA QUALITY READY: .../quality_manifest.json`.
- [ ] Inspect `logs/revision-latest.log` and `quality_manifest.json`.
- [ ] Reconcile enrollment counts with the exported patient inventory using clinical
  records; the export cannot explain unrecorded enrollment exclusions.
- [ ] Check source/export agreement, available metadata, exclusions and class
  composition. Inspect the 2D PCA, exploratory MI-guided PCA, spectral summaries,
  missingness, retention and test-mean repeatability.
- [ ] Inspect planned fold demographics. All sites belonging to a patient must
  remain in one fold; the quality and tuning stages share the split builder.
- [ ] Do not choose classifier features using a whole-cohort PCA or MI plot.

You may copy this completed directory to the PC before tuning:

```bash
mkdir -p results
scp -r jt292@excalibur.iae.ief.uni-rostock.de:/home/jt292/IIVV-Evaluation/results/revision-quality-v3 results/
IIVV_QUALITY=results/revision-quality-v3 jupyter lab Data-Quality.ipynb
```

Task 3 uses the requested label “Low-grade dysplasia” for normal/benign tissue and
mild dysplasia. Its captions must retain that explanation. Task 3 excludes the
clinically healthy sites; binary Task 2 includes them in Low risk.

## 4. Run all three tasks, all tuning, and all reports

In a second persistent server session:

```bash
tmux new -s iivv-three-tasks
cd /home/jt292/IIVV-Evaluation
source ./setup_environment.sh
CPUSET=224-255 ./run_all.sh --quality-results results/revision-quality-v3 --output results/revision-three-tasks-v3
```

- [ ] The launcher first runs the regression tests **on the server**, then checks
  source/quality provenance, schedules 15 outer jobs, fits nested candidates,
  creates all reports, verifies outputs and writes the parameter-explorer HTML.
- [ ] Watch `tail -f logs/revision-latest.log`. Progress includes stage, task,
  outer/inner fold, candidate count, current model and elapsed time. Completed
  folds have `fold-checkpoints/<task>/<fold>/complete.json`.
- [ ] `Revision-Snapshot.ipynb` can display verified completed folds while the
  run continues; it deliberately withholds complete-partition inference.
- [ ] Wait for `REVISION RESULTS READY: results/revision-three-tasks-v3`.
- [ ] Check `unavailable_results.csv`: convergence failures are reported, not
  silently treated as successful model comparisons.

Use a new result directory after any scientific configuration or task change.
To resume an interrupted/failed run, use its original output directory and
unchanged configuration/inputs:

```bash
CPUSET=224-255 ./run_all.sh --quality-results results/revision-quality-v3 --output results/revision-three-tasks-v3 --resume
```

An active or completed run cannot be resumed. Stop only your intended job before
requesting resume; do not start two writers in the same result directory. A run
left with status `running` after an uncatchable termination needs its termination
confirmed before its manifest is marked interrupted. Do not alter a live manifest.

## 5. Copy every result to the PC

From the PC repository root, after completion:

```bash
mkdir -p results logs
scp -r jt292@excalibur.iae.ief.uni-rostock.de:/home/jt292/IIVV-Evaluation/results/revision-quality-v3 results/
scp -r jt292@excalibur.iae.ief.uni-rostock.de:/home/jt292/IIVV-Evaluation/results/revision-three-tasks-v3 results/
scp jt292@excalibur.iae.ief.uni-rostock.de:/home/jt292/IIVV-Evaluation/logs/revision-three-tasks-v3-\*.log ./logs/
```

- [ ] Copy the whole directory, including manifests, source-at-start/resume,
  quality evidence and `fold-checkpoints/`. Do not select only figures/metrics.
- [ ] Keep original server logs; CSV timing exports provide measured runtimes.

## 6. Inspect final results and selected combinations

```bash
source ./setup_environment.sh
python -m revision.report_cli check --results-dir results/revision-three-tasks-v3
IIVV_RESULTS=results/revision-three-tasks-v3 jupyter lab Figures-and-Tables.ipynb
```

- [ ] Inspect AUC, accuracy, F1, precision, SN, SP, PPV, NPV, coverage and confidence
  intervals; compare site and patient endpoints without treating them as identical.
- [ ] Inspect per-class ROC/confusion errors, especially high-grade dysplasia,
  sensitivity/specificity trade-offs and quality-related abstentions.
- [ ] Inspect `review_selected_parameters.csv` and task `selections.json` for the
  chosen configuration in each outer fold. There is no single cross-fold “best”
  configuration selected using test scores.
- [ ] Inspect `inner_search.csv` for every evaluated configuration and its inner
  score. These scores are selection evidence, not held-out performance.
- [ ] Open `parameter-explorer.html` in a browser for interactive filters over
  task, model, objective, fold, parameter combinations, endpoint and metrics.
- [ ] Inspect corrected paired CV tests/Holm adjustment, matched covariate controls,
  demographic splits, selected coordinate identities, stability and timing.

## 7. Complete paper and reviewer responses together

```bash
IIVV_RESULTS=results/revision-three-tasks-v3 IIVV_QUALITY=results/revision-quality-v3 jupyter lab manuscript/Reviewer-Evidence.ipynb
```

- [ ] Follow Reviewer → Concern → response rationale → saved evidence.
- [ ] Resolve missing clinical records only from actual records. Code cannot
  establish pathology blinding, registration or controlled probe pressure/angle.
- [ ] Treat the omitted RFE and resistance/reactance comparisons as unresolved
  experimental requests. The reduced run compares PCA/MI and magnitude/phase;
  it does not prove omitted alternatives inferior. Justify that scope explicitly
  or plan a separate controlled comparison before claiming those requests resolved.
- [ ] Replace pending result rows and claims only after verification of this
  three-task run. Never relabel the old four-task/fine-tuning numbers.
- [ ] Keep `main.tex` unchanged. Put removed text in `\edits{}` and added text in
  `\edit{}` in `Revision-1.tex`. Quote the exact associated changes in the response.
- [ ] Apply `reviewing_ppt.txt`: clear US scientific English, precise established
  terms, unchanged mathematics and no unsupported safety/significance claims.
- [ ] Retain the Fig. 9 workflow figure; insert newly measured timings when available.
- [ ] Build documents only when their sources are ready:

```bash
make manuscript
```

Paper/response files, document assets, templates and `manuscript.mk` are local-only
and ignored by Git. Copy them separately if building elsewhere. The clean revision
is generated from the marked source; do not edit it independently.
