# IIVV evaluation

Code, notebooks, figures, and documents for the current oral EIS revision.
The verified export contains **276 recording sites from 100 patients**.

| Task | Classes and purpose | Sites / patients |
|---|---|---|
| 1: Healthy vs Cancer | Clinically healthy tissue versus invasive carcinoma | 196 / 98 |
| 2: High risk vs Low risk | High risk includes moderate/severe or high-grade dysplasia, CIS, and invasive cancer. Low risk includes healthy/normal tissue, benign tissue, and mild dysplasia | 272 / 100 |
| 3: Lesion categorization | Cancer contains invasive carcinoma. High-grade dysplasia includes moderate/severe dysplasia and CIS. The Low-grade dysplasia label includes normal/benign tissue and mild dysplasia. Clinically healthy sites are excluded | 172 / 97 |

Task 2 addresses whether a finding may warrant further evaluation or a procedure.
Task 3 exposes category errors hidden by binary pooling. These study labels do not
establish treatment rules. Four unmapped sites are excluded from classification.

## Workflow

1. **Prepare inputs and the environment.** Keep `data/`, `source/`, results, and logs.
   NPZ sites require `Zdata`, `patient_no`, `patient_sample`, `LesionAssessment`,
   and `Avg_Cutpats`. `source/IIVV_patterns.txt` defines coordinate order.
   The MATLAB source supplies available clinical metadata and processed test means.

   ```bash
   conda env create -f environment.yml
   # Existing environment: conda env update -f environment.yml --prune
   source ./setup_environment.sh
   ```

2. **Run quality analysis on the server and inspect it before tuning.**

   ```bash
   tmux new -s iivv-revision
   CPUSET=224-255 ./run_all.sh --data-quality-only --output results/revision-quality-v3
   ```

   Wait for `DATA QUALITY READY`. Inspect source agreement, cohort composition,
   missingness, retention, exploratory PCA/MI, repeatability, and fold populations
   in `Data-Quality.ipynb`. Global PCA/MI never supplies classifier features.

3. **Run tuning and reporting after the quality review.**

   ```bash
   CPUSET=224-255 ./run_all.sh --quality-results results/revision-quality-v3 --output results/revision-three-tasks-v3
   ```

   The launcher runs server regression checks, saves checkpoints, and verifies
   reports. Monitor `tail -f logs/revision-latest.log` until `REVISION RESULTS READY`.
   Detach tmux with Ctrl-b, d. Reattach with `tmux attach -t iivv-revision`.
   Default affinity is 224–255. Use `CPUSET=0-23` for 24 allocated logical CPUs.
   Use fresh output paths after scientific changes. Add `--resume` only for an
   interrupted/failed run with unchanged configuration and inputs. Never start
   two writers in one output directory. Retain diagnostic logs if tests stall.

4. **Copy whole run directories to the PC, then verify.** Replace the server/path.

   ```bash
   mkdir -p results logs
   scp -r USER@SERVER:/PATH/results/revision-quality-v3 results/
   scp -r USER@SERVER:/PATH/results/revision-three-tasks-v3 results/
   scp 'USER@SERVER:/PATH/logs/revision-three-tasks-v3-*.log' logs/
   python -m revision.report_cli check --results-dir results/revision-three-tasks-v3
   ```

5. **Inspect saved results and create figures.** Notebooks do not train models.

   | Notebook | Purpose |
   |---|---|
   | `Data-Quality.ipynb` | Cohort, quality, PCA/MI, repeatability, demographics |
   | `Revision-Evaluation.ipynb` | Commands and protocol |
   | `Revision-Snapshot.ipynb` | Completed-fold progress during an ongoing run |
   | `Figures-and-Tables.ipynb` | Final metrics, uncertainty, parameters, figures |
   | `manuscript/Reviewer-Evidence.ipynb` | Reviewer → concern → response → evidence |

   ```bash
   IIVV_RESULTS=results/revision-three-tasks-v3 jupyter lab Figures-and-Tables.ipynb
   python scripts/restyle_results.py
   python scripts/create_parameter_explorer.py --results-dir results/revision-three-tasks-v3 --output results/revision-palette/revision-three-tasks-v3/parameter-explorer.html
   PYTHONPATH=. python manuscript/prepare_revision_results.py
   ```

   `IIVV_RESULTS` and `IIVV_QUALITY` override notebook paths. Selected settings are in
   `review_selected_parameters.csv` and task `selections.json`. Candidate scores
   are in task `inner_search.csv`. The HTML explorer provides interactive filters.
   Restyled figures use `revision/plot_style.py` and live in
   `results/revision-palette/`, preserving the verified source artifacts.

6. **Review documents together, build, and package for Overleaf.**

   ```bash
   make manuscript
   python scripts/package_overleaf.py
   ```

   `main.tex` remains the original. `Revision-1.tex` marks additions with `\edit{}`
   and removals with `\edits{}`. The clean paper uses that same source. The response
   quotes exact associated changes. Upload `overleaf.zip` for the marked/clean paper
   and `response-letter.zip` as a separate project. Each includes compilation notes.

## Protocol, status, and remaining work

The current configuration expands the training-selected rejection cutoffs to
20, 40, 60, 80 and 100.000001 and adds an unnormalized reference. Existing v3
results use the previous search. The expanded search requires a new quality run
and a new evaluation output directory, for example `revision-quality-v4` and
`revision-three-tasks-v4`; do not resume or overwrite the v3 runs.

`configs/revision.json` defines five outer and four inner patient-grouped folds,
with seed 1729 and **15 outer computations**. Training partitions learn preprocessing,
frequency/IIVV selection, and classifiers. Inner validation selects settings and
families. Outer-training OOF predictions determine calibration and binary thresholds.
This is nested grouped CV, not leave-one-patient-out or independent validation.
`PROTOCOL.md` remains the detailed search specification required by the code.

Reports include AUC, accuracy, F1, precision/PPV, SN, SP, NPV, coverage, per-class
and patient endpoints, calibration, decision curves, covariate controls, and timings.
Intervals use 1000 whole-patient bootstrap resamples of frozen predictions. CV
comparisons use corrected resampled tests and global Holm adjustment (216 hypotheses).

The current quality and three-task runs are verified across all 15 folds. Primary
patient-equal site AUCs are 0.884, 0.798, and 0.794. No comparison passes global Holm
correction. Dysplasia sensitivity remains limited. Clinical safety and hardware
savings are not established. Before submission:

- [ ] Resolve Sophie placeholders for enrollment, pathology correspondence/blinding,
  mixed histology, healthy-site confirmation, contact controls, acquisition/test-mean
  reconciliation, and approved data sharing.
- [ ] Explain unperformed RFE/resistance-reactance comparisons and the limited
  quality-cutoff sweep. Omitted methods have not been shown inferior.
- [ ] Review both paper versions, responses, and upload packages. Independent clinical
  validation and prospective hardware/acquisition timing remain outside this study.

Private inputs, results, manuscript files, and upload ZIPs are ignored by Git and
need separate transfer. Active code/tests and current results are retained.
Superseded files are preserved in `archive/superseded-20261104.zip` with a
`CLEANUP.json` inventory. See `LICENSE` and `CITATION.cff` for reuse and citation.
