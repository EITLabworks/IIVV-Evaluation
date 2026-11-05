# IIVV evaluation

Code, server workflow and study protocol for the oral electrical impedance
spectroscopy (EIS) revision. Protocol `three-tasks-v1-lesion-only-task3`, schema 3.
The verified export contains **276 recording sites from 100 patients**. New reports
use English while preserving recorded diagnoses and source identifiers.

## Tasks and populations

Class order below matches the exported probability columns.

| Task / code | Classes and composition | Sites / patients |
|---|---|---|
| 1: `healthy_vs_cancer` | Healthy; Cancer (invasive carcinoma) | 196 / 98 |
| 2: `highrisk_vs_lowrisk` | Low risk (healthy/normal, benign, mild dysplasia); High risk (moderate/severe or high-grade dysplasia, CIS, invasive carcinoma) | 272 / 100 |
| 3: `cancer_vs_lr_hr` | Cancer (invasive carcinoma); Low-grade dysplasia (normal/benign lesions and mild dysplasia); High-grade dysplasia (moderate/severe dysplasia and CIS). Clinically healthy sites are excluded | 172 / 97 |

Task 2 addresses whether a finding may warrant further evaluation or a procedure.
Task 3 separates invasive cancer from dysplasia and lower-grade lesions to expose
errors hidden by binary pooling. These disjoint study categories preserve the
recorded diagnoses and do not establish treatment rules. Four unmapped sites are
excluded explicitly. There is no four-class task. Patient counts overlap across
pathology groups and must not be added. Captions may refer to the class definitions
in Methods instead of repeating them.

## Run workflow

1. **Prepare inputs and environment.** Keep `data/`, `source/`, results and logs.
   Each site NPZ requires `Zdata`, `patient_no`, `patient_sample`,
   `LesionAssessment` and `Avg_Cutpats`. `source/IIVV_patterns.txt` defines IIVV
   coordinate order. MATLAB records supply available clinical metadata and
   processed test means.

   ```bash
   conda env create -f environment.yml
   # Existing environment: conda env update -f environment.yml --prune
   source ./setup_environment.sh
   ```

2. **Generate quality evidence before tuning.** The launcher extracts/verifies
   source sidecars and saves the analyses and planned partitions described below.

   ```bash
   tmux new -s iivv-revision
   CPUSET=224-255 make revision-quality
   ```

   Wait for `DATA QUALITY READY`, then inspect `Data-Quality.ipynb`. The default
   output is `results/revision-quality-v5`.

3. **Run tuning and reporting after reviewing quality.**

   ```bash
   CPUSET=224-255 make revision-evaluation
   tail -f logs/revision-latest.log
   ```

   The default output is `results/revision-three-tasks-v5`. The launcher runs
   regression checks, fits folds, merges verified checkpoints, generates reports,
   checks patient separation/provenance and creates the parameter explorer. Wait
   for `REVISION RESULTS READY`. Tests print their names, slow durations and
   Python stacks after five minutes. Detach tmux with Ctrl-b, d; reattach with
   `tmux attach -t iivv-revision`.

   Override paths with `QUALITY_OUTPUT=...` and `REVISION_OUTPUT=...`. For an
   interrupted run with unchanged configuration and inputs, use
   `make revision-resume` with the same paths. Use fresh paths after protocol or
   seed changes. Never start two writers in one output directory or update the
   checkout while a job is running. The code and this guide are archived as run
   provenance. A quality manifest must match the inputs and planned CV protocol.

4. **Copy complete run directories to the PC and verify.** Replace USER/SERVER/PATH.

   ```bash
   mkdir -p results logs
   scp -r USER@SERVER:/PATH/results/revision-quality-v5 results/
   scp -r USER@SERVER:/PATH/results/revision-three-tasks-v5 results/
   scp 'USER@SERVER:/PATH/logs/revision-three-tasks-v5-*.log' logs/
   python -m revision.report_cli check --results-dir results/revision-three-tasks-v5
   ```

5. **Inspect saved results.** Notebooks read existing artifacts rather than train
   models. Set `IIVV_RESULTS` and `IIVV_QUALITY` to the run being inspected.

   | Notebook | Purpose |
   |---|---|
   | `Data-Quality.ipynb` | Cohort, quality, PCA/MI, repeatability and demographics |
   | `Revision-Evaluation.ipynb` | Execution commands and protocol |
   | `Revision-Snapshot.ipynb` | Completed-fold progress during an ongoing run |
   | `Figures-and-Tables.ipynb` | Final metrics, uncertainty, parameters and figures |

   ```bash
   IIVV_RESULTS=results/revision-three-tasks-v5 IIVV_QUALITY=results/revision-quality-v5 jupyter lab
   python scripts/create_parameter_explorer.py --results-dir results/revision-three-tasks-v5
   ```

   Selected settings are in `review_selected_parameters.csv` and each task's
   `selections.json`; evaluated candidates are in `inner_search.csv`. Timings are
   in `final_search_budget.csv` and fold checkpoints. The HTML explorer filters
   configurations interactively. `revision/plot_style.py` defines shared colors;
   optional v3 restyling via `scripts/restyle_results.py` writes separate copies
   under `results/revision-palette/` and preserves verified originals.

6. **Build documents and package for Overleaf.** Current projects are
   `manuscript/Revision/` and `manuscript/Response_Letter/`.

   ```bash
   make manuscript
   python scripts/package_overleaf.py
   # Rebuild after source saves; Ctrl-C stops the watchers:
   make watch
   ```

   Additions use `\edit{}` and removals `\edits{}`. The clean paper uses the same
   source, and the response reproduces associated changes. Preserve the original
   manuscript. Watch mode refreshes the response's reference snapshot atomically
   after successful marked-paper builds and opens no PDF previews. Upload
   `overleaf.zip` and `response-letter.zip` as separate projects.

## Quality analysis

Quality outputs include PDF/PNG figures under `exploration/` and CSV/JSON records:

- Site/patient/raw-pathology populations, repeated sites, pathology overlap,
  class-by-location counts, available clinical variables and missing metadata.
- Source/export agreement, acquisition rejection percentages, non-finite averaged
  coordinates by frequency/IIVV geometry, and spectral magnitude medians/IQRs.
  Acquisition rejection and missingness after averaging are different quantities.
- Class-specific retention at strict rejection cutoffs 0, 10, 20, 30, 40, 50, 60,
  70, 80, 90 and 100.000001. Zero retains no sites; the last value includes recorded
  values of 100 because the rule is `rejection < cutoff`. These descriptive audit
  cutoffs do not choose or expand the classifier search.
- Saved-test-mean coefficients of variation and absolute-agreement ICC with
  whole-patient bootstrap intervals. Processed means are not individual bursts;
  contact or repositioning reproducibility cannot be inferred from them.
- Unsupervised 2D PCA of site-level frequency summaries and a second PCA after
  exploratory MI selection of five frequencies. Summaries use
  `log1p(mean finite magnitude across IIVV coordinates / 1 ohm)`. Patient-balanced
  imputation, scaling and weighted SVD give each patient equal total projection
  weight. Coordinates, loadings, explained variance and missing-data weights are
  exported. Classifier PCA instead ranks training-site/IIVV spectra.
- Per-task MI ranks and 20 deterministic one-site-per-patient draws describing
  sensitivity to repeated sites. Draw summaries are not confidence intervals.
  Patients with mixed site diagnoses have no implied single label.
- Exact planned outer/inner partitions, site/patient counts, raw pathology,
  demographics, metadata availability and retention for each partition.

Whole-cohort PCA/MI is descriptive and supplies no fitted transforms, ranks or
features to classifier validation. Visual separation does not estimate held-out
performance. Design choices informed by these plots must be disclosed as
cohort-informed development. The quality manifest hashes inputs and artifacts.

## Nested selection protocol

`configs/revision.json` specifies **five outer folds repeated five times, four
inner folds and seed 42**: 25 outer jobs per task, **75 overall**. All sites from a
patient stay together. Outer split seeds use `seed + repeat * 1000`; inner split
seeds additionally use `fold * 100 + 1`, with zero-based indices. Repeated CV reuses
patients and is neither leave-one-patient-out nor independent external validation.
Partitions follow the confirmed task definitions and need not match earlier runs.

For each outer job:

1. Set outer-test patients aside before fitting any preprocessing.
2. Fit training inclusion, imputation, scaling, frequency ranks, magnitude masks
   and classifiers on each inner-training partition; score inner-validation sites.
3. Screen 42 configurations, each changing one reference setting. For each of
   AUC, accuracy, F1 and sensitivity, retain the best screened choices and jointly
   tune frequency count, geometric/magnitude mask and classifier parameters.
4. Tune SVM, RF and LR separately. Inner scores also choose the family for each
   `*_selected` procedure. Multiclass AUC and F1 selection use macro averages;
   binary reported F1 is positive-class F1. Binary sensitivity selection maximizes
   specificity at an assumed inner sensitivity target of 0.95. Task 3 sensitivity
   selection targets High-grade dysplasia. Coverage penalizes inner objectives.
5. Fit calibration and binary thresholds from inner OOF predictions of outer
   training patients. Refit on all outer-training patients, freeze, then test.
6. Save candidates/failures, selected parameters, exact frequencies/coordinates,
   input dimensions, raw/calibrated predictions, timings, populations and hashes.

The search reads no prior winning configurations or outer-test scores. Its reduced
space follows earlier experience on the cohort, so the study remains adaptive.
Nonconverged fits are rejected while other streams continue. A stream missing an
expected fold receives no complete-partition pooled metric or invented score.

### Search space and reasons for omitted alternatives

| Setting | Retained | Reason for limiting the search |
|---|---|---|
| Frequency count | 1, 2, 3, 4, 5, 10, 20, 31 | Small, intermediate and complete inputs without a 31-point sweep |
| Ranking | PCA and MI | Unsupervised and supervised alternatives; ANOVA, sparse LR, RFE and permutation ranking add searches and model fits |
| Geometry | All 13 existing masks | Coordinate selection is a study objective; unmeasured geometries are not inferred |
| Magnitude masks | Training quantiles 0.1, 0.25, 0.5, 0.75, 0.9 | Short grid of data-dependent alternatives |
| Representation | Magnitude; magnitude plus phase | One complex alternative; real/imaginary, ratios and circuit fitting add branches or nonlinear fitting |
| Rejection cutoff | `<20`, `<40`, `<60`, `<80`, `<100.000001`; fixed `<80` baseline | Compares exclusion levels with full retention; the baseline is a reference, not a globally selected optimum |
| Normalization | None, global and per-frequency z scores | Unnormalized reference and two scaling choices; no per-coordinate branch |
| Imputation | Training mean | Removes a search dimension; all-missing training coordinates use the existing zero fallback |
| Class weights | None; balanced | Tests imbalance sensitivity; patient-equal training weights apply throughout |
| Joint breadth | `top_k=1`, plus best magnitude mask | At most 34 joint candidates per objective; staged screening can miss interactions |
| SVM | Linear/RBF; C = 0.001, 0.01, 0.1, 1, 10; max 100000 iterations | Omits polynomial/sigmoid kernels and larger C to bound costly fits |
| RF | 100 trees; depth 8; feature fractions 0.3, 0.7, 1 | Bounds tree cost without a large tree/depth sweep |
| LR | C = 0.001, 0.01, 0.1, 1; max 20000 iterations | Avoids weakly regularized extremes; training-span implementation reduces feature-space cost |

Omitted methods have not been shown inferior or equivalent. The nominal maximum
is 42 screened + 4 × 34 joint + 3 baseline + 2 covariate + 1 matched-control
candidate evaluations: **184 per outer job, 55200 inner fits across 75 jobs**.
Exact duplicates are cached, so actual counts can be lower. Outer refits,
calibration, reporting and bootstraps add work; this is not a runtime guarantee.

## Endpoints and statistics

Reports include AUC, accuracy, F1, precision/PPV, sensitivity (SN), specificity
(SP), NPV, coverage/abstention, per-class errors, patient-equal site metrics and
separate patient endpoints. They also include ROC, calibration, Brier scores,
decision curves, prevalence scenarios, covariate controls and timings. Binary
patient endpoints use any-positive pathology and maximum site probability;
multiclass patient endpoints exclude mixed-label patients and cannot replace
site-level analysis. Available-covariate controls are matched comparisons, not
causal adjustment for unrecorded clinical variables.

Pooled site probabilities are averaged across repetitions, requiring coverage in
every repetition. Sensitivity-oriented binary and fixed-specificity decisions use
majority votes. Repeated predictions do not count as additional patients.
Confidence intervals use **1000 whole-patient bootstrap resamples** of these frozen
predictions, without rerunning selection. Approximate corrected resampled paired
tests use separate repeat/fold scores, common-covered sites and scheduled patient
test/train ratios: 25 pairs per task when all are estimable. Global Holm correction
covers 3 tasks × 3 families × 4 objectives × 6 unique metrics = **216 hypotheses**;
PPV aliases precision, and unavailable comparisons remain in the family.
Conditional patient-block permutation summaries are supplementary.

## Performance and reproducibility

Default affinity is logical CPUs **224–255**. Up to 16 fold workers use two
BLAS/RF threads each, respecting available jobs, affinity and scheduler/cgroup
quotas. `CPUSET=0-23` allocates 24 logical CPUs and permits up to 12 such workers.
Logical CPUs are not necessarily distinct physical cores. Each outer job has an
8 GiB feature-cache ceiling; total memory also includes datasets and model state.
Feature and SVM-Gram caches are restricted to the exact training/test partition.

Workers use `spawn` to avoid inherited native thread-pool locks and share one
read-only memory-mapped dataset. Temporary storage must hold the uncompressed
arrays; set `TMPDIR` to fast scratch if needed. Progress identifies repeat/fold,
for example `task/r3/f2`. Checkpoints preserve seeds and resumable completed folds.

Each run records configuration, Python/package versions, Git commit, source/input
hashes, partitions, CPU affinity and thread settings. `environment.yml` specifies
supported dependencies; `environment-freeze.txt` records installed Python package
versions. This README is copied and hashed with each run as its protocol rationale.

## Status and remaining work

The completed v5 results contain all 75 outer computations. The saved-result
verifier passed, and the marked manuscript, tables, figure data and response
letter now describe v5. See [the v3–v5 comparison](manuscript/v3-v5-comparison.md)
for the reported changes and provenance limits. V3 raw metric files are absent
locally, so its comparison values come from the previous manuscript.

- [x] Verify v5 saved results and update paper/response evidence together.
- [ ] Restore the four missing source-extraction files listed in
  `manuscript/Revision/revision-evidence/v5-local-input-check.json` before a full
  input-file verification. All 278 available input files match their recorded hashes.
- [ ] Resolve Sophie placeholders for enrollment/exclusions, pathology registration
  and blinding, mixed histology, healthy-site confirmation, contact controls,
  acquisition/test-mean reconciliation and approved data sharing.
- [ ] Keep claims within the evidence: omitted methods are not shown inferior,
  and independent clinical validation, reduced hardware and prospective
  acquisition/single-site latency remain untested.
- [ ] Review marked/clean papers, responses and upload packages before submission.

`python scripts/export_v5_manuscript_evidence.py` regenerates the tables, native
figure data, response quality PNGs and comparison from frozen v5 outputs. It fits
no models and compiles no PDFs. Manuscript wording remains an editorial step.

Private inputs, results, manuscript files and upload ZIPs are ignored by Git and
need separate transfer. See `LICENSE` and `CITATION.cff` for reuse and citation.
