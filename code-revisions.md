# Code revisions for the first manuscript revision

The authoritative revision workflow is `./run_all.sh`. It replaces cohort-wide preprocessing and sequential reuse of selected configurations with one nested patient-group evaluation. The original manuscript and original result directories are retained as historical records. New results must be interpreted separately from those exploratory results.

## Running the analyses

**Author decision, 20 October 2026:** the final run uses the expanded
`configs/revision.json` profile: 5 outer × 4 inner patient folds, 10 repeats.
See [SERVER_RUN.md](SERVER_RUN.md) for the ordered schedule. No project code,
tests, evaluation or LaTeX build was executed during this preparation.

Use Python 3.10 or newer and install `requirements-revision.txt` in your analysis environment. The supplied run uses CPU scikit-learn; two independent diagnostic tasks run concurrently on fork-capable systems, sharing immutable raw arrays while keeping all fitted state and outputs separate. Set `task_workers` to 1 for serial execution or platforms without fork. The statistical protocol and seeds do not depend on worker count. `environment-freeze.txt` in each output directory records the actual installed versions. The historical GPU environment in `environment.yml` is not required by this workflow.

```sh
python -m pip install -r requirements-revision.txt
# Verify source alignment, inputs, grouping and all target populations:
./run_all.sh --preflight --output results/revision-expanded-preflight
# Fresh final expanded run; preserve old clinical-profile checkpoints separately:
./run_all.sh --output results/revision-expanded
# Verify partitions, predictions, matched controls, Holm family and inputs:
python 5_check_results.py --results-dir results/revision-expanded --verify-inputs
# Numerical summary:
python print_eval_results.py --results-dir results/revision-expanded
# Unit tests and a separately identified synthetic integration exercise:
python -m pytest -q tests/test_revision.py
./run_revision.sh --synthetic --output results/revision-synthetic-check
```

A result directory is not overwritten after fitting starts. Use a fresh `--output` for another run. To continue an interrupted/failed expanded run with identical inputs and configuration, use `./run_all.sh --resume --output results/revision-expanded`. Only fully checkpointed folds are reused; hashes/configuration must match, partial folds are rerun, and source snapshots plus per-fold source versions/resume manifests are retained. A completed or active run cannot be resumed. Rejected resumes retain the original input inventory, so a second attempt cannot silently accept changed inputs. `--preflight` must use a separate output directory and cannot be combined with `--resume`; it computes quality/repeatability audits but fits no classifiers. Synthetic runs are explicitly marked and cannot enter clinical reporting. Large runs require substantial memory, disk and CPU time; the available test means alone occupy approximately 4 GB.

The shell runner defaults BLAS and OpenMP to eight threads, respecting explicit environment settings. The server schedule sets BLAS/OpenMP/MKL to one thread to limit nested contention with two task workers and four RF threads. CPU affinity is optional; choose CPU IDs allowed on the server. Set `PYTHON=/path/to/environment/bin/python` to select an interpreter. The manifest records effective affinity and thread settings.

The previous computation is interrupted with its checkpoint files preserved. No full clinical completion is claimed. The current handoff is code preparation only: run the computation and completion check above, then return for manuscript/reviewer updates. The runner writes numerical tables, audits and plots only; it does not invoke the later manuscript/rebuttal import commands. For a fresh run, replace the output path in all subsequent checking commands with that fresh path.

`run_all.sh` selects `configs/revision.json` and requires/verifies the MATLAB source automatically. Reusing an extraction requires matching MATLAB, NPZ and sidecar hashes. To supply different inputs:

```sh
./run_all.sh --data /path/to/npz --patterns /path/to/ordered_IIVV.txt \
  --source-mat /path/to/source.mat --source-output results/new-source \
  --output results/new-analysis
# Optional explicit clinical sidecar / repeat array:
./run_revision.sh --config configs/revision.json --data data \
  --metadata results/revision-source/clinical_metadata.csv \
  --repeats results/revision-source/test_mean_repeats.npy \
  --repeat-index results/revision-source/repeat_index.csv \
  --output results/explicit-input-analysis
```

## Prespecified configurations and interpretation

The historical compact clinical profile (`configs/revision-clinical.json`) uses five outer folds, three inner folds, one outer partition and seed 1729. It is retained for provenance. Both profiles use 1,000 patient-cluster bootstrap/permutation draws, all 31 frequency counts, 13 geometry masks, six selectors, five representations, three normalization/imputation choices and default/balanced class weights. The compact profile has quantiles 0.25/0.50/0.75 and cutoffs 40/60/80/100.000001%. Cutoffs are strict `<`; the final value permits the full supplied quality-score range with training-fitted imputation. The reference cutoff is 60%.

Search is staged: a fixed SVM screens one factor at a time inside outer-training data. Each objective retains the configured number of frequency counts and geometry masks plus its best impedance-threshold mask, then jointly tunes these with the screened preprocessing, selector and representation. Every candidate, failure and selection is recorded. Outer ablations assess the screened alternatives; incomplete ablation populations are withheld and identified in `unavailable_streams.json`. Results compare pipelines under the configured search budget, not a global optimum.

**Final profile:** `configs/revision.json` has five outer/four inner folds, ten partitions, top-four frequency/mask choices, impedance quantiles 0.10/0.25/0.50/0.75/0.90 and cutoffs 20/40/60/80/100.000001%. SVM uses four kernels and seven C values (0.001 through 1000); LR uses the same C values. RF uses 100/300/800 trees, depth 8/16/unlimited and feature fractions 0.3/0.7/1.0. This is the default of `run_all.sh`. The older All-mask `revision-available-data.json` is an interim profile, not the final protocol.

The four objectives are macro AUC, accuracy, macro F1 and sensitivity. The binary sensitivity objective maximizes specificity subject to a **training** sensitivity target of 0.95. Multiclass sensitivity selection targets high-grade dysplasia recall. Specificity operating targets 0.80/0.90/0.95 and projected prevalences 0.10/0.20 are analysis assumptions, not validated clinical requirements. A held-out sensitivity below the training target is possible and must be reported.

## Input and label audit

Each NPZ contains one site's complex `Zdata` with common frequency/pattern coordinates, `patient_no`, `patient_sample`, `LesionAssessment` and `Avg_Cutpats`. No normalization, missing-value replacement or completeness exclusion occurs in the loader. Non-finite values are preserved. Optional `frequencies_hz` must agree across files; otherwise the documented original 31-point 100 Hz–100 kHz geometric grid is used. The ordered coordinate file must have exactly one four-electrode row per impedance column. Geometry order is supplied, not inferred from impedance values.

The supplied 276 sites correspond to **100 distinct patients**, using both exported IDs and original cohort/case pairs. The paper's previous 104-patient count cannot be reproduced from these analysis files. All 276 exported complex arrays and cohort/case identifiers were checked against the MATLAB source before extracting metadata/repeats.

Canonical pathology mapping follows the conversion notebook and is checked against stored multiclass labels. Moderate dysplasia belongs to high-grade dysplasia, not the benign/mild group. Raw unmapped diagnoses are excluded and listed in `site_inclusion_audit.csv`; they are never assigned using substring guesses. An explicit alternative adjudicated mapping may be provided using `--label-map`, with a fresh analysis output. Current populations before quality filtering are:

| Task | Explicit class order / probability columns | Sites | Patients |
|---|---|---:|---:|
| healthy_vs_cancer | Healthy (0), Cancer (1, positive) | 196 | 98 |
| cancer_vs_lr_hr | Cancer (0), High-grade dysplasia (1), Non-malignant (2) | 172 | 97 |
| MC | Healthy (0), Cancer (1), High-grade dysplasia (2), Non-malignant (3) | 272 | 100 |
| highrisk_vs_lowrisk | Low risk (0), High risk (1, positive) | 272 | 100 |

The last task combines OSCC/high-grade dysplasia/CIS against healthy/benign/mild dysplasia. Four other diagnoses remain excluded. The 60% cutoff retains 256 of 272 eligible sites: Healthy 90/100, Cancer 96/96, high-grade dysplasia 29/32, non-malignant 41/44. Including the four excluded diagnoses gives 260 retained sites, explaining the old table's apparent discrepancy.

`Avg_Cutpats` is a supplied acquisition quality score. It is **not silently replaced** by NaN counts: the averaged spectra differ by more than one percentage point in 209/276 sites, with maximum absolute discrepancy 36.154 percentage points. Observed missing patterns are defined as patterns with at least one non-finite frequency. A direct source audit verifies that all 276 supplied quality scores equal the mean of per-test rejection percentages, and every exported spectrum equals the NaN-aware mean of the available test spectra. Averaging recovers coordinates absent from an individual test; the average test-mean NaN-pattern percentage also agrees with the recorded mean rejection score for every site. Exact within-burst filtering semantics remain unverified. Quality and missingness-only controls use explicit definitions and report both quantities.

## Leakage corrections and features

- `revision/data.py`: raw complex/NaN loading, explicit label mapping, patient/site IDs, input hashing and optional keyed clinical metadata/repeat arrays. Object pickle loading is disabled unless explicitly requested for trusted local files.
- `revision/features.py`: training-fitted frequency/global/no scaling and fixed-column mean/median/zero imputation; entirely absent training columns use fixed zero. Masks, coordinate imputation, frequency ranking and models are frozen before held-out transforms. Held-out values cannot change training statistics.
- PCA uses unweighted summed squared coefficients of the first ten available axes. It is unsupervised, not evidence of diagnostic association. Training rows are deterministically subsampled to at most 100,000 for PCA. Supervised ANOVA, mutual information, sparse LR, RFE and RF permutation rankings use per-site frequency summaries within training data. The permutation ranking is a training heuristic, with its classification value assessed on outer patients.
- Geometry masks reuse the repository's electrode rules. Impedance-threshold masks use equal-patient-weighted **training** 100-Hz magnitudes and training quantiles, then freeze both threshold and selected columns. `source/Avg_Zs.npy` is not used for selection.
- Representations compare magnitude, real/imaginary, magnitude/phase, magnitude ratios to each site's 100-Hz reference and an effective Cole–Cole fit. The latter fits a mean complex spectrum over selected patterns and returns R_inf, Delta_R, tau, alpha and relative fit error; it is a phenomenological feature baseline, not validated tissue compartment inference. Missingness indicators are a separate negative-control representation. Report failed fits rather than inventing parameters.
- `revision/pipeline.py`: all tasks, preprocessing/feature/model searches, class weights, calibration and threshold fitting stay inside patient-disjoint outer-training data. Inner folds also group patients. SVC internal probability fitting is disabled; logistic calibration uses grouped inner out-of-fold scores. Baselines receive the same calibration and threshold treatment. Inverse sites-per-patient fitting weights are used; augmentation before splitting is forbidden.
- Bounded feature caching is local to one outer fold and keyed by exact train/test site IDs, valid-test indicators, seed, mask, ranking, representation, normalization and imputation. No fitted feature state is shared across training partitions. Frequency-prefix candidates reuse the same training-fitted full ranking.
- `revision/kernel_svm.py`: standard linear/RBF/polynomial/sigmoid SVC kernels are computed with vectorized Gram matrices using training-only gamma and unchanged libsvm optimization. Binary/multiclass decision-score parity is tested against ordinary SVC. Completed outer fits are released after their last shared reporting stream, bounding model memory without changing fitted predictions.
- `revision/linear_span.py`: high-dimensional L2 logistic regression is solved in an orthonormal basis of the training-row span, then coefficients are mapped back to original feature coordinates for prediction. This preserves the L2 objective up to numerical null directions, without selecting an acquisition subset or consulting held-out data. Binary/multiclass weighted regression checks compare probabilities with a direct full-coordinate sklearn fit. The optimized dimension and original input feature count are both archived.
- Saved frequency indices are actual zero-based coordinate identities and physical hertz. Subset-size rankings are stored separately, fixing the previous count-versus-frequency artifact mismatch. Outer summaries are reporting artifacts and are never loaded back into evaluation.

## Metrics, uncertainty and missing measurements

`revision/metrics.py` and `revision/analysis.py` report per-class one-versus-rest AUC, sensitivity, specificity, PPV, NPV, F1, TP/FP/FN/TN; macro/micro summaries, balanced accuracy, Brier score, log loss, count/normalized confusion matrices and correctly aligned ROC curves. A missing reference class gives an undefined full-target macro AUC rather than silently reducing the class set. Dysplasia false-negative destinations are visible in the full confusion matrices.

Three estimands are separate: pooled sites, equal-patient-weighted sites, and an explicit patient endpoint. Binary patient truth is any-positive and risk is maximum site probability; a positive decision at any site refers the patient. Multiclass patient summaries include only patients whose analyzed sites have homogeneous pathology; excluded mixed-pathology patients are listed, and remain included in the site analyses. Weights are formed before removing quality abstentions.

Training-selected quality failure is a held-out abstention (`y_pred=-1`), never a hidden test exclusion. Inner selection multiplies its score by patient-weighted coverage to discourage free performance gains from discarding difficult sites. Each metric reports coverage and denominators. Binary referral-workflow summaries additionally refer unevaluable measurements, without assigning them fabricated probabilities. ROC/calibration metrics describe covered sites and must be read alongside coverage.

Calibration tables/figures, multiclass Brier and log loss, train-selected operating thresholds, prevalence-projected PPV/NPV and decision curves accompany predictions. Projection requires transportable sensitivity/specificity; net benefit uses this surgical cohort's prevalence and is exploratory. Thresholds are not chosen from outer ROC curves.

Confidence intervals resample whole patient blocks, never individual sites or dependent fold scores. Comparisons pair same-model/objective pipelines against their corresponding all-input baseline on common covered sites, swap predictions by whole patient, and apply Holm correction across the complete requested task/model/objective family. The preparation-and-fit timer can include the test transform on a cache miss; it is not a pure model-fit time. A separate model-fit timer is reported where measured, with earlier source-version folds left missing rather than reconstructed. Cached transforms may process the full ranked feature matrix before selecting a prefix, so they are not deployed selected-input preprocessing benchmarks. Task concurrency can affect measured computation latency, which is a software benchmark under the recorded environment rather than instrument timing. These intervals and permutation results are **conditional on fitted outer predictions**, not full refitting uncertainty or a population-level confirmatory trial. They do not retrospectively validate nominal p-values or error bars in the original figures. Repeated outer partitions, when configured, produce one mean-probability row per site; uncertainty never counts repeated predictions as independent subjects.

## Clinical covariates, quality and repeatability

The supplied anatomy, prior radiation and prior surgery are available. The source extractor joins them only after verifying source/export identity. Class/location tables, quality retention counts, patient-cluster sandwich association tests with Holm-adjusted families, location-stratified outer metrics, and training-fitted clinical-only, matched EIS-only and combined LR comparisons assess these available factors. Unrecorded age, sex, smoking, alcohol, inflammation and detailed treatment/contact variables are not inferred. HPV and grade fields serialized as opaque MATLAB string objects are not interpreted as Unicode or invented metadata.

MATLAB `Full_data` holds repositioned **test means**: 268 sites have three, six have four, two have two. All available means contribute to CV/repositioning summaries. The balanced absolute-agreement ICC(2,1) uses the first three test positions and reports complete-site exclusions, class-specific estimates and patient-cluster intervals. The ICC endpoint is mean magnitude over available frequencies/patterns per test; coordinate CVs and site scalar variability are distinct outputs. This is not nine-burst repeatability, a reliability measure of every classifier feature, or evidence that contact pressure is controlled. Blinding, section-to-footprint matching, mixed-histology adjudication, burst-level noise and prospective acquisition/hardware validation still need author records or new measurements.

## Output map and provenance

Each output root contains `run_manifest.json`, `resolved_config.json`, `environment-freeze.txt`, `analysis_status.json`, `quality/`, `repeatability/` and one directory per task. Only a manifest with `status: completed` establishes completion. Interrupted/failed/preflight manifests are not clinical result sets.

| Output | Purpose |
|---|---|
| `all_metrics.csv`, `BaselineResults.csv`, `FinalResults.csv` | Cross-task metrics and familiar baseline/tuned tables |
| `paired_comparisons_holm.csv` | Paired patient-block comparisons with the full-family correction |
| `TASK/outer_predictions.csv`, `classes.json` | Every outer prediction, class order, raw/calibrated probabilities, abstentions and decisions |
| `TASK/splits.json`, `selections.json` | Exact patient partitions, fitted feature identities, hyperparameters, thresholds, warnings and timing |
| `TASK/split_population_counts.csv` | Class/location site and patient counts for every outer/inner partition and quality cutoff |
| `TASK/unavailable_streams.json` | Incomplete ablation populations withheld from numerical comparisons |
| `TASK/inner_search.csv`, `candidate_failures.json` | Search scores, coverage and explicit failed candidates |
| `TASK/*site_predictions.csv`, `*patient_predictions.csv` | One row per site or defined patient endpoint |
| `TASK/per_class_metrics.csv`, `*confusion_counts.csv` | Class-specific counts and dysplasia error destinations |
| `TASK/*confidence_intervals.csv` | Overall and per-class patient-cluster conditional intervals with valid-draw counts |
| `TASK/*operating_points.csv`, `*referral_workflow.json` | Held-out threshold and referral-fallback evaluation |
| `TASK/*prevalence_projections.csv`, `*calibration.csv`, `*decision_curve.csv` | Clinical interpretation at explicitly stated assumptions |
| `TASK/location_stratified_metrics.csv` | Available anatomy strata without retraining on test outcomes |
| `TASK/frequency_and_pattern_identities.json`, `ranking_stability.csv` | Actual selected coordinates and descriptive cross-fold Jaccard stability |
| `TASK/FreqSel_ranking_*`, `MaskSel_ranking_*`, `zthres_tuning_*` | Inner-only frequency/mask/threshold reports |
| `TASK/measured_computation_times.csv`, `hardware_requirements.csv` | Preparation/fit, separately measured model-fit where available, held-out transformation/prediction times and required electrodes/drive pairs; no physical simplification claim |
| `plots_TASK_*` | Regenerated confusion/ROC/calibration/decision figures, including familiar legacy filenames under the new output root |

Manifests hash scientific source files and inputs and record actual software versions, resolved settings, timestamps, synthetic flag and completion. Model serialization is optional (`save_models`); default results retain reproducible fits through inputs, settings and partitions. Sharing clinical data or artifacts remains subject to the original access conditions. Neither private source data nor generated patient results are automatically added to Git.

The old numbered analysis entry points now delegate to this single workflow, rather than independently fitting and reusing globally selected artifacts. `--task`, `--metric`, `--model` and fold-count aliases are supported by the compatibility adapter; see `--help` for current flags. Unsupported historical optimization flags are deliberately rejected. The legacy loader rejects requests for cohort-wide normalization and old helpers reject pre-split augmentation. Reporting entry points inspect the new result schema. Historical artifacts remain available separately.

## Manuscript, rebuttal and reviewer traceability

Paper revisions are in `manuscript/Paper/Revision-1.tex`, with additions in `\edit{}` and retained deletions in `\edits{}`. `manuscript/Paper/main.tex` must remain unchanged. The response letter uses a purple star for comments linked to changed code and the corresponding rerun outputs, including comments whose experimental/clinical component still needs external evidence. It distinguishes implementation, completed computations and unresolved clinical evidence. Screenshots are supplementary excerpts from the compiled revised manuscript, not substitutes for precise textual locations.

Detailed comment-by-comment actions and outstanding evidence are recorded in `manuscript/REVISION_SUMMARY.md`. Major code-related groups are:

| Reviewer comments | Implementation / evidence |
|---|---|
| R1 C1–3, C7–8, C21; R3 C1, C3 | Patient-nested fitting, train-only transforms, frozen masks and saved splits/identities |
| R1 C4–5, C12–14 | Completeness/NaN audits, class/anatomy tables, missingness control, available covariates |
| R1 C6, C9; R3 C3–4 | Six selectors, complex/ratio/circuit representations, ranking stability |
| R1 C10, C15–16 | Patient weights/endpoints, per-class counts, dysplasia errors, macro/balanced metrics |
| R1 C17–19 | Whole-patient conditional uncertainty, paired comparisons and Holm correction |
| R1 C20–21, C26–27; R3 C2 | Verified feature identities, frozen impedance masks, measured software timing, electrode/drive counts |
| R1 C24–25 | Available test-mean repeatability; contact controls and burst-level evidence remain unavailable |
| R1 C30; R2 C3–5 | Combined referral task, sensitivity-aware tuning, calibration, PPV/NPV and decision curves |
| R1 M6–8, M10 | Reconciled populations, weights, CPU environment, commands and provenance |
| R1 C11; R2 C1–2 | No code can recover missing pathology registration, blinding or mixed-histology records |

Build the marked/clean drafts and response with `make revision revision-clean response`. The clean entry point suppresses deletions, unwraps additions and includes the current marked source; it does not modify either manuscript source. The clean PDF is `manuscript/Paper/Revision-1-clean.pdf`. Generate supplementary screenshots from the marked PDF only after we review and import completed results. Importers require completed non-synthetic evidence and validate result structure; they do not silently replace changed generated manuscript blocks.

The available-covariate comparison includes a matched EIS-only LR control (`covariate_eis_only`) with the same C=1, frozen selected EIS feature/preprocessing rule, quality eligibility and patient folds as `covariate_combined`. This separates covariate addition from changing classifier settings. `run_all.sh` includes this control for new executions. For an older completed run, `python scripts/add_matched_covariate_control.py --results results/revision-clinical-final` adds the frozen-rule control and regenerates reports without changing original primary predictions. The postprocessing manifest records its source/input hashes.

The older interrupted clinical run has a documented legacy input-inventory recovery and archived source history. It is preserved separately; the final expanded run starts afresh. New runs persist input hashes before fitting, verify cached extraction sidecars, and include the original MATLAB source and extraction manifest in the provenance inventory. A missing MATLAB file is an error for `run_all.sh`.

Forest trees are trained on four threads; inference resets to one thread to preserve fixed tree-order probability accumulation. Tests verify identical tree splits/thresholds and bit-identical probabilities against single-thread fitting. Two diagnostic tasks may run concurrently, so software fit timings describe that recorded resource setting. Legacy binary plotting helpers now require an explicit positive label instead of silently selecting the largest encoded label.

Deferred until the user returns with a completed clinical run and matched controls: update the paper and rebuttal using:

```sh
python scripts/update_manuscript_results.py --results results/revision-expanded
python scripts/update_abstract_results.py --results results/revision-expanded
python scripts/update_response_results.py --results results/revision-expanded
python scripts/create_rebuttal_screenshots.py
make revision-clean response
python scripts/validate_revision.py --results results/revision-expanded --require-completed
```

The numerical imports reject synthetic/incomplete evidence. Marked manuscript replacements append a reversible ledger; changed generated blocks require explicit revision handling rather than silent replacement.
