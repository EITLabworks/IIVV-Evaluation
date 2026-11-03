# Three-task revision protocol

Protocol: `three-tasks-v1-lesion-only-task3`, configuration schema 3.
All narrative, figures and new reports use English. Original patient/source labels,
identifiers, citations and the untouched original manuscript are preserved.

## Clinical questions and exact classes

| Task | Classes, in exported probability-column order | Population |
|---|---|---|
| 1: `healthy_vs_cancer` | Healthy; Cancer | Clinically healthy sites versus invasive carcinoma |
| 2: `highrisk_vs_lowrisk` | Low risk; High risk | Low risk: healthy/normal, benign and mild dysplasia. High risk: moderate/severe or high-grade dysplasia, CIS and invasive carcinoma |
| 3: `cancer_vs_lr_hr` | Cancer; Low-grade dysplasia; High-grade dysplasia | Lesion categorization, retaining the previous lesion-only population. Low-grade dysplasia is the requested display label for normal/benign tissue and mild dysplasia. High-grade dysplasia includes moderate/severe dysplasia and CIS. Clinically healthy sites are excluded |

Task 3 classes are disjoint: invasive cancer occurs only in Cancer. Every Task 3
figure caption explains the pooled lower-grade/non-malignant composition and the
inclusion of moderate dysplasia in the high-grade group. These are study groupings,
not a reassignment of the recorded histological diagnoses. Raw pathology remains
in inclusion, exclusion and fold-demographic exports. Unknown diagnoses are excluded
explicitly. No four-class classification task remains.

The clinical motivation is the surgeon's question: does this finding warrant
further evaluation or a procedure? Task 2 evaluates a histology-based grouping
related to that question. Task 3 separates invasive cancer from dysplasia and
lower-grade/non-malignant lesions, so binary pooling cannot hide these errors.
Task 1 retains the healthy-versus-cancer comparison. None of these retrospective
labels is a treatment recommendation: a low-risk prediction does not establish
that no procedure is needed, and management also depends on clinical information.

## Stage A: describe data before fitting classifiers

`./run_all.sh --data-quality-only` extracts/verifies MATLAB sidecars and exports:

- Site/patient/raw-pathology populations, repeated sites, patient pathology overlap,
  available clinical variables, class-by-location counts and metadata gaps.
- Acquisition rejection percentages separately from non-finite coordinates in
  averaged spectra; source/export equality checks; frequency, IIVV-coordinate and
  geometry-mask missingness; spectral magnitude medians and interquartile ranges.
- Class-specific retention at strict cutoffs 0, 20, 40, 60, 80 and 100.000001.
  Zero retains no sites. The last value includes measurements recorded at 100;
  comparisons in existing code use `< cutoff`. These audit thresholds do not
  expand or choose the classifier search.
- Saved test-mean repeatability, per-site CV and absolute-agreement ICC with
  whole-patient bootstrap intervals. Saved test means are not individual bursts;
  filtering/contact/repositioning controls are not inferred from them.
- Two-dimensional PCA of per-site frequency summaries, and a second PCA after
  exploratory MI selection of five frequencies. The summary is the logarithm
  `log1p(mean finite magnitude across IIVV coordinates / 1 ohm)`. Patient-balanced
  mean imputation, standardization and weighted SVD prevent patients with more
  sites from receiving greater total projection weight. Loadings, coordinates,
  explained variance and missing-data weights are saved.
  This descriptive PCA uses site-level frequency means; classifier PCA ranking
  instead uses training-site/IIVV spectra. They are distinct analyses.
- Per-task exploratory MI ranks and 20 deterministic one-site-per-patient draws
  to describe dependence on repeated sites. Draw summaries are not confidence
  intervals. A patient with different site diagnoses has no single implied label.
- The exact planned outer/inner patient partitions and detailed demographic,
  site-count, pathology, metadata-availability and quality-retention breakdowns.

PCA plots and MI ranks use the available cohort descriptively. They do not estimate
held-out accuracy, establish separability, or supply features/ranks/imputation to
tuning. All classifier transforms are independently refitted inside training
partitions. The quality manifest hashes its inputs, protocol and artifacts; tuning
requires a matching verified manifest. Changing design after inspecting cohort
plots must be disclosed as cohort-informed development.

Core quality figures are saved as PDF and PNG under `exploration/`, alongside
CSV/JSON statistics and PCA coordinates/loadings. They include PCA/MI views,
retention, missingness and saved-test-mean ICC when repeats are available.

## Stage B: nested patient-grouped selection

The retained protocol is **five outer folds, four inner folds, one outer partition,
seed 1729**, with all sites from a patient kept together. This is nested grouped
cross-validation, not leave-one-patient-out. Three tasks give **15 outer jobs**.
Partitions are regenerated under the confirmed class definitions; they are not
claimed to be identical to all prior runs. No independent external cohort exists
unless explicitly provided and verified.

For each outer job:

1. Hold outer-test patients aside before any learned preprocessing.
2. Within each inner-training partition, fit quality-dependent training inclusion,
   imputation, scaling, frequency ranking, data-dependent IIVV thresholds and the
   classifier. Apply the fitted pipeline to inner-validation patients.
3. Screen 35 fixed configurations. For each of AUC, accuracy, F1 and sensitivity,
   retain the best screened choices, then jointly tune the selected frequency
   count, geometric/data-dependent mask and classifier parameters.
4. Tune SVM, RF and LR separately and select the model family using inner scores
   for the `*_selected` procedure. Multiclass sensitivity selection targets the
   Task 3 high-grade dysplasia class. In binary tasks it maximizes specificity at
   the assumed inner sensitivity target of 0.95, with a coverage penalty.
   AUC/F1 aggregate multiclass classes by macro
   averaging. F1 selection uses macro F1 in every task; binary diagnostic tables
   report positive-class F1. Coverage penalizes inner objectives when quality causes abstention.
5. Fit calibration and binary operating thresholds using inner out-of-fold
   predictions from outer-training patients. Refit the chosen feature pipeline
   and classifier on all outer-training patients, then freeze before outer testing.
6. Save all candidates, failures, selected parameters, exact frequencies/IIVV
   coordinates, input dimensions, raw/calibrated predictions, postprocessing
   provenance, elapsed times, split populations and completed-fold hashes.

The search is fresh: it reads no previous winning parameter files or outer-test
scores. Its reduced design follows earlier experience on this cohort. It is an
adaptive analysis, not independent confirmation; nesting cannot erase that history.
Nonconverged fits are rejected and recorded. Other model streams continue. A
stream missing an expected fold is unavailable for complete-partition pooled
inference; its absence is not replaced by an invented score.

## Retained search and omitted alternatives

| Setting | Retained | Reason for excluding alternatives |
|---|---|---|
| Frequency count | 1, 3, 5, 10, 31 | Covers small, intermediate and complete inputs without a 31-point sweep |
| Ranking | PCA and mutual information | Keeps an unsupervised and a supervised alternative; omits ANOVA, sparse LR, RFE and permutation ranking to avoid redundant searches and repeated model fitting. No equivalence is claimed |
| Geometry | All 13 existing geometric masks | Selection of probe coordinates is a study objective; no unmeasured geometry is inferred |
| Magnitude masks | Training quantiles 0.1, 0.25, 0.5, 0.75, 0.9 | Keeps data-dependent alternatives on a short grid |
| Representations | Magnitude; magnitude plus phase | Retains the original signal and one complex alternative; omits real/imaginary, ratios and circuit fitting to limit branches and nonlinear fitting. No superiority is assumed |
| Quality | `<80` and `<100.000001` in search; `<60` baseline | Compares moderate exclusion with full retention. Other cutoffs are retention audits only; 60 is a reference, not a globally selected optimum |
| Normalization | Global and per-frequency z scores | Tests the preprocessing question within training; no extra per-coordinate branch |
| Imputation | Training mean | Avoids another search dimension; all-missing training coordinates use the explicit existing fallback |
| Class weights | None; balanced | Retains imbalance sensitivity; patient-equal training weights apply in both |
| Joint search breadth | One screened choice, plus best magnitude mask | Limits combinatorial expansion; staged search can miss interactions and is not exhaustive optimization |
| SVM | Linear/RBF, C = 0.001, 0.01, 0.1, 1, 10 | Drops polynomial/sigmoid kernels and larger C; limits expensive poorly conditioned fits |
| RF | 100 trees, depth 8, feature fractions 0.3, 0.7, 1 | Bounded tree cost, no large tree/depth sweep |
| LR | C = 0.001, 0.01, 0.1, 1; at most 20000 iterations | Avoids weakly regularized extremes; training-span implementation reduces feature-space cost |

At most 35 screened + 4 × 34 joint + 3 baseline + 2 covariate + 1 matched-control
evaluations give **177 candidate evaluations per outer job**, or at most **10620
inner fits across 15 jobs**. Exact duplicates are cached, so actual counts can be
lower. Outer refits, calibration, reporting and bootstraps add work. This is a budget,
not a runtime guarantee. Saved candidate/fold timings provide the runtime evidence.

## Performance and reporting

Default affinity is the last 32 logical CPUs, 224–255. Up to 16 independent jobs
use two BLAS/RF threads each, capped by available jobs, affinity and scheduler/quota.
With `CPUSET=0-23`, up to 12 jobs run at two threads each. Affinity names logical
CPUs; these are not necessarily distinct physical cores. The current 15-job design
can occupy at most 30 worker threads at once. Training-feature and SVM-Gram caches
stay within each outer job and exact partition; they never share fitted transforms
across held-out patient populations. Each feature cache has an 8 GiB ceiling;
total process memory also includes other arrays and model state.

Reports retain AUC, accuracy, F1, precision, sensitivity (SN), specificity (SP),
PPV, NPV, coverage/abstention, per-class errors, ROC, calibration, decision curves,
prevalence scenarios, patient-equal site metrics and explicitly defined patient
endpoints. Multiclass patient endpoints exclude mixed-label patients. They are
not interchangeable with site endpoints. Confidence intervals use 1000
whole-patient bootstrap resamples of frozen predictions; selection is not rerun.
CV comparisons use approximate corrected resampled paired outer-fold tests,
correcting by patient training/test ratios. Holm correction covers the scheduled
3 tasks × 3 model families × 4 objectives × 6 unique metrics = **216 hypotheses**;
PPV aliases precision. Unavailable comparisons remain in the family. Conditional
permutation summaries remain supplementary. Available-covariate controls are
matched analyses, not causal adjustment for missing clinical variables.

No numerical results from the old four-task runs are imported into this protocol.
Final claims require the new server artifacts and completion verification.
