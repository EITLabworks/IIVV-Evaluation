#!/usr/bin/env python3
"""Import verified clinical outputs into marked Revision-1.tex; never touch main.tex.

The data audit may be imported separately. Classification imports require the
completed, non-synthetic clinical manifest and all requested task summaries.
Existing generated blocks may only be identical; changed evidence needs explicit
marked replacement rather than silent regeneration.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MAIN_HASH = "98173ed853cdcb4ec26dc61da4bec9a67517fb4957ffeb769eb41e0468921288"
TASK_LABELS = {
    "healthy_vs_cancer": "Healthy/cancer",
    "cancer_vs_lr_hr": "Lesion three-class",
    "MC": "Four-class",
    "highrisk_vs_lowrisk": "Low/high risk",
}


def edit(text):
    return r"\edit{" + str(text) + "}"


def number(x, digits=3):
    return f"{x:.{digits}f}" if pd.notna(x) else "undefined"


def cellrow(cells):
    return " & ".join(edit(v) for v in cells) + r" \\"


def insert(s, name, content, anchor):
    start = f"% BEGIN GENERATED {name}"
    end = f"% END GENERATED {name}"
    block = start + "\n" + content + "\n" + end + "\n\n"
    if start in s:
        old = re.search(re.escape(start) + ".*?" + re.escape(end) + "\n\n", s, re.S)
        if old is None or old.group() != block:
            raise RuntimeError(
                f"{name} changed; preserve old text with explicit revision markup before importing different results"
            )
        return s
    if s.count(anchor) != 1:
        raise RuntimeError(f"Ambiguous insertion anchor for {name}")
    return s.replace(anchor, block + anchor, 1)


def data_audit(root):
    counts = pd.read_csv(root / "quality/missing_patterns_by_class.csv").set_index(
        "pathology"
    )
    icc = pd.read_csv(
        root / "repeatability/icc_and_inter_site_variability.csv"
    ).set_index("class_name")
    retention = pd.read_csv(root / "quality/retention_by_threshold_class.csv")
    retention = retention[retention.cutoff == 60].set_index("class_name")
    quality = json.loads((root / "quality/source_quality_audit.json").read_text())
    if (
        quality["n_sites"] != 276
        or not quality["all_reported_quality_equals_mean_of_test_quality"]
        or quality["n_exported_spectra_equal_nanmean_of_test_means"] != 276
    ):
        raise RuntimeError(
            "Source-quality findings differ from the verified study population; inspect before importing"
        )
    overall = icc.loc["All"]
    text = r"""\subsection{\edit{Source-verified data, quality, and test-mean repeatability}}
\label{sub:revision_data_results}
\edit{The exported dataset contains 276 sites from 100 distinct patients; the three original tasks use 196, 172, and 272 eligible sites before quality filtering, and the combined low/high-risk task uses 272. Moderate dysplasia is assigned to the high-grade group consistently in the conversion notebook and stored labels; four Other diagnoses are excluded. Every exported complex spectrum and original cohort/case pair was verified against the MATLAB source. For all 276 sites, the exported spectrum equals the NaN-aware mean of available test means, and the supplied rejection percentage equals the arithmetic mean of the per-test rejection percentages. Thus, rejection percentage and missing coordinates after averaging measure different quantities: 209 sites differ by more than one percentage point, with a maximum absolute difference of 36.154 percentage points. Neither quantity is silently substituted for the other.}
\edit{Quality distributions differ by class (Table~\ref{tab:revision_quality}). The descriptive patient-cluster class-association test gives $p=1.39\times10^{-13}$; sparse location/joint tests with singular covariance are unavailable, not evidence of no association. Missingness can therefore be a predictive artifact and a selection mechanism. Anatomical distributions and available prior-radiation/surgery comparisons are reported separately, with residual confounding by unavailable risk/contact variables acknowledged.}
"""
    text += (
        edit(
            "The test-mean scalar endpoint is the mean magnitude over available patterns and frequencies per test. Its absolute-agreement ICC(2,1) across all supplied sites is "
            + number(overall.icc_2_1_absolute)
            + " (patient-cluster 95\\% interval "
            + number(overall.icc_ci_low)
            + "--"
            + number(overall.icc_ci_high)
            + f"; {int(overall.balanced_icc_sites)} complete sites with three test positions). All two to four available test means contribute to CV/repositioning summaries; the balanced ICC excludes sites without three finite test means. Class-specific intervals are wide, particularly for non-malignant tissue. These estimates do not quantify within-burst variability or validate individual classifier-feature reliability."
        )
        + "\n"
    )
    text += (
        r"\begin{table*}[t]"
        + "\n"
        + r"\centering"
        + "\n"
        + r"\caption{"
        + edit(
            "Verified class quality and test-mean repeatability. Quality is the supplied mean rejection percentage. ICC intervals resample patients; the endpoint is scalar mean magnitude, not every impedance coordinate."
        )
        + r"}\label{tab:revision_quality}"
        + "\n"
        + r"\begin{tabular}{lrrrrl}\toprule"
        + "\n"
    )
    text += (
        cellrow(
            [
                "Class",
                "Sites",
                "Retained at 60\\%",
                "Mean rejection (\\%)",
                "ICC",
                "95\\% interval",
            ]
        )
        + "\n"
        + r"\midrule"
        + "\n"
    )
    for label in ["Healthy", "Cancer", "High-grade dysplasia", "Non-malignant"]:
        row = counts.loc[label]
        r = retention.loc[label]
        v = icc.loc[label]
        text += (
            cellrow(
                [
                    label,
                    int(row["count"]),
                    int(r.n_retained_sites),
                    number(row["mean"], 2),
                    number(v.icc_2_1_absolute),
                    number(v.icc_ci_low) + "--" + number(v.icc_ci_high),
                ]
            )
            + "\n"
        )
    return text + r"\bottomrule\end{tabular}\end{table*}" + "\n"


def classifier_results(root, manifest):
    if manifest.get("status") != "completed" or manifest.get("synthetic"):
        raise RuntimeError(
            "Only completed non-synthetic classification results may enter the manuscript"
        )
    frame = pd.read_csv(root / "all_metrics.csv")
    classes = pd.concat(
        [
            pd.read_csv(root / t / "per_class_metrics.csv")
            for t in manifest["config"]["tasks"]
        ]
    )
    expected = set(manifest["config"]["tasks"])
    if set(frame.task) != expected:
        raise RuntimeError("Incomplete task summaries")
    cfg = manifest["config"]
    text = r"""\subsection{\edit{Patient-nested reanalysis}}
\label{sub:revision_results}
"""
    text += (
        edit(
            f'The completed clinical rerun uses {cfg["outer_folds"]} outer and {cfg["inner_folds"]} inner patient folds, {cfg["outer_repeats"]} outer partition(s), seed {cfg["seed"]}, and {cfg["bootstrap_samples"]} patient-cluster resamples. '
            f'It screens {len(cfg.get("frequency_counts",list(range(1,32))))} frequency counts, {len(cfg["masks"])} geometry masks, '
            f'{len(cfg["z_quantiles"])} training-derived impedance quantiles, {len(cfg["selectors"])} selectors, '
            f'{len(cfg["representations"])} representations, {len(cfg["cutoffs"])} quality cutoffs, '
            f'{len(cfg["normalizations"])} scaling choices, {len(cfg["imputations"])} imputation choices, and '
            f'{len(cfg["class_weights"])} class-weight settings. The staged search retains up to {cfg["top_k"]} '
            "frequency counts and geometry masks per objective, plus the best impedance-threshold mask, for model-grid tuning; "
            "it is not an exhaustive joint search. The exact model grids are archived in resolved\\_config.json. "
            "All choices remain inside outer-training data. The primary discrimination comparison uses the AUC-selected pipeline; "
            "no outer outcome selects the reporting stream."
        )
        + "\n"
    )
    text += (
        r"\begin{table*}[t]\centering\footnotesize"
        + "\n"
        + r"\caption{"
        + edit(
            "New patient-nested outer results. These are site-level covered-case metrics, accompanied by coverage. Baseline uses all physical inputs with training-fitted scaling. AUC-selected model/feature choice is made independently within each outer-training population. Conditional patient-cluster intervals and patient-weighted/patient endpoint results are supplied in the archived reports."
        )
        + r"}\label{tab:revision_nested}"
        + "\n"
        + r"\begin{tabular}{lllrrrrr}\toprule"
        + "\n"
    )
    text += (
        cellrow(
            [
                "Task",
                "Pipeline",
                "AUC 95\\% interval",
                "Macro AUC",
                "Accuracy",
                "Macro F1",
                "Balanced accuracy",
                "Coverage",
            ]
        )
        + "\n"
        + r"\midrule"
        + "\n"
    )
    for task in manifest["config"]["tasks"]:
        for stream in [
            "baseline_svm",
            "baseline_rf",
            "baseline_lr",
            "auc_svm",
            "auc_rf",
            "auc_lr",
            "auc_selected",
        ]:
            row = frame[
                (frame.task == task)
                & (frame.evaluation == "nested_cv")
                & (frame.unit == "site")
                & (frame.stream == stream)
            ]
            if len(row) != 1:
                raise RuntimeError(f"Missing unique metrics {task}/{stream}")
            row = row.iloc[0]
            label = (
                stream.replace("baseline_", "Baseline ")
                .replace("auc_", "AUC ")
                .replace("_", " ")
            )
            interval = (
                pd.read_csv(
                    root / task / f"nested_cv_{stream}_site_confidence_intervals.csv"
                )
                .set_index("metric")
                .loc["macro_auc"]
            )
            text += (
                cellrow(
                    [
                        TASK_LABELS[task],
                        label,
                        number(interval.ci_low) + "--" + number(interval.ci_high),
                        number(row.macro_auc),
                        number(row.accuracy),
                        number(row.macro_f1),
                        number(row.balanced_accuracy),
                        number(row.coverage),
                    ]
                )
                + "\n"
            )
        text += r"\midrule" + "\n"
    text += r"\bottomrule\end{tabular}\end{table*}" + "\n"
    # Always report the prespecified stream, even when dysplasia detection is poor.
    for task in ["cancer_vs_lr_hr", "MC"]:
        row = classes[
            (classes.task == task)
            & (classes.unit == "site")
            & (classes.evaluation == "nested_cv")
            & (classes.stream == "auc_selected")
            & (classes.class_name == "High-grade dysplasia")
        ].iloc[0]
        cm = pd.read_csv(
            root / task / "nested_cv_auc_selected_site_confusion_counts.csv",
            index_col=0,
        )
        destinations = ", ".join(
            f'{col}: {int(cm.loc["High-grade dysplasia",col])}'
            for col in cm.columns
            if col != "High-grade dysplasia"
        )
        interval = pd.read_csv(
            root
            / task
            / "nested_cv_auc_selected_site_per_class_confidence_intervals.csv"
        )
        interval = interval[
            (interval.class_name == "High-grade dysplasia")
            & (interval.metric == "sensitivity")
        ].iloc[0]
        text += (
            edit(
                f"For the AUC-selected {TASK_LABELS[task].lower()} pipeline, high-grade dysplasia has {int(row.tp)} true positives and {int(row.fn)} false negatives among {int(row.support)} covered reference sites; sensitivity is {number(row.sensitivity)} (conditional patient-cluster 95\\% interval {number(interval.ci_low)}--{number(interval.ci_high)}), specificity {number(row.specificity)}, PPV {number(row.precision)}, NPV {number(row.npv)}, F1 {number(row.f1)}, and one-versus-rest AUC {number(row.auc)}. False-negative destinations are {destinations}."
            )
            + "\n"
        )
    text += (
        r"\begin{table*}[t]\centering\footnotesize"
        + "\n"
        + r"\caption{"
        + edit(
            "Sensitivity-selected binary outer-site operating points. The 0.95 target is fitted in training and is not guaranteed in test patients. The fallback refers quality abstentions without assigning artificial probabilities."
        )
        + r"}\label{tab:revision_referral}"
        + "\n"
        + r"\begin{tabular}{llrrrrrr}\toprule"
        + "\n"
        + cellrow(
            [
                "Task",
                "Pipeline",
                "Sensitivity",
                "Specificity",
                "PPV",
                "NPV",
                "Coverage",
                "Fallback referrals",
            ]
        )
        + "\n"
        + r"\midrule"
        + "\n"
    )
    for task in ["healthy_vs_cancer", "highrisk_vs_lowrisk"]:
        for stream in [
            "sensitivity_svm",
            "sensitivity_rf",
            "sensitivity_lr",
            "sensitivity_selected",
        ]:
            row = frame[
                (frame.task == task)
                & (frame.unit == "site")
                & (frame.evaluation == "nested_cv")
                & (frame.stream == stream)
            ].iloc[0]
            referral = json.loads(
                (root / task / f"nested_cv_{stream}_referral_workflow.json").read_text()
            )
            text += (
                cellrow(
                    [
                        TASK_LABELS[task],
                        stream.replace("sensitivity_", ""),
                        number(row.sensitivity),
                        number(row.specificity),
                        number(row.precision),
                        number(row.npv),
                        number(row.coverage),
                        number(referral["referral_fraction"]),
                    ]
                )
                + "\n"
            )
    text += r"\bottomrule\end{tabular}\end{table*}" + "\n"
    text += (
        r"\begin{table*}[t]\centering\footnotesize"
        + "\n"
        + r"\caption{"
        + edit(
            "Requirements of the AUC-selected outer refits, shown as fold ranges. Circuit features still require the full selected raw spectrum. These coordinate requirements do not validate channel removal or a redesigned instrument."
        )
        + r"}\label{tab:revision_hardware}"
        + "\n"
        + r"\begin{tabular}{lrrrrrr}\toprule"
        + "\n"
        + cellrow(
            [
                "Task",
                "Frequencies",
                "Patterns",
                "Classifier inputs",
                "Current electrodes",
                "Voltage electrodes",
                "Drive pairs",
            ]
        )
        + "\n"
        + r"\midrule"
        + "\n"
    )

    def extent(values):
        lo, hi = int(min(values)), int(max(values))
        return str(lo) if lo == hi else f"{lo}--{hi}"

    for task in manifest["config"]["tasks"]:
        selected = [
            v
            for v in json.loads((root / task / "selections.json").read_text())
            if v["stream"] == "auc_selected" and v["evaluation"] == "nested_cv"
        ]
        hardware = pd.read_csv(root / task / "hardware_requirements.csv")
        hardware = hardware[hardware.stream == "auc_selected"]
        text += (
            cellrow(
                [
                    TASK_LABELS[task],
                    extent([len(v["frequency_indices_0based"]) for v in selected]),
                    extent([len(v["pattern_indices_0based"]) for v in selected]),
                    extent([v["input_features"] for v in selected]),
                    extent(hardware.n_current_electrodes),
                    extent(hardware.n_voltage_electrodes),
                    extent(hardware.n_drive_pairs),
                ]
            )
            + "\n"
        )
    text += r"\bottomrule\end{tabular}\end{table*}" + "\n"
    covariates = []
    for task in manifest["config"]["tasks"]:
        values = []
        for stream, label in [
            ("covariate_only", "clinical-only"),
            ("covariate_eis_only", "matched EIS-only LR"),
            ("covariate_combined", "combined LR"),
        ]:
            row = frame[
                (frame.task == task)
                & (frame.unit == "site")
                & (frame.evaluation == "nested_cv")
                & (frame.stream == stream)
            ].iloc[0]
            values.append(label + " " + number(row.macro_auc))
        covariates.append(TASK_LABELS[task] + ": " + ", ".join(values))
    text += (
        edit(
            "Available-covariate outer macro AUCs are "
            + "; ".join(covariates)
            + ". The matched EIS-only/combined comparison uses LR with C=1, identical frozen EIS features/preprocessing, quality eligibility and patient folds. This assesses available variables without identifying causal effects or excluding unrecorded confounding."
        )
        + "\n"
    )
    import os

    def new_figure(label, caption, panels):
        result = r"\begin{figure*}[t]\centering" + "\n"
        for name, path in panels:
            if not path.is_file():
                raise RuntimeError(f"Missing real rerun figure {path}")
            relative = os.path.relpath(path, ROOT / "manuscript/Paper")
            result += (
                r"\begin{subfigure}[t]{0.48\textwidth}\centering"
                + "\n"
                + r"\edit{\includegraphics[width=\linewidth]{"
                + relative
                + r"}}"
                + "\n"
                + r"\caption{"
                + edit(name)
                + r"}\end{subfigure}\hfill"
                + "\n"
            )
        return (
            result
            + r"\caption{"
            + edit(caption)
            + r"}\label{"
            + label
            + r"}\end{figure*}"
            + "\n"
        )

    text += new_figure(
        "fig:revision_CM",
        "New four-class AUC-selected outer predictions: reference rows and predicted columns have verified class order. Both raw counts and row-normalized fractions are shown for covered sites; quality abstentions are reported separately.",
        [
            ("Count confusion matrix", root / "plots_MC_auc/auc_selected/CM.png"),
            (
                "Row-normalized confusion matrix",
                root / "plots_MC_auc/auc_selected/CM_norm.png",
            ),
        ],
    )
    text += new_figure(
        "fig:revision_ROC",
        "New AUC-selected outer-patient ROC curves with explicitly aligned class names. Calibration and selection use training patients only; operating thresholds are not chosen from these test curves.",
        [
            (
                "Healthy versus cancer",
                root / "plots_healthy_vs_cancer_auc/auc_selected/ROC.png",
            ),
            (
                "Low versus high risk",
                root / "plots_highrisk_vs_lowrisk_auc/auc_selected/ROC.png",
            ),
        ],
    )
    text += new_figure(
        "fig:revision_clinical_curves",
        "New sensitivity-selected low/high-risk outer predictions: retrospective calibration and decision curves at study prevalence. Treat-all and treat-none references are included; these cohort curves do not establish transportability or prospective clinical net benefit.",
        [
            (
                "Calibration",
                root
                / "plots_highrisk_vs_lowrisk_sensitivity/sensitivity_selected/calibration.png",
            ),
            (
                "Decision curve",
                root
                / "plots_highrisk_vs_lowrisk_sensitivity/sensitivity_selected/decision_curve.png",
            ),
        ],
    )
    text += (
        edit(
            "All per-class confusion counts, conditional patient-cluster intervals, calibration/Brier/log-loss results, specificity-target operating points, prevalence projections, decision curves, and equal-patient/patient-endpoint estimates accompany the archived outer predictions. Selector/representation/quality/missingness ablations use the same outer patients. Paired same-model/objective comparisons against baseline use the prespecified Holm family; they remain conditional on frozen predictions and common quality coverage. These retrospective analyses do not establish prospective clinical benefit, pathology-reference accuracy, or external transportability."
        )
        + "\n"
    )
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/revision-clinical-final")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.results)
    if not root.is_absolute():
        root = ROOT / root
    manifest = json.loads((root / "run_manifest.json").read_text())
    if manifest.get("synthetic"):
        raise RuntimeError("Synthetic evidence is never imported")
    paper = ROOT / "manuscript/Paper/Revision-1.tex"
    main = paper.with_name("main.tex")
    assert hashlib.sha256(main.read_bytes()).hexdigest() == MAIN_HASH
    s = paper.read_text()
    original = s
    anchor = r"\edits{Results from testing the top parameters for frequency reduction and IIVV pattern set selections separately are reported.}"
    s = insert(s, "DATA AUDIT RESULTS", data_audit(root), anchor)
    if not args.audit_only:
        if not manifest.get("matched_covariate_control_completed"):
            raise RuntimeError(
                "Complete matched covariate controls before manuscript import"
            )
        from revision.verification import verify_results

        verify_results(root)
        s = insert(
            s,
            "NESTED CLASSIFICATION RESULTS",
            classifier_results(root, manifest),
            anchor,
        )
    paper.write_text(s)
    log = ROOT / "manuscript/result-import-manifest.json"
    record = json.loads(log.read_text()) if log.exists() else {"imports": []}
    record["imports"].append(
        dict(
            results=str(root.relative_to(ROOT)),
            audit_only=args.audit_only,
            run_manifest_sha256=hashlib.sha256(
                (root / "run_manifest.json").read_bytes()
            ).hexdigest(),
            before_tex_sha256=hashlib.sha256(original.encode()).hexdigest(),
            after_tex_sha256=hashlib.sha256(s.encode()).hexdigest(),
            source_hashes={
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [
                    root / "quality/source_quality_audit.json",
                    root / "repeatability/icc_and_inter_site_variability.csv",
                ]
                + ([] if args.audit_only else [root / "all_metrics.csv"])
            },
        )
    )
    log.write_text(json.dumps(record, indent=2) + "\n")
    print(
        "Imported marked audit results"
        + (
            " only"
            if args.audit_only
            else " and completed nested classification results"
        )
    )


if __name__ == "__main__":
    main()
