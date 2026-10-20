#!/usr/bin/env python3
"""Insert completed, real rerun evidence in author responses; preserve comments."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TASKS = {
    "healthy_vs_cancer": "healthy/cancer",
    "cancer_vs_lr_hr": "lesion three-class",
    "MC": "four-class",
    "highrisk_vs_lowrisk": "low/high risk",
}


def n(value, digits=3):
    return f"{value:.{digits}f}" if pd.notna(value) else "undefined"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/revision-clinical-final")
    args = parser.parse_args()
    root = Path(args.results)
    root = root if root.is_absolute() else ROOT / root
    manifest = json.loads((root / "run_manifest.json").read_text())
    if manifest["status"] != "completed" or manifest.get("synthetic"):
        raise ValueError("Completed real clinical evidence is required")
    if not manifest.get("matched_covariate_control_completed"):
        raise ValueError("Finish the matched clinical-covariate controls first")
    from revision.verification import verify_results

    verify_results(root)
    cfg = manifest["config"]
    frame = pd.read_csv(root / "all_metrics.csv")
    site = frame[(frame.unit == "site") & (frame.evaluation == "nested_cv")]
    perclass = pd.concat(
        [pd.read_csv(root / t / "per_class_metrics.csv") for t in TASKS]
    )

    def row(task, stream, unit="site"):
        result = frame[
            (frame.task == task)
            & (frame.stream == stream)
            & (frame.unit == unit)
            & (frame.evaluation == "nested_cv")
        ]
        if len(result) != 1:
            raise ValueError(f"No unique metric row: {task}/{stream}/{unit}")
        return result.iloc[0]

    def ci(task, stream, metric="macro_auc", unit="site"):
        result = (
            pd.read_csv(
                root / task / f"nested_cv_{stream}_{unit}_confidence_intervals.csv"
            )
            .set_index("metric")
            .loc[metric]
        )
        return n(result.ci_low) + "--" + n(result.ci_high)

    def primary(task):
        r = row(task, "auc_selected")
        return f'{TASKS[task]}: macro AUC {n(r.macro_auc)} (conditional 95\\% interval {ci(task,"auc_selected")}), accuracy {n(r.accuracy)}, coverage {n(r.coverage)}'

    overview = "; ".join(primary(t) for t in TASKS) + "."
    evidence = {}
    evidence["R1-General"] = overview
    evidence["R1-C1"] = overview
    evidence["R2-General"] = primary("highrisk_vs_lowrisk") + "."
    quality = []
    for stream in site[
        (site.task == "healthy_vs_cancer")
        & site.stream.str.startswith("ablation_quality_")
    ].stream:
        r = row("healthy_vs_cancer", stream)
        cutoff = stream.removeprefix("ablation_quality_")
        quality.append(f"{cutoff}\\%: AUC {n(r.macro_auc)}, coverage {n(r.coverage)}")
    evidence["R1-C2"] = evidence["R1-C4"] = (
        "For the fixed-reference SVM quality ablation in healthy/cancer, "
        + "; ".join(quality)
        + ". These are outer-patient covered-case AUCs with abstention coverage, not a clinical acceptability standard."
    )
    missing = row("healthy_vs_cancer", "ablation_missingness")
    eis = row("healthy_vs_cancer", "auc_selected")
    evidence["R1-C3"] = evidence["R1-C5"] = (
        f"The missingness-only healthy/cancer control has outer macro AUC {n(missing.macro_auc)} at coverage {n(missing.coverage)}, versus {n(eis.macro_auc)} for the AUC-selected EIS pipeline. Measurement-failure information can therefore contribute to discrimination; neither result establishes pathology-specific causation."
    )
    selector = []
    for method in ["pca", "anova", "mutual_info", "sparse", "rfe", "permutation"]:
        r = row("MC", "ablation_selector_" + method)
        selector.append(method.replace("_", " ") + ": " + n(r.macro_auc))
    evidence["R1-C6"] = evidence["R3-C3"] = (
        "Four-class fixed-reference SVM selector macro AUCs are "
        + "; ".join(selector)
        + ". The comparisons use training-fitted rankings and common outer patients."
    )
    representation = []
    for method in ["magnitude", "real_imag", "magnitude_phase", "ratio", "circuit"]:
        r = row("MC", "ablation_representation_" + method)
        representation.append(method.replace("_", " ") + ": " + n(r.macro_auc))
    evidence["R1-C9"] = (
        "Four-class representation/ranking-pipeline macro AUCs are "
        + "; ".join(representation)
        + ". These do not isolate phase at a common fixed frequency set or establish circuit physiology."
    )
    units = []
    for unit, label in [
        ("site", "pooled sites"),
        ("patient_weighted_site", "equal-patient sites"),
        ("patient", "any-positive patient endpoint"),
    ]:
        r = row("healthy_vs_cancer", "auc_selected", unit)
        units.append(
            f"{label}: AUC {n(r.macro_auc)}, {int(r.n_covered)}/{int(r.n_sites)} covered units"
        )
    evidence["R1-C10"] = (
        "For healthy/cancer, "
        + "; ".join(units)
        + ". The any-positive endpoint is specific to this surgical cohort and is not a screening-population estimate."
    )
    cov = []
    for task in TASKS:
        parts = [
            f"{name} {n(row(task,stream).macro_auc)}"
            for stream, name in [
                ("covariate_only", "clinical-only"),
                ("covariate_eis_only", "matched EIS-only LR"),
                ("covariate_combined", "combined LR"),
            ]
        ]
        cov.append(TASKS[task] + ": " + ", ".join(parts))
    evidence["R1-C12"] = evidence["R1-C13"] = evidence["R1-C14"] = (
        "Matched outer macro AUC comparisons are "
        + "; ".join(cov)
        + ". The EIS-only/combined LR comparison uses identical selected EIS features, C=1, quality eligibility and patient folds; unrecorded covariates remain uncontrolled."
    )
    dysplasia = []
    for task in ["cancer_vs_lr_hr", "MC"]:
        r = perclass[
            (perclass.task == task)
            & (perclass.stream == "auc_selected")
            & (perclass.evaluation == "nested_cv")
            & (perclass.unit == "site")
            & (perclass.class_name == "High-grade dysplasia")
        ].iloc[0]
        cm = pd.read_csv(
            root / task / "nested_cv_auc_selected_site_confusion_counts.csv",
            index_col=0,
        )
        errors = ", ".join(
            f'{name} {int(cm.loc["High-grade dysplasia",name])}'
            for name in cm
            if name != "High-grade dysplasia"
        )
        intervals = pd.read_csv(
            root
            / task
            / "nested_cv_auc_selected_site_per_class_confidence_intervals.csv"
        )
        interval = intervals[
            (intervals.class_name == "High-grade dysplasia")
            & (intervals.metric == "sensitivity")
        ].iloc[0]
        dysplasia.append(
            f"{TASKS[task]}: {int(r.tp)} TP/{int(r.fn)} FN among {int(r.support)} covered dysplasia sites, sensitivity {n(r.sensitivity)} (conditional 95\\% interval {n(interval.ci_low)}--{n(interval.ci_high)}), specificity {n(r.specificity)}, PPV {n(r.precision)}, NPV {n(r.npv)}, F1 {n(r.f1)}, one-vs-rest AUC {n(r.auc)}; FN destinations: {errors}"
        )
    evidence["R1-C15"] = evidence["R1-C16"] = "; ".join(dysplasia) + "."
    comparisons = pd.read_csv(root / "paired_comparisons_holm.csv")
    pcol = "p_holm_all_tasks_models_objectives"
    significance = f"The full prespecified family contains {len(comparisons)} same-model/objective baseline contrasts; {int((comparisons[pcol]<.05).sum())} have adjusted p below 0.05. Results are conditional paired-prediction comparisons on common quality coverage, not confirmatory refitting or external-validation inference."
    evidence["R1-C17"] = evidence["R1-C18"] = significance
    evidence["R1-C19"] = (
        "Patient-cluster AUC intervals for the primary streams are "
        + "; ".join(TASKS[t] + " " + ci(t, "auc_selected") for t in TASKS)
        + f'. Each interval uses {cfg["bootstrap_samples"]} patient-block draws; valid-draw counts and per-class intervals are archived.'
    )
    sizes = []
    hardware = []
    timings = []
    stability = []
    for task in TASKS:
        selections = json.loads((root / task / "selections.json").read_text())
        selected = [
            v
            for v in selections
            if v["stream"] == "auc_selected" and v["evaluation"] == "nested_cv"
        ]
        vals = [v["input_features"] for v in selected]
        sizes.append(f"{TASKS[task]} {min(vals)}--{max(vals)} classifier features")
        hw = pd.read_csv(root / task / "hardware_requirements.csv")
        hw = hw[hw.stream == "auc_selected"]
        hardware.append(
            f"{TASKS[task]}: {int(hw.n_current_electrodes.min())}--{int(hw.n_current_electrodes.max())} current electrodes, {int(hw.n_voltage_electrodes.min())}--{int(hw.n_voltage_electrodes.max())} voltage electrodes, {int(hw.n_drive_pairs.min())}--{int(hw.n_drive_pairs.max())} drive pairs"
        )
        timing = pd.read_csv(root / task / "measured_computation_times.csv")
        timing = timing[
            (timing.stream == "auc_selected") & (timing.eligible_test_sites > 0)
        ]
        milliseconds = 1000 * timing.test_predict_seconds / timing.eligible_test_sites
        timings.append(TASKS[task] + " " + n(milliseconds.median(), 4) + " ms")
        stab = pd.read_csv(root / task / "ranking_stability.csv")
        stab = stab[stab.stream == "auc_selected"]
        stability.append(TASKS[task] + " " + n(stab.frequency_jaccard.mean()))
    evidence["R1-C20"] = (
        "The primary outer refits use "
        + "; ".join(sizes)
        + ". Circuit descriptors can use few classifier features while requiring the full selected raw spectrum; scalar reduction is not equivalent to acquisition or channel reduction."
    )
    evidence["R1-C7"] = evidence["R3-C4"] = (
        "Mean pairwise frequency-set Jaccard values across the primary outer refits are "
        + "; ".join(stability)
        + ". These describe stability of fitted coordinate sets, not causal frequency mechanisms or independent replicates."
    )
    evidence["R1-C21"] = (
        "Each outer refit stores its training-only raw-impedance threshold, quantile and frozen pattern IDs in selections.json; the same held-out spectra are evaluated without recomputing the mask from test patients. Actual coordinate sets and dimensionalities are reported in Table "
        + r"\loc{tab:revision_hardware}"
        + "."
    )
    evidence["R1-C26"] = (
        "Median amortized batch-prediction time per covered test site is "
        + "; ".join(timings)
        + ". Feature-transform/preparation times are separate; caching and simultaneous tasks affect these software benchmarks. They are not individual-device latency or prospective acquisition measurements."
    )
    evidence["R1-C27"] = (
        "Primary refit requirements are "
        + "; ".join(hardware)
        + ". These are software-selected coordinate requirements, not a validated reduced-channel instrument."
    )
    referral = []
    for task in ["healthy_vs_cancer", "highrisk_vs_lowrisk"]:
        r = row(task, "sensitivity_selected")
        flow = json.loads(
            (
                root / task / "nested_cv_sensitivity_selected_referral_workflow.json"
            ).read_text()
        )
        referral.append(
            f'{TASKS[task]}: sensitivity {n(r.sensitivity)}, specificity {n(r.specificity)}, PPV {n(r.precision)}, NPV {n(r.npv)}, coverage {n(r.coverage)}, fallback referral fraction {n(flow["referral_fraction"])}'
        )
    evidence["R1-C30"] = evidence["R2-C4"] = evidence["R2-C5"] = (
        "Sensitivity-selected outer operating points are "
        + "; ".join(referral)
        + ". The 0.95 training sensitivity target is not a guaranteed held-out result or a validated clinical safety requirement."
    )
    evidence["R2-C3"] = primary("highrisk_vs_lowrisk") + ". " + evidence["R2-C4"]
    evidence["R3-C2"] = evidence["R1-C20"]
    evidence["R3-M2"] = primary("MC") + ". " + evidence["R1-C16"]
    baseline = json.loads((ROOT / "manuscript/revision-baseline.json").read_text())
    ids = [v["id"] for v in baseline["original_action_rows"]]
    letter = ROOT / "manuscript/Response_Letter/resposeletter.tex"
    s = letter.read_text()
    summary = (
        r"\noindent\textbf{Completed clinical reanalysis.} "
        + overview
        + " "
        + significance
        + " "
        + f'The run uses {cfg["outer_folds"]} outer and {cfg["inner_folds"]} inner patient folds, {cfg["outer_repeats"]} outer partitions, and fixed seed {cfg["seed"]}. '
        + r"These estimates replace neither prospective validation nor missing pathology/contact records. See the marked revised manuscript, \loc{sub:revision_results}, \loc{tab:revision_nested}, and \loc{tab:revision_referral}; complete predictions, per-class metrics and conditional intervals accompany the code. A matched EIS-only LR control has also been completed for the available-covariate comparisons."
    )
    s = re.sub(
        r"% BEGIN REANALYSIS SUMMARY.*?% END REANALYSIS SUMMARY",
        lambda _: "% BEGIN REANALYSIS SUMMARY\n"
        + summary
        + "\n% END REANALYSIS SUMMARY",
        s,
        flags=re.S,
    )
    matches = list(re.finditer(r"\\begin\{response\}.*?\\end\{response\}", s, re.S))
    assert len(matches) == len(ids) == 55
    for key, match in reversed(list(zip(ids, matches))):
        if key not in evidence:
            continue
        block = match.group()
        block = re.sub(
            r"% BEGIN GENERATED RERUN EVIDENCE.*?% END GENERATED RERUN EVIDENCE\n",
            "",
            block,
            flags=re.S,
        )
        extra = (
            "% BEGIN GENERATED RERUN EVIDENCE\n"
            + r"\par\smallskip\textbf{Completed rerun evidence.} "
            + evidence[key]
            + r"\par\smallskip\textbf{New result location:} \loc{sub:revision_results}, Tables \loc{tab:revision_nested} and \loc{tab:revision_referral}; archived task-level reports."
            + "\n% END GENERATED RERUN EVIDENCE\n"
        )
        block = block.replace(r"\end{response}", extra + r"\end{response}")
        s = s[: match.start()] + block + s[match.end() :]
    letter.write_text(s)
    (ROOT / "manuscript/response-result-import.json").write_text(
        json.dumps(
            dict(
                run_manifest_sha256=hashlib.sha256(
                    (root / "run_manifest.json").read_bytes()
                ).hexdigest(),
                metrics_sha256=hashlib.sha256(
                    (root / "all_metrics.csv").read_bytes()
                ).hexdigest(),
                comments_with_numeric_evidence=sorted(evidence),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        "Updated completed summary and",
        len(evidence),
        "comment-specific rerun evidence blocks",
    )


if __name__ == "__main__":
    main()
