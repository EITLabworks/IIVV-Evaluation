#!/usr/bin/env python3
"""Foreground completed nested results in the marked abstract and conclusion."""

import argparse
from pathlib import Path
import json, hashlib
import sys
import pandas as pd
from marked_manuscript import MarkedManuscript, ROOT

sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="results/revision-clinical-final")
    a = p.parse_args()
    root = Path(a.results)
    root = root if root.is_absolute() else ROOT / root
    manifest = json.loads((root / "run_manifest.json").read_text())
    if (
        manifest["status"] != "completed"
        or manifest.get("synthetic")
        or not manifest.get("matched_covariate_control_completed")
    ):
        raise ValueError("Complete real primary and matched-control analyses first")
    from revision.verification import verify_results

    verify_results(root)
    cfg = manifest["config"]
    metrics = pd.read_csv(root / "all_metrics.csv")
    site = metrics[
        (metrics.unit == "site") & (metrics.evaluation == "nested_cv")
    ].set_index(["task", "stream"])

    def r(task, stream="auc_selected"):
        return site.loc[(task, stream)]

    binary = r("healthy_vs_cancer")
    referral = r("highrisk_vs_lowrisk")
    missing = r("healthy_vs_cancer", "ablation_missingness")
    targets = [
        ("healthy_vs_cancer", "healthy/cancer"),
        ("cancer_vs_lr_hr", "three-class lesions"),
        ("MC", "four-class tissue"),
        ("highrisk_vs_lowrisk", "low/high risk"),
    ]
    aucs = ", ".join(
        label + " " + f"{r(task).macro_auc:.3f}" for task, label in targets
    )
    m = MarkedManuscript()
    before = hashlib.sha256(m.text.encode()).hexdigest()
    marker = ROOT / "manuscript/abstract-result-import.json"
    if marker.exists():
        raise ValueError(
            "Abstract results already imported; preserve prior additions before a changed import"
        )
    m.replace(
        "The supplied analysis dataset contains 276 measured sites from 100 distinct patients undergoing oral cancer resection or biopsy; the earlier count of 104 could not be reproduced from either exported identifiers or original cohort/case pairs.",
        "Complex spectra from 276 sites in 100 patients undergoing oral lesion resection or biopsy were analyzed.",
    )
    m.replace(
        "Three classification tasks were evaluated: (1) healthy vs. cancer, (2) multi-class lesion-type discrimination (cancer, high-grade dysplasia, non-malignant), and (3) multi-class discrimination between the three lesion pathologies and healthy tissue.",
        "Four tasks address healthy/cancer discrimination, three-class lesion classification, four-class tissue classification, and a combined low-risk/high-risk referral target.",
    )
    m.replace(
        "For each task, signal frequencies were independently ranked and reduced using PCA, and different current injection/voltage measurement (IIVV) pattern geometries were tested.",
        "Frequency selectors, current-injection/voltage-measurement (IIVV) masks, complex feature representations, preprocessing and classifiers were compared within training data.",
    )
    m.replace(
        "Classification performance was explored using five-fold leave-one-patient-group-out cross-validation (LOPGO-CV); preprocessing and configuration selection were not enclosed in an independent outer validation loop, so the reported estimates may be optimistic.",
        f'Primary performance uses {cfg["outer_folds"]} outer and {cfg["inner_folds"]} inner patient-grouped folds across {cfg["outer_repeats"]} outer partitions, with training-only selection, calibration and operating thresholds; historical non-nested estimates remain separate.',
    )
    m.replace(
        "Input dimensionality was reduced by more than 99\\% in the selected configurations, with apparent discrimination maintained or improved relative to the corresponding full-input models in this exploratory analysis.",
        "Selected feature sets vary across outer folds; reduced classifier dimension does not by itself establish acquisition or hardware savings.",
    )
    m.replace(
        "The selected binary logistic regression configuration had a reported accuracy of 80\\% and AUC of 0.90, while selected multiclass configurations had AUCs of 0.82 and 0.83; these are retrospective estimates requiring independent validation.",
        "AUC-selected outer pipelines yield macro AUCs of "
        + aucs
        + f". Healthy/cancer accuracy is {binary.accuracy:.3f} at {100*binary.coverage:.1f}\\% coverage.",
    )
    m.replace(
        "All top-performing models utilized the significantly reduced IIVV set as input.",
        f"A missingness-only healthy/cancer control yields AUC {missing.macro_auc:.3f}, highlighting acquisition-quality confounding that limits pathology-specific interpretation.",
    )
    old = r"Using optimized subsets of frequencies and IIVV patterns, parsimonious and interpretable ML models achieved promising but preliminary discrimination, with AUCs up to \num{0.90} for healthy vs cancer classification and \num{0.83} for multiclass discrimination."
    m.replace(
        old,
        "The completed patient-nested AUC-selected pipelines yield macro AUCs of "
        + aucs
        + ". These retrospective covered-case estimates must be interpreted with quality coverage, class-specific errors and conditional patient-cluster intervals.",
    )
    old = "Notably, classification performance for severe dysplasia and non-malignant lesions was lower in \\textit{Tasks 2} and \\textit{3}, highlighting the need for expanded datasets with more non-cancerous samples."
    values = []
    for task in ["cancer_vs_lr_hr", "MC"]:
        pc = pd.read_csv(root / task / "per_class_metrics.csv")
        row = pc[
            (pc.stream == "auc_selected")
            & (pc.unit == "site")
            & (pc.evaluation == "nested_cv")
            & (pc.class_name == "High-grade dysplasia")
        ].iloc[0]
        values.append(
            f'{task.replace("cancer_vs_lr_hr","three-class").replace("MC","four-class")}: {row.sensitivity:.3f} sensitivity, {int(row.fn)} missed among {int(row.support)} covered sites'
        )
    m.replace(
        old,
        "High-grade dysplasia remains a separate clinical limitation ("
        + "; ".join(values)
        + "). The combined referral task and sensitivity-oriented evaluation do not establish a safe biopsy or referral rule.",
    )
    anchor = r"\label{sec:discussion}"
    text = f"The new nested analysis should be interpreted independently of the retained historical tables. Healthy/cancer macro AUC is {binary.macro_auc:.3f} at {100*binary.coverage:.1f}\\% coverage; the missingness-only control reaches {missing.macro_auc:.3f}. Measurement-quality patterns therefore carry substantial predictive information, and the observed discrimination cannot be attributed exclusively to tissue electrophysiology. The combined low/high-risk macro AUC is {referral.macro_auc:.3f}. Class-specific errors, sensitivity-selected operating points, matched available-covariate controls, and abstention/referral reports qualify these estimates. None replaces independent pathology, contact-control, prospective utility, or external-validation evidence."
    # Insert after the label without altering it; all added prose is marked.
    assert m.text.count(anchor) == 1
    new = anchor + "\n" + r"\edit{" + text + "}\n"
    m.text = m.text.replace(anchor, new)
    m.changes.append(dict(old=anchor, new=new))
    m.save()
    marker.write_text(
        json.dumps(
            dict(
                metrics_sha256=hashlib.sha256(
                    (root / "all_metrics.csv").read_bytes()
                ).hexdigest(),
                before_tex_sha256=before,
                after_tex_sha256=hashlib.sha256(m.text.encode()).hexdigest(),
                source=str(root.relative_to(ROOT)),
                marked_changes=len(m.changes),
            ),
            indent=2,
        )
        + "\n"
    )
    print("Imported marked abstract/discussion/conclusion results")


if __name__ == "__main__":
    main()
