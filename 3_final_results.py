#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    recall_score,
    confusion_matrix,
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# from sklearn.model_selection import GroupKFold
from sklearn.model_selection import StratifiedGroupKFold as GroupKFold
from sklearn.svm import SVC

from toolbox import Masking
from toolbox.results import Results_data_loading, ResultsROCAUC
from toolbox.visualization import plot_confusion_matrix_custom


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

MASK_SEL_ALL = {
    "All": "all",
    "Z-threshold": "z_threshold",  # special-case
    "Long a+": "vv_long_across",
    "Long a+ ext.": "vv_long_across_extended",
    "Med. a+ ext.": "vv_medium_across_extended",
    "Skip1 close": "vv_distance_close_skip_1",
    "Adj. close": "adjacent_close",
    "Med. adj.": "vv_distance_medium_adj",
    "Adj. far": "adjacent_far",
    "Skip1 medium": "skip1_medium",
    "Skip1 far": "skip1_far",
    "Opp. close": "opposite_close",
    "Opp. medium": "opposite_medium",
    "Opp. far": "opposite_far",
}


def parse_topN_from_fsel(fsel: str) -> int:
    m = re.search(r"(\d+)", str(fsel))
    if not m:
        raise ValueError(f"Cannot parse Top-N from fsel='{fsel}'")
    return int(m.group(1))


def load_freq_indices(
    results_path: str, class_task: str, metric: str, n_freq: int
) -> np.ndarray:
    p = Path(results_path) / f"FreqSel_ranking_{class_task}_{metric}.json"
    if not p.exists():
        raise FileNotFoundError(p)

    j = json.loads(p.read_text(encoding="utf-8"))
    order = j["order_good_to_bad"]
    return np.array(order[:n_freq], dtype=int) - 1  # convert to 0-based


def load_best_trial(csv_path: Path) -> Dict:
    df = pd.read_csv(csv_path)
    best = df.loc[df["score_mean"].idxmax()]

    params = json.loads(best["params_json"])

    return {
        "fsel": best["fsel"],
        "msel": best["msel"],
        "params": params,
    }


def make_model(name: str, params: Dict):
    name = name.lower()

    if name == "svm":
        return SVC(probability=True, random_state=0, **params)
    if name == "rf":
        return RandomForestClassifier(random_state=0, **params)
    if name == "lr":
        params.setdefault("max_iter", 1000)
        return LogisticRegression(random_state=0, **params)

    raise ValueError(name)


def z_threshold_for_task(class_task: str) -> Optional[int]:
    if class_task == "healthy_vs_cancer":
        return 424
    if class_task == "cancer_vs_lr_hr":
        return 314
    if class_task == "MC":
        return 212
    return None


# ------------------------------------------------------------
# core evaluation
# ------------------------------------------------------------


def run_model(results_path, outdir, class_task, metric, model_name):

    csv_path = Path(results_path) / f"GPU_tuning_{model_name}_{class_task}_{metric}.csv"
    best = load_best_trial(csv_path)

    n_freq = parse_topN_from_fsel(best["fsel"])
    freq_idx = load_freq_indices(results_path, class_task, metric, n_freq)

    mask_idx = best["msel"]

    print(f"\n=== {model_name.upper()} ===")
    print("Top N freq:", n_freq)
    print("Mask:", best["msel"])
    print("Params:", best["params"])

    # load data
    X, Y_labels, _, Case, _, _ = Results_data_loading(
        class_task, data_augmentation=False
    )

    if class_task == "healthy_vs_cancer":
        class_names = ["Healthy", "Cancer"]
    elif class_task == "cancer_vs_lr_hr":
        class_names = ["Cancer", "Low-Risk", "High-Risk"]
    elif class_task == "MC":
        class_names = ["Cancer", "Healthy", "High-grade\ndysplasia", "Non-malignant"]

    # masking
    IIVV = np.loadtxt("source/IIVV_patterns.txt", dtype=int)
    mask = Masking(IIVV, plot=False, prints=False)
    mask.IIVV_mask(MASK_SEL_ALL[best["msel"]], plot=False)
    print(f"Selected mask: {MASK_SEL_ALL[best['msel']]} (index {mask_idx})")
    print(f"Seleted freq indices (0-based): {freq_idx}")

    X_tmp = X[:, freq_idx, :]
    X_tmp = X_tmp[:, :, mask.mask]
    X_tmp = X_tmp.reshape(X_tmp.shape[0], -1)

    clf = make_model(model_name, best["params"])

    # CV
    cv = GroupKFold(n_splits=5)

    y_true, y_pred, y_score = [], [], []

    for tr, te in cv.split(X_tmp, Y_labels, Case):
        clf.fit(X_tmp[tr], Y_labels[tr])
        probs = clf.predict_proba(X_tmp[te])
        preds = np.argmax(probs, axis=1)

        y_true.extend(Y_labels[te])
        y_pred.extend(preds)
        y_score.extend(probs)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)

    # ---- NEW: compute metrics + save CSV ----
    n_classes = len(np.unique(y_true))

    acc = float(accuracy_score(y_true, y_pred))

    if n_classes == 2:
        f1 = float(f1_score(y_true, y_pred))
        auc = float(roc_auc_score(y_true, y_score[:, 1]))
        recall = float(recall_score(y_true, y_pred))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        spec = float(tn / (tn + fp))
    else:
        f1 = float(f1_score(y_true, y_pred, average="macro"))
        auc = float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))
        recall = float(recall_score(y_true, y_pred, average="macro"))
        spec = float("nan")

    metrics_df = pd.DataFrame(
        [
            {
                "task": class_task,
                "model": model_name,
                "metric_opt": metric,
                "accuracy": acc,
                "f1": f1,
                "auc": auc,
                "recall": recall,
                "specificity": spec,
            }
        ]
    )

    Path(outdir).mkdir(parents=True, exist_ok=True)

    tag = f"{class_task}_{model_name}_{metric}"

    ResultsROCAUC(
        class_task,
        y_true,
        y_score,
        Y_labels,
        s_name=str(Path(outdir) / f"ROC_{tag}.pdf"),
    )

    plot_confusion_matrix_custom(
        y_true,
        y_pred,
        class_names,
        normalize=False,
        path=str(Path(outdir) / f"CM_{tag}.pdf"),
    )

    plot_confusion_matrix_custom(
        y_true,
        y_pred,
        class_names,
        normalize=True,
        path=str(Path(outdir) / f"CM_{tag}_norm.pdf"),
    )

    ResultsROCAUC(
        class_task,
        y_true,
        y_score,
        Y_labels,
        s_name=str(Path(outdir) / f"ROC_{tag}.png"),
    )

    plot_confusion_matrix_custom(
        y_true,
        y_pred,
        class_names,
        normalize=False,
        path=str(Path(outdir) / f"CM_{tag}.png"),
    )

    plot_confusion_matrix_custom(
        y_true,
        y_pred,
        class_names,
        normalize=True,
        path=str(Path(outdir) / f"CM_{tag}_norm.png"),
    )
    metrics_path = Path(outdir) / f"METRICS_{tag}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print("Metrics CSV saved:", metrics_path)

    print("Plots saved.")


# ------------------------------------------------------------
# main
# ------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--task", required=True, choices=["healthy_vs_cancer", "cancer_vs_lr_hr", "MC"]
    )
    ap.add_argument("--metric", required=True)
    ap.add_argument("--model", default="all", choices=["all", "rf", "lr", "svm"])
    ap.add_argument("--results-path", default="results")
    ap.add_argument("--outdir", default=None)

    args = ap.parse_args()
    class_task = args.task
    outdir = args.outdir or f"{args.results_path}/plots_{class_task}_{args.metric}"

    models = ["rf", "lr", "svm"] if args.model == "all" else [args.model]

    for m in models:
        run_model(args.results_path, outdir, class_task, args.metric, m)


if __name__ == "__main__":
    main()
