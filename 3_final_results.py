#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    recall_score,
    precision_score,
    confusion_matrix,
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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


def load_best_trial(csv_path: Path) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    best = df.loc[df["score_mean"].idxmax()]
    params = json.loads(best["params_json"])
    return {"fsel": best["fsel"], "msel": best["msel"], "params": params}


def make_model(name: str, params: Dict[str, Any]):
    name = name.lower()
    if name == "svm":
        return SVC(probability=True, random_state=0, **params)
    if name == "rf":
        return RandomForestClassifier(random_state=0, **params)
    if name == "lr":
        params.setdefault("max_iter", 1000)
        return LogisticRegression(random_state=0, **params)
    raise ValueError(name)


def class_names_for_task(class_task: str) -> List[str]:
    if class_task == "healthy_vs_cancer":
        return ["Healthy", "Cancer"]
    if class_task == "cancer_vs_lr_hr":
        return ["Cancer", "Low-Risk", "High-Risk"]
    if class_task == "MC":
        return ["Cancer", "Healthy", "High-grade\ndysplasia", "Non-malignant"]
    raise ValueError(f"Unknown task: {class_task}")


def choose_pos_label_for_task(class_task: str, global_classes: np.ndarray) -> int:
    """
    Deterministic positive class choice for binary ROC/AUC.
    If you want "Cancer" to be positive, set it explicitly here.
    """
    if len(global_classes) != 2:
        raise ValueError("choose_pos_label_for_task called for non-binary task.")
    return int(global_classes[-1])


def align_proba_to_global_classes(
    probs: np.ndarray,
    model_classes: np.ndarray,
    global_classes: np.ndarray,
) -> np.ndarray:
    """
    Reorder probability columns so they match global_classes exactly.
    Raises if anything is missing.
    """
    probs = np.asarray(probs)
    model_classes = np.asarray(model_classes)
    global_classes = np.asarray(global_classes)

    if probs.ndim != 2:
        raise ValueError(f"probs must be 2D (N,K), got shape {probs.shape}")

    if probs.shape[1] != len(model_classes):
        raise ValueError(
            f"probs has {probs.shape[1]} columns but model_classes has {len(model_classes)} labels."
        )

    missing = [c for c in global_classes if c not in set(model_classes.tolist())]
    if missing:
        raise ValueError(
            f"Missing classes in model output: {missing}. "
            f"model_classes={model_classes.tolist()} global_classes={global_classes.tolist()}"
        )

    col_idx = [int(np.where(model_classes == c)[0][0]) for c in global_classes]
    return probs[:, col_idx]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score_aligned: np.ndarray,
    global_classes: np.ndarray,
    pos_label: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute metrics safely. y_score_aligned columns MUST be ordered as global_classes.
    """
    global_classes = np.asarray(global_classes)
    n_classes = int(len(global_classes))

    acc = float(accuracy_score(y_true, y_pred))

    if n_classes == 2:
        if pos_label is None:
            pos_label = int(global_classes[-1])

        pos_idx = int(np.where(global_classes == pos_label)[0][0])
        y_true_bin = (y_true == pos_label).astype(int)

        f1 = float(f1_score(y_true, y_pred, pos_label=pos_label))
        aucv = float(roc_auc_score(y_true_bin, y_score_aligned[:, pos_idx]))
        recall = float(recall_score(y_true, y_pred, pos_label=pos_label))
        precision = float(
            precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        )

        neg_label = int([c for c in global_classes.tolist() if c != pos_label][0])
        cm = confusion_matrix(y_true, y_pred, labels=[neg_label, pos_label])
        tn, fp, fn, tp = cm.ravel()

        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
        npv = float(tn / (tn + fn)) if (tn + fn) > 0 else np.nan

        return dict(
            accuracy=acc,
            f1=f1,
            auc=aucv,
            recall=recall,
            specificity=spec,
            precision=precision,
            npv=npv,
        )

    # multiclass
    f1 = float(f1_score(y_true, y_pred, average="micro"))
    aucv = float(
        roc_auc_score(
            y_true,
            y_score_aligned,
            multi_class="ovr",
            average="micro",
            labels=global_classes,
        )
    )
    recall = float(recall_score(y_true, y_pred, average="micro"))
    precision = float(precision_score(y_true, y_pred, average="micro", zero_division=0))

    return dict(
        accuracy=acc,
        f1=f1,
        auc=aucv,
        recall=recall,
        specificity=np.nan,
        precision=precision,
        npv=np.nan,
    )


def append_final_results(row: Dict[str, Any], final_csv_path: Path) -> None:
    """
    Append one row to FinalResults.csv (create with header if missing),
    enforcing the required column order.
    """
    cols = [
        "Task",
        "Model",
        "Tuning metric",
        "Freq",
        "IIVV mask",
        "N_params",
        "accuracy",
        "F1",
        "Auc",
        "Recall",
        "Precision",
    ]

    final_csv_path.parent.mkdir(parents=True, exist_ok=True)

    df_row = pd.DataFrame([row], columns=cols)
    write_header = not final_csv_path.exists()
    df_row.to_csv(final_csv_path, mode="a", header=write_header, index=False)


# ------------------------------------------------------------
# core evaluation
# ------------------------------------------------------------


def run_model(
    results_path: str, outdir: str, class_task: str, metric: str, model_name: str
) -> None:
    csv_path = Path(results_path) / f"GPU_tuning_{model_name}_{class_task}_{metric}.csv"
    best = load_best_trial(csv_path)

    n_freq = parse_topN_from_fsel(best["fsel"])
    freq_idx = load_freq_indices(results_path, class_task, metric, n_freq)

    print(f"\n=== {model_name.upper()} ===")
    print("Top N freq:", n_freq)
    print("Mask:", best["msel"])
    print("Params:", best["params"])

    # load data
    X, Y_labels, _, Case, _, _ = Results_data_loading(
        class_task, data_augmentation=False
    )

    # fixed global class order (used everywhere)
    global_classes = np.array(sorted(np.unique(Y_labels))).astype(int)
    pos_label: Optional[int] = None
    if len(global_classes) == 2:
        pos_label = choose_pos_label_for_task(class_task, global_classes)

    class_names = class_names_for_task(class_task)

    # masking
    IIVV = np.loadtxt("source/IIVV_patterns.txt", dtype=int)
    mask = Masking(IIVV, plot=False, prints=False)
    mask.IIVV_mask(MASK_SEL_ALL[best["msel"]], plot=False)
    print(f"Selected mask: {MASK_SEL_ALL[best['msel']]} (index {best['msel']})")
    print(f"Selected freq indices (0-based): {freq_idx}")

    X_tmp = X[:, freq_idx, :]
    X_tmp = X_tmp[:, :, mask.mask]
    X_tmp = X_tmp.reshape(X_tmp.shape[0], -1)

    clf = make_model(model_name, best["params"])

    # CV
    cv = GroupKFold(n_splits=5)

    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    y_score_all: List[np.ndarray] = []
    backends: List[str] = []  # keep for parity / debug

    for tr, te in cv.split(X_tmp, Y_labels, Case):
        clf.fit(X_tmp[tr], Y_labels[tr])

        probs_raw = clf.predict_proba(X_tmp[te])
        model_classes = np.asarray(clf.classes_).astype(int)

        # align proba to global class order
        probs = align_proba_to_global_classes(probs_raw, model_classes, global_classes)

        # predictions from aligned probs (NEVER argmax->label)
        preds = global_classes[np.argmax(probs, axis=1)].astype(int)

        y_true_all.extend(Y_labels[te].astype(int).tolist())
        y_pred_all.extend(preds.tolist())
        y_score_all.extend(list(probs))
        backends.append("cpu")

    y_true = np.asarray(y_true_all, dtype=int)
    y_pred = np.asarray(y_pred_all, dtype=int)
    y_score = np.asarray(y_score_all, dtype=float)

    # metrics
    m = compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_score_aligned=y_score,
        global_classes=global_classes,
        pos_label=pos_label,
    )

    metrics_df = pd.DataFrame(
        [
            {
                "task": class_task,
                "model": model_name,
                "metric_opt": metric,
                "accuracy": m["accuracy"],
                "f1": m["f1"],
                "auc": m["auc"],
                "recall": m["recall"],
                "specificity": m["specificity"],
                "precision": m["precision"],
                "npv": m["npv"],
                "pos_label": (pos_label if pos_label is not None else ""),
                "global_classes": ",".join(map(str, global_classes.tolist())),
            }
        ]
    )

    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)

    tag = f"{class_task}_{model_name}_{metric}"

    # ------------------------------------------------------------
    # Append summary to results/FinalResults.csv
    # ------------------------------------------------------------
    final_csv = Path(results_path) / "FinalResults.csv"

    final_row = {
        "Task": class_task,
        "Model": model_name,
        "Tuning metric": metric,
        "Freq": best["fsel"],  # e.g., "Top-50" / whatever your fsel string is
        "IIVV mask": best["msel"],  # the selected mask label from tuning CSV
        "N_params": int(X_tmp.shape[1]),
        "accuracy": float(m["accuracy"]),
        "F1": float(m["f1"]),
        "Auc": float(
            m["auc"]
        ),  # using your computed AUC but keeping requested column name "Acc"
        "Recall": float(m["recall"]),
        "Precision": float(m["precision"]),
    }

    append_final_results(final_row, final_csv)
    print("Final results appended:", final_csv)
    # ROC (requires new ResultsROCAUC signature: score_classes mandatory)
    ResultsROCAUC(
        class_task,
        y_true,
        y_score,
        Y_labels,
        s_name=str(outdir_p / f"ROC_{tag}.pdf"),
        score_classes=global_classes,
        eval_classes=global_classes,
        pos_label=pos_label,
        npz_path=str(outdir_p / f"ROC_meta_{tag}.npz"),
    )

    ResultsROCAUC(
        class_task,
        y_true,
        y_score,
        Y_labels,
        s_name=str(outdir_p / f"ROC_{tag}.png"),
        score_classes=global_classes,
        eval_classes=global_classes,
        pos_label=pos_label,
        npz_path=str(outdir_p / f"ROC_meta_{tag}.npz"),
    )

    # Confusion matrices
    plot_confusion_matrix_custom(
        y_true,
        y_pred,
        class_names,
        normalize=False,
        path=str(outdir_p / f"CM_{tag}.pdf"),
    )
    plot_confusion_matrix_custom(
        y_true,
        y_pred,
        class_names,
        normalize=True,
        path=str(outdir_p / f"CM_{tag}_norm.pdf"),
    )

    plot_confusion_matrix_custom(
        y_true,
        y_pred,
        class_names,
        normalize=False,
        path=str(outdir_p / f"CM_{tag}.png"),
    )
    plot_confusion_matrix_custom(
        y_true,
        y_pred,
        class_names,
        normalize=True,
        path=str(outdir_p / f"CM_{tag}_norm.png"),
    )

    metrics_path = outdir_p / f"METRICS_{tag}.csv"
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
