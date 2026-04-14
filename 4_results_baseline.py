#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
4_results_baseline.py

Baseline evaluation (NO masking, NO frequency selection) for default SVM / RF / LR.
- Tries GPU (RAPIDS cuML) first; if it fails for a model/fold, falls back to CPU (scikit-learn).
- Computes fold-wise metrics:
    * accuracy, f1, auc
    * recall (sensitivity), specificity, precision
    * (binary only) npv
  Notes:
    - For multiclass tasks: recall/precision are micro-averaged; specificity/npv are NaN.
- Saves into results/:
    * BASELINE_folds_<task>.csv          (fold-wise rows for all models)
    * BASELINE_summary_<task>.json       (task + per-model mean/std for all metrics)
- Also creates plots (ROC + CM + CM_norm) but tagged as baseline.

IMPORTANT (label safety):
- This version NEVER assumes that prob-column j corresponds to class label j.
- It always:
    1) takes the model's "classes_" (or a safe fallback),
    2) aligns probabilities to a global, fixed class order (global_classes),
    3) derives predictions from the aligned probabilities,
    4) computes AUC/ROC using the exact same alignment.
So labels/prob-columns cannot silently get swapped.

Usage:
  python3 4_results_baseline.py --task healthy_vs_cancer --model all
  python3 4_results_baseline.py --task MC --model svm --no-gpu
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold as GroupKFold
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
)

# CPU models
from sklearn.svm import SVC as skSVC
from sklearn.ensemble import RandomForestClassifier as skRF
from sklearn.linear_model import LogisticRegression as skLR

# Project plotting + loading
from toolbox.results import Results_data_loading, ResultsROCAUC
from toolbox.visualization import plot_confusion_matrix_custom


# -----------------------------
# Optional GPU imports (cuML)
# -----------------------------
GPU_AVAILABLE = False
try:
    import cupy as cp
    from cuml.svm import SVC as cuSVC
    from cuml.ensemble import RandomForestClassifier as cuRF
    from cuml.linear_model import LogisticRegression as cuLR

    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False


# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


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
    Deterministic choice of positive class for binary ROC/AUC.
    Change this if you want a different "positive" meaning.

    Default: largest label (same rule as many ROC helpers).
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
    Reorder probability columns so that they match global_classes exactly.
    Raises if any global class is missing in model_classes.
    """
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


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score_aligned: np.ndarray,
    global_classes: np.ndarray,
    pos_label: int | None = None,
) -> Dict[str, float]:
    """
    Returns a dict with:
      - accuracy, f1, auc
      - recall (sensitivity), specificity, precision
      - npv (binary only; else NaN)

    Requirements:
      - y_score_aligned columns MUST be ordered as global_classes.
    """
    global_classes = np.asarray(global_classes)
    n_classes = int(len(global_classes))

    acc = float(accuracy_score(y_true, y_pred))

    if n_classes == 2:
        if pos_label is None:
            pos_label = int(global_classes[-1])

        # Find column for pos_label in aligned score matrix
        pos_idx = int(np.where(global_classes == pos_label)[0][0])
        y_true_bin = (y_true == pos_label).astype(int)

        f1 = float(f1_score(y_true, y_pred, pos_label=pos_label))
        aucv = float(roc_auc_score(y_true_bin, y_score_aligned[:, pos_idx]))

        recall = float(recall_score(y_true, y_pred, pos_label=pos_label))  # sensitivity
        precision = float(
            precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        )

        # Confusion matrix in fixed label order: [neg, pos]
        neg_label = int([c for c in global_classes.tolist() if c != pos_label][0])
        cm = confusion_matrix(y_true, y_pred, labels=[neg_label, pos_label])
        tn, fp, fn, tp = cm.ravel()

        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
        npv = float(tn / (tn + fn)) if (tn + fn) > 0 else np.nan

        return dict(
            accuracy=acc,
            f1=f1,
            auc=aucv,
            recall=recall,
            specificity=specificity,
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


def make_cpu_model(name: str):
    name = name.lower()
    if name == "svm":
        return skSVC(probability=True, random_state=0)
    if name == "rf":
        return skRF(random_state=0)
    if name == "lr":
        return skLR(random_state=0, max_iter=1000)
    raise ValueError(f"Unknown model: {name}")


def make_gpu_model(name: str):
    name = name.lower()
    if name == "svm":
        return cuSVC(probability=True, max_iter=5000)
    if name == "rf":
        return cuRF()
    if name == "lr":
        return cuLR(max_iter=1000)
    raise ValueError(f"Unknown model: {name}")


def _safe_get_classes_after_fit(model: Any, y_train: np.ndarray) -> np.ndarray:
    """
    Prefer model.classes_ if present; else fall back to sorted unique y_train.
    """
    cls = getattr(model, "classes_", None)
    if cls is None:
        return np.array(sorted(np.unique(y_train))).astype(int)
    return np.asarray(cls).astype(int)


def fit_predict_proba_with_fallback(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    prefer_gpu: bool,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Returns (probs (N,K), model_classes (K,), backend_used in {"cuml","cpu"})
    """
    if prefer_gpu and GPU_AVAILABLE:
        try:
            Xtr_g = cp.asarray(X_train.astype(np.float32, copy=False))
            Xte_g = cp.asarray(X_test.astype(np.float32, copy=False))
            ytr_g = cp.asarray(y_train.astype(np.int32, copy=False))

            m = make_gpu_model(model_name)
            m.fit(Xtr_g, ytr_g)

            probs = m.predict_proba(Xte_g).get()
            model_classes = _safe_get_classes_after_fit(m, y_train)
            return probs, model_classes, "cuml"
        except Exception as e:
            print(
                f"[WARN] GPU failed for {model_name.upper()} -> CPU fallback. Reason: {e}"
            )

    m = make_cpu_model(model_name)
    m.fit(X_train, y_train)

    probs = m.predict_proba(X_test)
    model_classes = _safe_get_classes_after_fit(m, y_train)
    return probs, model_classes, "cpu"


def backend_summary(backends: List[str]) -> str:
    u = sorted(set(backends))
    return u[0] if len(u) == 1 else "mixed"


# -----------------------------
# Core evaluation
# -----------------------------
def run_baseline_model(
    *,
    class_task: str,
    model_name: str,
    results_path: Path,
    plots_dir: Path,
    use_gpu: bool,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    print(f"\n=== BASELINE {model_name.upper()} | task={class_task} ===")

    X, Y_labels, _, Case, _, _ = Results_data_loading(
        class_task, data_augmentation=False
    )

    # Fixed, global class order (THIS is the only order used everywhere)
    global_classes = np.array(sorted(np.unique(Y_labels))).astype(int)

    # Positive label (binary only) — deterministic; change in choose_pos_label_for_task if needed
    pos_label = None
    if len(global_classes) == 2:
        pos_label = choose_pos_label_for_task(class_task, global_classes)

    X_flat = X.reshape(X.shape[0], -1)
    cv = GroupKFold(n_splits=5)

    fold_rows: List[Dict[str, Any]] = []
    backends_used: List[str] = []

    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    y_score_all: List[np.ndarray] = []

    for fold_id, (tr, te) in enumerate(cv.split(X_flat, Y_labels, Case), start=1):
        probs_raw, model_classes, backend_used = fit_predict_proba_with_fallback(
            model_name=model_name,
            X_train=X_flat[tr],
            y_train=Y_labels[tr],
            X_test=X_flat[te],
            prefer_gpu=use_gpu,
        )

        # Align probability columns to global_classes (prevents label swapping)
        probs_aligned = align_proba_to_global_classes(
            probs_raw, model_classes, global_classes
        )

        # Derive predictions from aligned probs (so labels match global_classes)
        y_pred = global_classes[np.argmax(probs_aligned, axis=1)].astype(int)

        y_true = Y_labels[te].astype(int)
        y_score = probs_aligned

        m = compute_all_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_score_aligned=y_score,
            global_classes=global_classes,
            pos_label=pos_label,
        )

        fold_rows.append(
            dict(
                task=class_task,
                model=model_name,
                backend=backend_used,
                fold=fold_id,
                n_test=int(len(te)),
                **m,
            )
        )
        backends_used.append(backend_used)

        print(
            f"Fold {fold_id}: "
            f"AUC={m['auc']:.4f}  F1={m['f1']:.4f}  ACC={m['accuracy']:.4f}  "
            f"REC={m['recall']:.4f}  SPEC={m['specificity']:.4f}  "
            f"PREC={m['precision']:.4f}  backend={backend_used}"
        )

        y_true_all.extend(y_true.tolist())
        y_pred_all.extend(y_pred.tolist())
        y_score_all.extend(list(y_score))

    fold_df = pd.DataFrame(fold_rows)

    # Summary (mean/std over folds)
    summary = {
        "task": class_task,
        "model": model_name,
        "backend": backend_summary(backends_used),
        "n_folds": int(len(fold_df)),
        "accuracy_mean": float(fold_df["accuracy"].mean()),
        "accuracy_std": float(fold_df["accuracy"].std(ddof=0)),
        "f1_mean": float(fold_df["f1"].mean()),
        "f1_std": float(fold_df["f1"].std(ddof=0)),
        "auc_mean": float(fold_df["auc"].mean()),
        "auc_std": float(fold_df["auc"].std(ddof=0)),
        "recall_mean": float(fold_df["recall"].mean()),
        "recall_std": float(fold_df["recall"].std(ddof=0)),
        "specificity_mean": float(fold_df["specificity"].mean()),
        "specificity_std": float(fold_df["specificity"].std(ddof=0)),
        "precision_mean": float(fold_df["precision"].mean()),
        "precision_std": float(fold_df["precision"].std(ddof=0)),
        "npv_mean": float(fold_df["npv"].mean()),
        "npv_std": float(fold_df["npv"].std(ddof=0)),
    }

    # ---- Save plots (baseline-tagged) ----
    ensure_dir(plots_dir)
    tag = f"baseline_{class_task}_{model_name}"

    y_true_all_np = np.asarray(y_true_all, dtype=int)
    y_pred_all_np = np.asarray(y_pred_all, dtype=int)
    y_score_all_np = np.asarray(y_score_all, dtype=float)

    # NOTE: This assumes you replaced toolbox.results.ResultsROCAUC with the robust version
    # that REQUIRES score_classes and aligns properly.
    ResultsROCAUC(
        class_task,
        y_true_all_np,
        y_score_all_np,
        Y_labels,
        s_name=str(plots_dir / f"ROC_{tag}.pdf"),
        score_classes=global_classes,
        eval_classes=global_classes,
        pos_label=pos_label,
        npz_path=str(plots_dir / f"ROC_meta_{tag}.npz"),
    )

    ResultsROCAUC(
        class_task,
        y_true_all_np,
        y_score_all_np,
        Y_labels,
        s_name=str(plots_dir / f"ROC_{tag}.png"),
        score_classes=global_classes,
        eval_classes=global_classes,
        pos_label=pos_label,
        npz_path=str(plots_dir / f"ROC_meta_{tag}.npz"),
    )

    class_names = class_names_for_task(class_task)
    plot_confusion_matrix_custom(
        y_true_all_np,
        y_pred_all_np,
        class_names,
        normalize=False,
        path=str(plots_dir / f"CM_{tag}.pdf"),
    )
    plot_confusion_matrix_custom(
        y_true_all_np,
        y_pred_all_np,
        class_names,
        normalize=True,
        path=str(plots_dir / f"CM_{tag}_norm.pdf"),
    )

    plot_confusion_matrix_custom(
        y_true_all_np,
        y_pred_all_np,
        class_names,
        normalize=False,
        path=str(plots_dir / f"CM_{tag}.png"),
    )
    plot_confusion_matrix_custom(
        y_true_all_np,
        y_pred_all_np,
        class_names,
        normalize=True,
        path=str(plots_dir / f"CM_{tag}_norm.png"),
    )

    return summary, fold_df


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--task", required=True, choices=["healthy_vs_cancer", "cancer_vs_lr_hr", "MC"]
    )
    ap.add_argument(
        "--model",
        default="all",
        choices=["all", "svm", "rf", "lr"],
        help="Default: all",
    )
    ap.add_argument(
        "--results-path",
        default="results",
        help="Folder to write CSV/JSON + plots dir.",
    )
    ap.add_argument(
        "--no-gpu", action="store_true", help="Force CPU even if GPU is available."
    )
    args = ap.parse_args()

    class_task = args.task
    results_path = Path(args.results_path)
    use_gpu = not args.no_gpu

    ensure_dir(results_path)

    models = ["svm", "rf", "lr"] if args.model == "all" else [args.model]

    plots_dir = results_path / f"plots_{class_task}_baseline"
    ensure_dir(plots_dir)

    all_summaries: List[Dict[str, Any]] = []
    all_fold_dfs: List[pd.DataFrame] = []

    for m in models:
        summary, fold_df = run_baseline_model(
            class_task=class_task,
            model_name=m,
            results_path=results_path,
            plots_dir=plots_dir,
            use_gpu=use_gpu,
        )
        all_summaries.append(summary)
        all_fold_dfs.append(fold_df)

    # Save CSV (fold-wise)
    folds_all = pd.concat(all_fold_dfs, ignore_index=True)
    folds_csv = results_path / f"BASELINE_folds_{class_task}.csv"
    folds_all.to_csv(folds_csv, index=False)

    # Save JSON (summary)
    summary_json_path = results_path / f"BASELINE_summary_{class_task}.json"
    payload = {
        "task": class_task,
        "baseline": True,
        "gpu_available": bool(GPU_AVAILABLE),
        "models": {
            s["model"]: {
                "backend": s["backend"],
                "n_folds": s["n_folds"],
                "accuracy_mean": s["accuracy_mean"],
                "accuracy_std": s["accuracy_std"],
                "f1_mean": s["f1_mean"],
                "f1_std": s["f1_std"],
                "auc_mean": s["auc_mean"],
                "auc_std": s["auc_std"],
                "recall_mean": s["recall_mean"],
                "recall_std": s["recall_std"],
                "specificity_mean": s["specificity_mean"],
                "specificity_std": s["specificity_std"],
                "precision_mean": s["precision_mean"],
                "precision_std": s["precision_std"],
                "npv_mean": s["npv_mean"],
                "npv_std": s["npv_std"],
            }
            for s in all_summaries
        },
    }
    summary_json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # Terminal summary
    summary_df = pd.DataFrame(all_summaries).sort_values("auc_mean", ascending=False)
    print("\n=== BASELINE SUMMARY (mean over folds) ===")
    print(
        summary_df[
            [
                "task",
                "model",
                "backend",
                "auc_mean",
                "auc_std",
                "f1_mean",
                "f1_std",
                "accuracy_mean",
                "accuracy_std",
                "recall_mean",
                "recall_std",
                "specificity_mean",
                "specificity_std",
                "precision_mean",
                "precision_std",
                "npv_mean",
                "npv_std",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(f"  CSV (fold-wise): {folds_csv}")
    print(f"  JSON (summary):  {summary_json_path}")
    print(f"  Plots dir:       {plots_dir}")


if __name__ == "__main__":
    main()
