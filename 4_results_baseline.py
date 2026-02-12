#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Results-baseline.py

Baseline evaluation (NO masking, NO frequency selection) for default SVM / RF / LR.
- Tries GPU (RAPIDS cuML) first; if it fails for a model/fold, falls back to CPU (scikit-learn).
- Computes fold-wise metrics: AUC, F1, Accuracy
- Saves into results/:
    * BASELINE_folds_<task>.csv          (fold-wise rows for all models)
    * BASELINE_summary_<task>.json       (task + per-model mean/std for auc/f1/accuracy)
- Also creates plots like 3_final_results.py (ROC + CM + CM_norm) but tagged as baseline.

Usage:
  python3 Results-baseline.py --task healthy_vs_cancer --model all
  python3 Results-baseline.py --task MC --model svm --no-gpu
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# from sklearn.model_selection import GroupKFold
from sklearn.model_selection import StratifiedGroupKFold as GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

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


def compute_all_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray
) -> Tuple[float, float, float]:
    """
    Returns: (accuracy, f1, auc)
    y_score must be probability matrix shape (N, K) where K=n_classes.
    """
    n_classes = int(np.unique(y_true).size)

    acc = float(accuracy_score(y_true, y_pred))

    if n_classes <= 2:
        f1 = float(f1_score(y_true, y_pred))
        auc = float(roc_auc_score(y_true, y_score[:, 1]))
    else:
        f1 = float(f1_score(y_true, y_pred, average="macro"))
        auc = float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))

    return acc, f1, auc


def make_cpu_model(name: str):
    name = name.lower()
    if name == "svm":
        # default sklearn SVC (rbf) + proba for ROC
        return skSVC(probability=True, random_state=0)
    if name == "rf":
        return skRF(random_state=0)
    if name == "lr":
        return skLR(random_state=0, max_iter=1000)
    raise ValueError(f"Unknown model: {name}")


def make_gpu_model(name: str):
    """
    Default-ish cuML models.
    NOTE: cuML SVC with probability=True can throw KernelCache errors on some setups;
          we catch and CPU-fallback fold-wise.
    """
    name = name.lower()
    if name == "svm":
        return cuSVC(probability=True, max_iter=5000)
    if name == "rf":
        return cuRF()
    if name == "lr":
        return cuLR(max_iter=1000)
    raise ValueError(f"Unknown model: {name}")


def fit_predict_proba_with_fallback(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    prefer_gpu: bool,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Returns (probs_cpu (N,K), preds_cpu (N,), backend_used in {"cuml","cpu"})
    """
    # --- Try GPU first ---
    if prefer_gpu and GPU_AVAILABLE:
        try:
            Xtr_g = cp.asarray(X_train.astype(np.float32, copy=False))
            Xte_g = cp.asarray(X_test.astype(np.float32, copy=False))
            ytr_g = cp.asarray(y_train.astype(np.int32, copy=False))

            m = make_gpu_model(model_name)
            m.fit(Xtr_g, ytr_g)

            probs = m.predict_proba(Xte_g).get()  # -> numpy
            preds = np.argmax(probs, axis=1).astype(int)
            return probs, preds, "cuml"

        except Exception as e:
            print(
                f"[WARN] GPU failed for {model_name.upper()} -> CPU fallback. Reason: {e}"
            )

    # --- CPU fallback ---
    m = make_cpu_model(model_name)
    m.fit(X_train, y_train)
    probs = m.predict_proba(X_test)  # numpy
    preds = np.argmax(probs, axis=1).astype(int)
    return probs, preds, "cpu"


def backend_summary(backends: List[str]) -> str:
    u = sorted(set(backends))
    if len(u) == 1:
        return u[0]
    return "mixed"


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

    # load data (same as your other scripts)
    X, Y_labels, _, Case, _, _ = Results_data_loading(
        class_task, data_augmentation=False
    )

    # Baseline: flatten full tensor (no freq/mask selection)
    X_flat = X.reshape(X.shape[0], -1)

    cv = GroupKFold(n_splits=5)

    fold_rows: List[Dict[str, Any]] = []
    backends_used: List[str] = []

    # For plots: collect predictions over all folds
    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    y_score_all: List[np.ndarray] = []

    for fold_id, (tr, te) in enumerate(cv.split(X_flat, Y_labels, Case), start=1):
        probs, preds, backend_used = fit_predict_proba_with_fallback(
            model_name=model_name,
            X_train=X_flat[tr],
            y_train=Y_labels[tr],
            X_test=X_flat[te],
            prefer_gpu=use_gpu,
        )

        y_true = Y_labels[te].astype(int)
        y_pred = preds.astype(int)
        y_score = probs

        acc, f1, auc = compute_all_metrics(y_true, y_pred, y_score)

        fold_rows.append(
            dict(
                task=class_task,
                model=model_name,
                backend=backend_used,
                fold=fold_id,
                accuracy=acc,
                f1=f1,
                auc=auc,
                n_test=int(len(te)),
            )
        )
        backends_used.append(backend_used)

        print(
            f"Fold {fold_id}: AUC={auc:.4f}  F1={f1:.4f}  ACC={acc:.4f}  backend={backend_used}"
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
        "accuracy_mean": float(fold_df["accuracy"].mean()),
        "accuracy_std": float(fold_df["accuracy"].std(ddof=0)),
        "f1_mean": float(fold_df["f1"].mean()),
        "f1_std": float(fold_df["f1"].std(ddof=0)),
        "auc_mean": float(fold_df["auc"].mean()),
        "auc_std": float(fold_df["auc"].std(ddof=0)),
        "n_folds": int(len(fold_df)),
    }

    # ---- Save plots (like 3_final_results.py, but baseline-tagged) ----
    ensure_dir(plots_dir)
    tag = f"baseline_{class_task}_{model_name}"

    y_true_all_np = np.asarray(y_true_all, dtype=int)
    y_pred_all_np = np.asarray(y_pred_all, dtype=int)
    y_score_all_np = np.asarray(y_score_all, dtype=float)

    # ROC
    ResultsROCAUC(
        class_task,
        y_true_all_np,
        y_score_all_np,
        Y_labels,
        s_name=str(plots_dir / f"ROC_{tag}.pdf"),
    )

    # Confusion matrices
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

    # Plots dir (baseline explicit)
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

    # -----------------------------
    # Save CSV (fold-wise)
    # -----------------------------
    folds_all = pd.concat(all_fold_dfs, ignore_index=True)
    folds_csv = results_path / f"BASELINE_folds_{class_task}.csv"
    folds_all.to_csv(folds_csv, index=False)

    # -----------------------------
    # Save JSON (task + scores of f1/auc/accuracy)
    # -----------------------------
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
            }
            for s in all_summaries
        },
    }
    summary_json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # Print a nice terminal summary
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
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(f"  CSV (fold-wise): {folds_csv}")
    print(f"  JSON (summary):  {summary_json_path}")
    print(f"  Plots dir:       {plots_dir}")


if __name__ == "__main__":
    main()
