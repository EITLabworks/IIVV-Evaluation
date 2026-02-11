#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPU Hyperparameter Tuning for SVM / RF / LR with Fsel & Msel loops (project-compatible).
- Uses RAPIDS cuML on GPU if available, otherwise falls back to scikit-learn CPU.
- Shows live status in the shell.
- Lets you choose class_task and (optionally) a positive class label to tune (one-vs-rest binarization).
- Reports best combination: (Fsel, Msel, model, hyperparams) by chosen metric.
"""

import argparse
import inspect
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# -----------------------------
# Optional GPU imports (cuML)
# -----------------------------
GPU_AVAILABLE = False
GPU_BACKEND = "cpu"

try:
    import cupy as cp  # noqa: F401
    import cudf  # noqa: F401

    from cuml.svm import SVC as cuSVC
    from cuml.ensemble import RandomForestClassifier as cuRF
    from cuml.linear_model import LogisticRegression as cuLR

    GPU_AVAILABLE = True
    GPU_BACKEND = "cuml"
except Exception:
    GPU_AVAILABLE = False
    GPU_BACKEND = "cpu"

# -----------------------------
# CPU fallback imports
# -----------------------------
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC as skSVC
from sklearn.ensemble import RandomForestClassifier as skRF
from sklearn.linear_model import LogisticRegression as skLR

# -----------------------------
# Your project imports (expected)
# -----------------------------
# If your project structure differs, adjust these imports.
try:
    from toolbox import Masking, find_all_indices, hot_coding  # type: ignore
    from toolbox.loading import load_data  # type: ignore
except Exception as e:
    Masking = None
    load_data = None
    find_all_indices = None
    hot_coding = None
    PROJECT_IMPORT_ERROR = str(e)
else:
    PROJECT_IMPORT_ERROR = ""


# -----------------------------
# Utilities
# -----------------------------
def now_str() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now_str()}] {msg}", flush=True)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def shuffle_groups_groupkfold(
    groups: np.ndarray, n_splits: int, seed: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    GroupKFold-like splitting with shuffling of groups (since sklearn GroupKFold has no shuffle).
    Returns list of (train_idx, val_idx).
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)

    folds = np.array_split(uniq, n_splits)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    all_idx = np.arange(groups.shape[0])

    for f in folds:
        val_mask = np.isin(groups, f)
        val_idx = all_idx[val_mask]
        train_idx = all_idx[~val_mask]
        splits.append((train_idx, val_idx))
    return splits


def to_gpu(x: np.ndarray):
    # cupy conversion only if available
    import cupy as cp  # type: ignore

    return cp.asarray(x)


def to_cpu(x):
    # handles cupy arrays
    try:
        import cupy as cp  # type: ignore

        if isinstance(x, cp.ndarray):
            return x.get()
    except Exception:
        pass
    return x


def accepted_kwargs(model_cls) -> set:
    try:
        sig = inspect.signature(model_cls)
        return set(sig.parameters.keys())
    except Exception:
        return set()


def filter_params_for_model(model_cls, params: Dict[str, Any]) -> Dict[str, Any]:
    keys = accepted_kwargs(model_cls)
    if not keys:
        return params
    return {k: v for k, v in params.items() if k in keys}


# def sample_log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
#    return float(np.exp(rng.uniform(np.log(low), np.log(high))))
def sample_log_C(rng):
    return float(rng.choice([1000, 100, 10, 1, 0.1, 0.01, 0.001]))


def sample_log_gamma(rng):
    return float(rng.choice([10, 1, 0.1, 0.01, 0.001, 0.0001]))


def maybe_standardize_cpu(
    X_train: np.ndarray, X_val: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    # Standardize for SVM/LR; RF typically doesn’t need it, but harmless.
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    return X_train_s, X_val_s


def get_outputs_for_metric(model, X, metric: str, n_classes: int):
    """
    For f1/acc -> return hard predictions (shape: [N])
    For auc:
      - binary -> return scores/probabilities (shape: [N])
      - multiclass -> return probability matrix (shape: [N, K]) if possible
    """
    metric = metric.lower()

    if metric in {"f1", "acc"}:
        y_pred = model.predict(X)
        return to_cpu(y_pred).ravel()

    # AUC branch
    if n_classes <= 2:
        if hasattr(model, "decision_function"):
            s = model.decision_function(X)
            return to_cpu(s).ravel()
        if hasattr(model, "predict_proba"):
            p = to_cpu(model.predict_proba(X))
            return p[:, 1].ravel()
        y_pred = to_cpu(model.predict(X)).ravel()
        return y_pred

    # multiclass AUC needs (N,K) probabilities/scores
    if hasattr(model, "predict_proba"):
        return to_cpu(model.predict_proba(X))  # (N,K)
    if hasattr(model, "decision_function"):
        return to_cpu(model.decision_function(X))  # (N,K)
    raise ValueError(
        "Multiclass AUC requires predict_proba or decision_function matrix."
    )


def metric_fn(y_true: np.ndarray, y_out, metric: str, n_classes: int) -> float:
    metric = metric.lower()

    if metric == "acc":
        return float(accuracy_score(y_true, y_out))

    if metric == "f1":
        if n_classes <= 2:
            # binary f1: ensure labels are 0/1 preds
            y_pred = y_out.astype(int)
            return float(f1_score(y_true, y_pred))
        # multiclass f1:
        y_pred = y_out.astype(int)
        return float(f1_score(y_true, y_pred, average="macro"))

    if metric == "auc":
        if n_classes <= 2:
            return float(roc_auc_score(y_true, y_out))
        # multiclass auc:
        # y_true must be int labels, y_out must be (N,K)
        return float(roc_auc_score(y_true, y_out, multi_class="ovr", average="macro"))

    raise ValueError(f"Unknown metric: {metric}")


def get_scores(model, X, backend: str) -> np.ndarray:
    """
    Return continuous scores for AUC.
    Prefer decision_function, fallback to predict_proba[:,1], else predict.
    """
    # cuML and sklearn both usually have decision_function for SVM
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return to_cpu(s).ravel()
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        p = to_cpu(p)
        if p.ndim == 2 and p.shape[1] >= 2:
            return p[:, 1].ravel()
    # fallback: hard predictions
    yhat = model.predict(X)
    return to_cpu(yhat).ravel()


# -----------------------------
# Data loading (project-compatible)
# -----------------------------
def load_project_data(class_task: str, data_path: str = "data", norm: str = "freq"):
    if load_data is None or Masking is None:
        raise RuntimeError(
            "Project imports failed. Make sure toolbox + toolbox.loading are importable.\n"
            f"Import error: {PROJECT_IMPORT_ERROR}"
        )

    IIVV = np.loadtxt("source/IIVV_patterns.txt", dtype=int)
    mask = Masking(IIVV)
    mask.prints = False

    (
        py_idx,
        patient_no,
        patient_sample,
        Zdata,
        Avg_Cutpats,
        LesionAssessment,
        labels_healthy_vs_cancer,
        labels_hr_vs_lr,
        labels_MC,
    ) = load_data(path=data_path, Cutpats_threshold=60, norm=norm)

    # Pick labels by class_task like in your notebook.
    if class_task == "healthy_vs_cancer":
        select_idx = find_all_indices(
            labels_healthy_vs_cancer, ["Healthy", "Cancer"], split=False
        )
        labels = labels_healthy_vs_cancer[select_idx]
        X = Zdata[select_idx]
        Y, inverse_mapping = hot_coding(labels)
        Y_labels = np.argmax(Y, axis=1)
        groups = patient_no[select_idx]
        label_names = (
            list(inverse_mapping.values())
            if hasattr(inverse_mapping, "values")
            else None
        )
        return X, Y_labels, labels, groups, mask, label_names

    if class_task == "hr_vs_lr":
        select_idx = find_all_indices(
            labels_hr_vs_lr, ["High risk", "Low risk"], split=False
        )
        labels = labels_hr_vs_lr[select_idx]
        X = Zdata[select_idx]
        Y, inverse_mapping = hot_coding(labels)
        Y_labels = np.argmax(Y, axis=1)
        groups = patient_no[select_idx]
        label_names = (
            list(inverse_mapping.values())
            if hasattr(inverse_mapping, "values")
            else None
        )
        return X, Y_labels, labels, groups, mask, label_names

    if class_task == "MC":
        # Multiclass available; you can still binarize via --positive-class
        select_idx = find_all_indices(
            labels_MC,
            ["Healthy", "Cancer", "High-grade dysplasia", "Non-malignant"],
            split=False,
        )
        labels = labels_MC[select_idx]
        X = Zdata[select_idx]
        Y, inverse_mapping = hot_coding(labels)
        Y_labels = np.argmax(Y, axis=1)
        groups = patient_no[select_idx]
        label_names = (
            list(inverse_mapping.values())
            if hasattr(inverse_mapping, "values")
            else None
        )
        return X, Y_labels, labels, groups, mask, label_names

    # If you have more tasks, extend here:
    raise ValueError(f"Unknown class_task: {class_task}")


def build_fsel_msel(
    class_task: str, results_path: str = "results"
) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    # Frequency selection
    freq_file = os.path.join(results_path, f"freq_fold_results_{class_task}.npz")
    tmp = np.load(freq_file, allow_pickle=True)
    top_freqs_baseline = tmp["top_freqs_baseline"]

    if class_task == "healthy_vs_cancer":
        Freq_sel = {
            "Top 6": top_freqs_baseline[:6],
            "Top 14": top_freqs_baseline[:14],
            "Top 5": top_freqs_baseline[:5],
            "Top 7": top_freqs_baseline[:7],
        }
    elif class_task == "hr_vs_lr":
        Freq_sel = {
            "Top 3": top_freqs_baseline[:3],
            "Top 4": top_freqs_baseline[:4],
            "Top 2": top_freqs_baseline[:2],
            "Top 5": top_freqs_baseline[:5],
        }
    elif class_task == "cancer_vs_lr_hr":
        Freq_sel = {
            "Top 4": top_freqs_baseline[:4],
            "Top 2": top_freqs_baseline[:2],
            "Top 29": top_freqs_baseline[:29],
            "Top 17": top_freqs_baseline[:17],
        }
    elif class_task == "hr_vs_lr":
        Freq_sel = {
            "Top 30": top_freqs_baseline[:30],
            "Top 3": top_freqs_baseline[:3],
            "Top 27": top_freqs_baseline[:27],
            "Top 28": top_freqs_baseline[:28],
        }
    elif class_task == "MC":
        Freq_sel = {
            "Top 13": top_freqs_baseline[:13],
            "Top 19": top_freqs_baseline[:19],
            "Top 20": top_freqs_baseline[:20],
            "Top 18": top_freqs_baseline[:18],
        }
    else:
        # generic fallback
        Freq_sel = {
            "Top 10": top_freqs_baseline[:10],
            "Top 20": top_freqs_baseline[:20],
        }

    # Masking selection
    mask_file = os.path.join(results_path, f"Top_IIVV_masks_{class_task}.npz")
    tmp2 = np.load(mask_file, allow_pickle=True)
    Mask_ranks = tmp2["Mask_ranks"][:4]

    Mask_sel_all = {
        # "All": "all",
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
    Mask_sel = {k: v for k, v in Mask_sel_all.items() if k in Mask_ranks}
    return Freq_sel, Mask_sel


# -----------------------------
# Model param samplers
# -----------------------------
def sample_params(model_name: str, rng: np.random.Generator) -> Dict[str, Any]:
    model_name = model_name.lower()

    if model_name == "svm":
        kernel = rng.choice(["linear", "rbf", "poly", "sigmoid"])
        C = sample_log_C(rng)
        params = {"kernel": kernel, "C": C}

        # Optional knobs for non-linear kernels
        if kernel in {"rbf", "poly", "sigmoid"}:
            gamma = sample_log_gamma(rng)
            params["gamma"] = gamma
        if kernel == "poly":
            params["degree"] = int(rng.integers(2, 6))
        return params

    if model_name == "rf":
        # cuML RF supports fewer knobs than sklearn; we sample a safe subset.
        n_estimators = int(rng.choice([100, 200, 300, 500, 800]))
        max_depth = int(rng.choice([8, 12, 16, 20, 24, 32]))
        max_features = float(rng.choice([0.3, 0.5, 0.7, 1.0]))
        return {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "max_features": max_features,
        }

    if model_name == "lr":
        C = sample_log_C(rng)
        max_iter = int(rng.choice([200, 500, 1000, 2000]))
        return {"C": C, "max_iter": max_iter}

    raise ValueError(f"Unknown model_name: {model_name}")


def make_model(model_name: str, params: Dict[str, Any], backend: str):
    model_name_l = model_name.lower()

    if backend == "cuml":
        if model_name_l == "svm":
            p = filter_params_for_model(cuSVC, params)
            # cuML SVC: enable probability if available (not always needed for AUC)
            if "probability" in accepted_kwargs(cuSVC):
                p.setdefault("probability", False)
            return cuSVC(**p)

        if model_name_l == "rf":
            p = filter_params_for_model(cuRF, params)
            return cuRF(**p)

        if model_name_l == "lr":
            p = filter_params_for_model(cuLR, params)
            # safer defaults
            if "penalty" in accepted_kwargs(cuLR):
                p.setdefault("penalty", "l2")
            return cuLR(**p)

    # CPU fallback
    if model_name_l == "svm":
        p = filter_params_for_model(skSVC, params)
        # For AUC: decision_function exists; probability not required (keeps it faster).
        p.setdefault("probability", False)
        return skSVC(**p)

    if model_name_l == "rf":
        # richer knobs on CPU; still accept our sampled subset
        p = dict(params)
        p.setdefault("n_jobs", -1)
        p.setdefault("random_state", 0)
        return skRF(**p)

    if model_name_l == "lr":
        p = dict(params)
        p.setdefault("solver", "lbfgs")
        p.setdefault("penalty", "l2")
        p.setdefault("n_jobs", -1)
        return skLR(**p)

    raise ValueError(f"Unknown model_name: {model_name}")


# -----------------------------
# Tuning loop
# -----------------------------
@dataclass
class TrialResult:
    model: str
    backend: str
    class_task: str
    positive_class: Optional[str]
    fsel: str
    msel: str
    n_params: int
    seed: int
    trial: int
    metric: str
    score_mean: float
    score_std: float
    params_json: str


def run_tuning(
    model_name: str,
    class_task: str,
    positive_class: Optional[str],
    metric: str,
    n_splits: int,
    n_group_seeds: int,
    n_trials: int,
    results_path: str,
    data_path: str,
    norm: str,
    use_gpu: bool,
) -> pd.DataFrame:
    backend = "cuml" if (use_gpu and GPU_AVAILABLE) else "cpu"
    log(f"Backend: {backend} (GPU_AVAILABLE={GPU_AVAILABLE})")

    X, Y_labels, labels_raw, groups, mask, label_names = load_project_data(
        class_task=class_task, data_path=data_path, norm=norm
    )

    # Optional binarization by positive class label
    y = np.asarray(Y_labels).astype(int)

    if positive_class is not None:
        labels_raw = np.asarray(labels_raw).astype(str)
        y = (labels_raw == positive_class).astype(int)
        if len(np.unique(y)) < 2:
            raise ValueError(
                f"--positive-class '{positive_class}' produced a single class only. "
                f"Available labels example: {np.unique(labels_raw)[:10]}"
            )

    Freq_sel, Mask_sel = build_fsel_msel(
        class_task=class_task, results_path=results_path
    )
    ensure_dir(results_path)

    rng_master = np.random.default_rng(12345)

    all_rows: List[TrialResult] = []
    best_global = -np.inf
    best_desc = ""

    # Pre-calc group splits seeds
    group_seeds = list(range(n_group_seeds))

    for fsel_name, freq_idx in Freq_sel.items():
        for msel_name, msel_key in Mask_sel.items():
            # Apply mask selection (project code)
            mask.IIVV_mask(msel_key, plot=False)

            # Select freq + mask + flatten
            X_tmp = X[:, freq_idx, :]
            X_tmp = X_tmp[:, :, mask.mask]
            X_tmp = X_tmp.reshape(X_tmp.shape[0], -1)
            n_params = X_tmp.shape[1]

            log(
                f"=== {class_task=} | {model_name=} | {fsel_name=} | {msel_name=} | n_params={n_params} ==="
            )

            # Convert once to GPU if needed (fold slicing still cheap)
            if backend == "cuml":
                X_mat = to_gpu(X_tmp.astype(np.float32, copy=False))
                y_vec = to_gpu(y.astype(np.int32, copy=False))
            else:
                X_mat = X_tmp.astype(np.float32, copy=False)
                y_vec = y.astype(int, copy=False)

            for trial in range(1, n_trials + 1):
                # Sample hyperparams
                rng = np.random.default_rng(int(rng_master.integers(0, 1_000_000_000)))
                params = sample_params(model_name, rng)

                # Evaluate across multiple group-shuffled splits (stabilizes your 10 random_states idea)
                scores: List[float] = []
                for seed in group_seeds:
                    splits = shuffle_groups_groupkfold(
                        np.asarray(groups), n_splits=n_splits, seed=seed
                    )

                    fold_scores: List[float] = []
                    for tr_idx, va_idx in splits:
                        if backend == "cuml":
                            X_tr = X_mat[tr_idx]
                            X_va = X_mat[va_idx]
                            y_tr = y_vec[tr_idx]
                            y_va = y_vec[va_idx]
                        else:
                            X_tr = X_mat[tr_idx]
                            X_va = X_mat[va_idx]
                            y_tr = y_vec[tr_idx]
                            y_va = y_vec[va_idx]

                        # Standardize (helps SVM/LR). For RF it’s okay.
                        if backend == "cpu" and model_name.lower() in {"svm", "lr"}:
                            X_tr, X_va = maybe_standardize_cpu(X_tr, X_va)
                        elif backend == "cuml" and model_name.lower() in {"svm", "lr"}:
                            # quick GPU standardization
                            import cupy as cp  # type: ignore

                            mu = X_tr.mean(axis=0)
                            sig = X_tr.std(axis=0) + 1e-8
                            X_tr = (X_tr - mu) / sig
                            X_va = (X_va - mu) / sig

                        model = make_model(
                            model_name=model_name, params=params, backend=backend
                        )
                        model.fit(X_tr, y_tr)

                        # y_score = get_scores(model, X_va, backend=backend)
                        # y_true = to_cpu(y_va).astype(int)

                        # fold_scores.append(metric_fn(y_true, y_score, metric))
                        n_classes = int(np.unique(to_cpu(y_vec)).size)

                        y_out = get_outputs_for_metric(
                            model, X_va, metric=metric, n_classes=n_classes
                        )
                        y_true = to_cpu(y_va).astype(int)

                        fold_scores.append(
                            metric_fn(y_true, y_out, metric=metric, n_classes=n_classes)
                        )

                    scores.append(float(np.mean(fold_scores)))

                score_mean = float(np.mean(scores))
                score_std = float(np.std(scores))

                if score_mean > best_global:
                    best_global = score_mean
                    best_desc = f"{class_task} | {model_name} | {fsel_name} | {msel_name} | {params} | {metric}={best_global:.5f}"
                    log(f"NEW BEST ✅  {best_desc}")

                log(
                    f"Trial {trial:03d}/{n_trials} | {fsel_name} / {msel_name} | "
                    f"{metric}={score_mean:.5f} ±{score_std:.5f} | params={params}"
                )

                all_rows.append(
                    TrialResult(
                        model=model_name,
                        backend=backend,
                        class_task=class_task,
                        positive_class=positive_class,
                        fsel=fsel_name,
                        msel=msel_name,
                        n_params=n_params,
                        seed=-1,
                        trial=trial,
                        metric=metric,
                        score_mean=score_mean,
                        score_std=score_std,
                        params_json=json.dumps(params, sort_keys=True),
                    )
                )

    df = pd.DataFrame([r.__dict__ for r in all_rows])
    out_csv = os.path.join(results_path, f"GPU_tuning_{model_name}_{class_task}.csv")
    df.to_csv(out_csv, index=False)
    log(f"Saved results: {out_csv}")

    # Aggregate (mean over identical hyperparam json) like you did
    agg = (
        df.groupby(["model", "fsel", "msel", "params_json"], as_index=False)
        .agg(
            score_mean=("score_mean", "mean"),
            score_std=("score_mean", "std"),
            n_params=("n_params", "max"),
        )
        .sort_values("score_mean", ascending=False)
    )

    out_agg = os.path.join(
        results_path, f"GPU_tuning_{model_name}_{class_task}_AGG.csv"
    )
    agg.to_csv(out_agg, index=False)
    log(f"Saved aggregated results: {out_agg}")

    best = agg.iloc[0].to_dict()
    log("=== BEST CONFIG (aggregated) ===")
    log(json.dumps(best, indent=2))

    return agg


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="GPU tuning for SVM/RF/LR over Fsel+Msel (project-compatible)."
    )
    p.add_argument(
        "--model",
        choices=["svm", "rf", "lr", "all"],
        default="svm",
        help="Which model to tune.",
    )
    p.add_argument(
        "--class-task", required=True, help="e.g. healthy_vs_cancer, hr_vs_lr, MC"
    )
    p.add_argument(
        "--positive-class",
        default=None,
        help="If set: binarize labels as (label == positive_class).",
    )
    p.add_argument(
        "--metric",
        choices=["auc", "f1", "acc"],
        default="auc",
        help="Optimization metric.",
    )
    p.add_argument("--n-splits", type=int, default=5, help="Group K-fold splits.")
    p.add_argument(
        "--n-group-seeds",
        type=int,
        default=10,
        help="How many group shuffles to average over.",
    )
    p.add_argument(
        "--n-trials", type=int, default=40, help="Random search trials per (Fsel,Msel)."
    )
    p.add_argument(
        "--results-path", default="results", help="Where to read/write .npz/.csv files."
    )
    p.add_argument(
        "--data-path",
        default="data",
        help="Data folder used by toolbox.loading.load_data",
    )
    p.add_argument("--norm", default="freq", help="Passed to load_data(norm=...)")
    p.add_argument(
        "--no-gpu", action="store_true", help="Force CPU even if GPU is available."
    )
    return p.parse_args()


def main():
    args = parse_args()

    use_gpu = not args.no_gpu

    if args.model == "all":
        models = ["svm", "rf", "lr"]
    else:
        models = [args.model]

    all_best_rows = []

    for m in models:
        agg = run_tuning(
            model_name=m,
            class_task=args.class_task,
            positive_class=args.positive_class,
            metric=args.metric,
            n_splits=args.n_splits,
            n_group_seeds=args.n_group_seeds,
            n_trials=args.n_trials,
            results_path=args.results_path,
            data_path=args.data_path,
            norm=args.norm,
            use_gpu=use_gpu,
        )
        best = agg.iloc[0].copy()
        best["model"] = m
        all_best_rows.append(best)

    if len(all_best_rows) > 1:
        final = pd.DataFrame(all_best_rows).sort_values("score_mean", ascending=False)
        out_final = os.path.join(
            args.results_path, f"GPU_tuning_BEST_OVERALL_{args.class_task}.csv"
        )
        final.to_csv(out_final, index=False)
        log("=== BEST OVERALL (across models) ===")
        log(final.head(10).to_string(index=False))
        log(f"Saved overall best: {out_final}")


if __name__ == "__main__":
    main()
