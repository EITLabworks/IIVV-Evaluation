#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPU Hyperparameter Tuning for SVM / RF / LR with Fsel & Msel loops (project-compatible).

UPDATED per your requirements:
- Loads TOP-4 Mask selections from: results/MaskSel_ranking_<task>_<metric>.json
- Loads TOP-4 Frequency selections (Top-N) from: results/FreqSel_ranking_<task>_<metric>.json
- Builds Fsel as "Top <N>" using your existing top_freqs_baseline (from freq_fold_results_<task>.npz),
  but the N values come from the JSON ranking.
- Prints a summary header: model, task, metric, top4 masks, top4 freqs (Top-N).
- Supports "Z-threshold" mask inside the Top-4:
    uses Avg_Zs.npy + task-specific thresholds to select patterns where avg_Z_fp[0,:] <= z_thresh.
- Avoids probability=True unless metric requires it (AUC); for other metrics uses predict().
"""

import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from sklearn.metrics import recall_score, confusion_matrix


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


def metric_to_rank_metric(metric: str) -> str:
    """
    Your tuning CLI uses: auc, f1, acc
    Your ranking files use: auc, f1, acc
    """
    metric = metric.lower()
    return metric


def z_threshold_for_task(class_task: str) -> Optional[int]:
    if class_task == "healthy_vs_cancer":
        return 424
    if class_task == "cancer_vs_lr_hr":
        return 314
    if class_task == "MC":
        return 212
    return None


def shuffle_groups_groupkfold(
    groups: np.ndarray, n_splits: int, seed: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """GroupKFold-like splitting with shuffling of groups."""
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
    import cupy as cp  # type: ignore

    return cp.asarray(x)


def to_cpu(x):
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


def sample_log_C(rng):
    return float(rng.choice([1000, 100, 10, 1, 0.1, 0.01, 0.001]))


def sample_log_gamma(rng):
    return float(rng.choice([10, 1, 0.1, 0.01, 0.001, 0.0001]))


def maybe_standardize_cpu(
    X_train: np.ndarray, X_val: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    return X_train_s, X_val_s


def _softmax_rows(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z)
    z = z - np.max(z, axis=1, keepdims=True)  # stability
    ez = np.exp(z)
    return ez / np.sum(ez, axis=1, keepdims=True)


def _pad_to_n_classes(scores: np.ndarray, model, n_classes: int) -> np.ndarray:
    """
    Map columns of `scores` (which correspond to model.classes_) into a full (N, n_classes) matrix.
    Missing classes get 0s.
    """
    scores = np.asarray(scores)
    if scores.ndim != 2:
        raise ValueError(f"Expected 2D scores for multiclass, got shape={scores.shape}")

    # If already correct, keep
    if scores.shape[1] == n_classes:
        return scores

    if not hasattr(model, "classes_"):
        raise ValueError(
            f"Scores have {scores.shape[1]} cols but n_classes={n_classes}, and model has no classes_."
        )

    full = np.zeros((scores.shape[0], n_classes), dtype=float)
    cls = np.asarray(model.classes_).astype(int)

    if len(cls) != scores.shape[1]:
        raise ValueError(
            f"len(model.classes_)={len(cls)} but scores has {scores.shape[1]} columns."
        )

    # Put each trained-class column into its global slot
    for j, c in enumerate(cls):
        if 0 <= c < n_classes:
            full[:, c] = scores[:, j]
    return full


def get_outputs_for_metric(model, X, metric: str, n_classes: int):
    metric = metric.lower()

    if metric in {"f1", "acc"}:
        y_pred = model.predict(X)
        return to_cpu(y_pred).ravel()

    # AUC
    if n_classes <= 2:
        if hasattr(model, "decision_function"):
            s = model.decision_function(X)
            return to_cpu(s).ravel()
        if hasattr(model, "predict_proba"):
            p = to_cpu(model.predict_proba(X))
            return p[:, 1].ravel()
        return to_cpu(model.predict(X)).ravel()

    # multiclass AUC -> need (N, n_classes)
    if hasattr(model, "predict_proba"):
        p = to_cpu(model.predict_proba(X))
        p = _pad_to_n_classes(p, model, n_classes)
        return p

    if hasattr(model, "decision_function"):
        s = to_cpu(model.decision_function(X))
        s = _pad_to_n_classes(s, model, n_classes)
        # convert scores to row-summing pseudo-probabilities (sklearn requires this for multiclass roc_auc)
        return _softmax_rows(s)

    raise ValueError("Multiclass AUC requires predict_proba or decision_function.")


def metric_fn(y_true: np.ndarray, y_out, metric: str, n_classes: int) -> float:
    metric = metric.lower()
    y_true = np.asarray(y_true).astype(int)
    uniq = np.unique(y_true)

    if metric == "acc":
        return float(accuracy_score(y_true, y_out))

    if metric == "f1":
        y_pred = np.asarray(y_out).astype(int)
        if n_classes <= 2:
            return float(f1_score(y_true, y_pred, zero_division=0))
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    if metric == "auc":
        # AUC is undefined if only one class is present in this fold
        if n_classes <= 2:
            if uniq.size < 2:
                return float(np.nan)
            return float(roc_auc_score(y_true, y_out))

        # Multiclass AUC requires all classes present in y_true for that fold
        if uniq.size < n_classes:
            return float(np.nan)

        return float(roc_auc_score(y_true, y_out, multi_class="ovr", average="macro"))

    raise ValueError(f"Unknown metric: {metric}")


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
        labels_cancer_vs_lr_hr,
        labels_MC,
    ) = load_data(path=data_path, Cutpats_threshold=60, norm=norm)

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

    if class_task == "cancer_vs_lr_hr":
        select_idx = find_all_indices(
            labels_cancer_vs_lr_hr, ["Cancer", "High risk", "Low risk"], split=False
        )
        labels = labels_cancer_vs_lr_hr[select_idx]
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

    raise ValueError(f"Unknown class_task: {class_task}")


# -----------------------------
# Load TOP-4 Fsel/Msel from JSON rankings
# -----------------------------
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


def load_top4_mask_names(
    results_path: str,
    class_task: str,
    rank_metric: str,
    N_top: int = 4,
) -> List[str]:
    p = os.path.join(results_path, f"MaskSel_ranking_{class_task}_{rank_metric}.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing mask ranking JSON: {p}")

    j = json.loads(Path(p).read_text(encoding="utf-8"))

    table = j.get("ranking_table")
    if not isinstance(table, list) or len(table) == 0:
        raise ValueError(f"Mask ranking JSON has no valid 'ranking_table': {p}")

    top = table[:N_top]

    names = []
    for row in top:
        if not isinstance(row, dict) or "IIVV" not in row:
            raise ValueError(f"Invalid row in ranking_table (missing 'IIVV'): {row}")
        names.append(str(row["IIVV"]))
    return names


def load_top4_freq_N(
    results_path: str,
    class_task: str,
    rank_metric: str,
    N_top: int = 4,
) -> Tuple[List[int], List[int]]:
    """
    Loads frequency ranking JSON and within it extracts:

    baseline_order  = order_good_to_bad list
    ranking_N_list  = list of N_freqs from ranking_table

    JSON schema expected:
        order_good_to_bad: [...]
        ranking_table: [
            { "N_freqs": 5, ... },
            ...
        ]
    """

    p = os.path.join(
        results_path,
        f"FreqSel_ranking_{class_task}_{rank_metric}.json",
    )
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing freq ranking JSON: {p}")

    j = json.loads(Path(p).read_text(encoding="utf-8"))

    # ---------- order_good_to_bad ----------
    order = j.get("order_good_to_bad", [])
    if not isinstance(order, list) or len(order) == 0:
        raise ValueError(f"No order_good_to_bad in {p}")

    baseline_order: List[int] = []
    for x in order:
        if isinstance(x, (int, float)):
            baseline_order.append(int(x))
        else:
            s = str(x).strip()
            if s.upper().startswith("N"):
                s = s[1:]
            baseline_order.append(int(float(s)))

    # ---------- ranking_table -> N_freqs ----------
    table = j.get("ranking_table", [])
    if not isinstance(table, list) or len(table) == 0:
        raise ValueError(f"No ranking_table in {p}")

    ranking_N_freqs: List[int] = []
    for row in table:
        if "N_freqs" not in row:
            continue
        ranking_N_freqs.append(int(row["N_freqs"]))
    if not ranking_N_freqs:
        raise ValueError(f"No N_freqs entries found in ranking_table of {p}")

    top4_freqs = [baseline_order[:n] for n in ranking_N_freqs[:N_top]]

    return [
        np.array(freqs) - 1 for freqs in top4_freqs
    ]  # correct for 1-based indexing in JSON -> 0-based indexing in code


def build_top4_fsel_msel(
    class_task: str,
    metric: str,
    results_path: str = "results",
) -> Tuple[Dict[str, np.ndarray], Dict[str, str], Dict[str, Any]]:
    """
    Returns:
      Freq_sel: {"Top 06": indices, ...} based on Top-N from JSON + baseline freq ordering (npz)
      Mask_sel: {"<mask_name>": "<mask_key or z_threshold>", ...} based on JSON
      meta: dict with pretty info (top4 lists + z-threshold)
    """
    rank_metric = metric_to_rank_metric(metric)

    # Load Top-Ns from freq ranking JSON
    top4_freqs = load_top4_freq_N(results_path, class_task, rank_metric)
    Freq_sel: Dict[str, np.ndarray] = {}

    for freqs in top4_freqs:
        key = f"Top {int(len(freqs))}"
        Freq_sel[key] = np.array(freqs)

    # Load Top-4 masks from mask ranking JSON
    top4_masks = load_top4_mask_names(results_path, class_task, rank_metric)

    # Convert mask names -> keys (and keep only known)
    Mask_sel: Dict[str, str] = {}
    for name in top4_masks:
        if name not in MASK_SEL_ALL:
            # keep it visible for debugging, but skip since we can't apply it
            log(f"Warning: mask '{name}' not in MASK_SEL_ALL, skipping.")
            continue
        Mask_sel[name] = MASK_SEL_ALL[name]

    zt = z_threshold_for_task(class_task)
    meta = {
        "rank_metric": rank_metric,
        "top4_freqs": top4_freqs,
        "top4_masks": top4_masks,
        "z_threshold": zt,
    }

    return Freq_sel, Mask_sel, meta


def print_top4_overview(
    model_name: str, class_task: str, metric: str, meta: Dict[str, Any]
) -> None:
    width = 92
    print("\n" + "=" * width)
    print("TUNING OVERVIEW (Top-4 from ranking JSON)")
    print(f"Model:   {model_name}")
    print(f"Task:    {class_task}")
    print(f"Metric:  {metric}  (ranking files use: {meta.get('rank_metric')})")
    print("-" * width)

    # Masks
    masks = meta.get("top4_masks", [])
    zt = meta.get("z_threshold", None)
    print("Top4 Masks (good → bad):")
    if not masks:
        print("  (none)")
    else:
        for i, m in enumerate(masks[:4], start=1):
            if m == "Z-threshold" and zt is not None:
                print(f"  {i}. Z-threshold (avg_Z <= {zt})")
            else:
                print(f"  {i}. {m}")

    # Freqs
    freqs = meta.get("top4_freqs", [])
    print("\nTop4 Freq selections (Top-N counts, good → bad):")
    if not freqs:
        print("  (none)")
    else:
        for i, freqs in enumerate(freqs[:4], start=1):
            print(f"  {i}. {freqs}")

    print("=" * width + "\n")


# -----------------------------
# Model param samplers
# -----------------------------
def sample_params(model_name: str, rng: np.random.Generator) -> Dict[str, Any]:
    model_name = model_name.lower()

    if model_name == "svm":
        kernel = rng.choice(["linear", "rbf", "poly", "sigmoid"])
        C = sample_log_C(rng)
        params = {"kernel": kernel, "C": C}

        if kernel in {"rbf", "poly", "sigmoid"}:
            gamma = sample_log_gamma(rng)
            params["gamma"] = gamma
        if kernel == "poly":
            params["degree"] = int(rng.integers(2, 6))
        return params

    if model_name == "rf":
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


def make_model(
    model_name: str, params: Dict[str, Any], backend: str, metric: str, task: str
):
    """
    IMPORTANT: probability=True only when needed.
    - For AUC:
        * CPU SVM/LR/RF can use decision_function / predict_proba.
        * cuML SVC: if we need predict_proba for multiclass AUC, enable probability.
      We keep it minimal: only set probability=True for cuML SVC when metric == "auc"
      AND predict_proba is needed (we still prefer decision_function when available).
    """
    model_name_l = model_name.lower()
    need_proba = metric.lower() == "auc"

    if backend == "cuml":
        if model_name_l == "svm":
            p = filter_params_for_model(cuSVC, params)
            if "probability" in accepted_kwargs(cuSVC):
                p.setdefault("probability", bool(need_proba))
            return cuSVC(**p)

        if model_name_l == "rf":
            p = filter_params_for_model(cuRF, params)
            return cuRF(**p)

        if model_name_l == "lr":
            p = filter_params_for_model(cuLR, params)
            if "penalty" in accepted_kwargs(cuLR):
                p.setdefault("penalty", "l2")
            return cuLR(**p)

    # CPU fallback
    if model_name_l == "svm":
        p = filter_params_for_model(skSVC, params)
        # decision_function is enough for AUC on CPU; keep probability False for speed.
        p.setdefault("probability", True if task != "healthy_vs_cancer" else False)
        return skSVC(**p)

    if model_name_l == "rf":
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
    trial: int
    metric: str
    score_mean: float
    score_std: float
    recall_mean: float
    specificity_mean: float
    params_json: str


def apply_mask_and_freq(
    *,
    X: np.ndarray,
    freq_idx: np.ndarray,
    mask: Any,
    msel_key: str,
    class_task: str,
    avg_Z_fp: Optional[np.ndarray],
) -> np.ndarray:
    """
    Returns flattened X_tmp after applying frequency selection and masking selection.
    Supports:
      - standard IIVV masks via mask.IIVV_mask(msel_key)
      - z-threshold special: avg_Z_fp[0,:] <= z_thresh
    """
    # freq selection first
    X_tmp = X[:, freq_idx, :]  # (N, freqs_sel, patterns)

    if msel_key == "z_threshold":
        if avg_Z_fp is None:
            raise RuntimeError("avg_Z_fp is required for z-threshold, but is None.")
        zt = z_threshold_for_task(class_task)
        if zt is None:
            raise ValueError(f"No z-threshold configured for task: {class_task}")

        above_threshold = avg_Z_fp[0, :] <= zt  # patterns mask
        X_tmp = X_tmp[:, :, above_threshold]
        return X_tmp.reshape(X_tmp.shape[0], -1)

    # Normal IIVV masking
    mask.IIVV_mask(msel_key, plot=False)
    X_tmp = X_tmp[:, :, mask.mask]
    return X_tmp.reshape(X_tmp.shape[0], -1)


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

    # Load top-4 selections from JSON rankings
    Freq_sel, Mask_sel, meta = build_top4_fsel_msel(
        class_task=class_task, metric=metric, results_path=results_path
    )

    # Print overview (per your request)
    print_top4_overview(
        model_name=model_name, class_task=class_task, metric=metric, meta=meta
    )

    ensure_dir(results_path)

    # For z-threshold support
    avg_Z_fp = None
    if "Z-threshold" in meta.get("top4_masks", []):
        avg_Z_fp = np.load("source/Avg_Zs.npy")

    rng_master = np.random.default_rng(12345)
    group_seeds = list(range(n_group_seeds))

    all_rows: List[TrialResult] = []
    best_global = -np.inf
    best_desc = ""

    # Precompute class count once (based on current y, including optional binarization)
    n_classes = int(np.unique(y).size)

    for fsel_name, freq_idx in Freq_sel.items():
        for msel_name, msel_key in Mask_sel.items():
            X_tmp = apply_mask_and_freq(
                X=X,
                freq_idx=freq_idx,
                mask=mask,
                msel_key=msel_key,
                class_task=class_task,
                avg_Z_fp=avg_Z_fp,
            )
            n_params = X_tmp.shape[1]

            log(
                f"=== task={class_task} | model={model_name} | {fsel_name} | {msel_name} | n_params={n_params} ==="
            )

            # Convert once to GPU if needed
            if backend == "cuml":
                X_mat = to_gpu(X_tmp.astype(np.float32, copy=False))
                y_vec = to_gpu(y.astype(np.int32, copy=False))
            else:
                X_mat = X_tmp.astype(np.float32, copy=False)
                y_vec = y.astype(int, copy=False)

            for trial in range(1, n_trials + 1):

                recall_scores = []
                spec_scores = []
                rng = np.random.default_rng(int(rng_master.integers(0, 1_000_000_000)))
                params = sample_params(model_name, rng)

                # Evaluate across group-shuffled splits
                scores: List[float] = []
                for seed in group_seeds:
                    splits = shuffle_groups_groupkfold(
                        np.asarray(groups), n_splits=n_splits, seed=seed
                    )

                    fold_scores: List[float] = []
                    for tr_idx, va_idx in splits:
                        X_tr = X_mat[tr_idx]
                        X_va = X_mat[va_idx]
                        y_tr = y_vec[tr_idx]
                        y_va = y_vec[va_idx]

                        # Standardize (SVM/LR)
                        if model_name.lower() in {"svm", "lr"}:
                            if backend == "cpu":
                                X_tr, X_va = maybe_standardize_cpu(X_tr, X_va)
                            else:
                                import cupy as cp  # type: ignore

                                mu = X_tr.mean(axis=0)
                                sig = X_tr.std(axis=0) + 1e-8
                                X_tr = (X_tr - mu) / sig
                                X_va = (X_va - mu) / sig

                        model = make_model(
                            model_name=model_name,
                            params=params,
                            backend=backend,
                            metric=metric,
                            task=class_task,
                        )
                        model.fit(X_tr, y_tr)

                        y_out = get_outputs_for_metric(
                            model, X_va, metric=metric, n_classes=n_classes
                        )
                        y_true = to_cpu(y_va).astype(int)

                        # --- NEW: derive predictions for recall/spec ---
                        if metric == "auc" and n_classes > 2:
                            y_pred_fold = np.argmax(y_out, axis=1)
                        elif metric == "auc":
                            y_pred_fold = (y_out >= 0.5).astype(int)
                        else:
                            y_pred_fold = y_out.astype(int)

                        if n_classes == 2:
                            # If a fold has no positives/negatives, recall can be undefined:
                            # make it explicit and silence UndefinedMetricWarning.
                            recall_fold = recall_score(
                                y_true, y_pred_fold, zero_division=0
                            )
                            # Force a 2x2 confusion matrix even if a class is missing in this fold.
                            cm = confusion_matrix(y_true, y_pred_fold, labels=[0, 1])
                            tn, fp, fn, tp = cm.ravel()
                            denom = tn + fp
                            spec_fold = (tn / denom) if denom > 0 else np.nan
                        else:
                            # Macro recall across a fixed label set; also handles missing classes cleanly.
                            recall_fold = recall_score(
                                y_true,
                                y_pred_fold,
                                average="macro",
                                labels=np.arange(n_classes),
                                zero_division=0,
                            )
                            spec_fold = np.nan

                        recall_scores.append(recall_fold)
                        spec_scores.append(spec_fold)
                        fold_scores.append(
                            metric_fn(y_true, y_out, metric=metric, n_classes=n_classes)
                        )

                    scores.append(float(np.nanmean(fold_scores)))

                score_mean = float(np.nanmean(scores))
                score_std = float(np.nanstd(scores))
                recall_mean = float(np.nanmean(recall_scores))
                spec_mean = float(np.nanmean(spec_scores))

                if score_mean > best_global:
                    best_global = score_mean
                    best_desc = (
                        f"{class_task} | {model_name} | {fsel_name} | {msel_name} | "
                        f"{params} | {metric}={best_global:.5f}"
                    )
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
                        trial=trial,
                        metric=metric,
                        score_mean=score_mean,
                        score_std=score_std,
                        recall_mean=recall_mean,
                        specificity_mean=spec_mean,
                        params_json=json.dumps(params, sort_keys=True),
                    )
                )

    df = pd.DataFrame([r.__dict__ for r in all_rows])
    out_csv = os.path.join(
        results_path, f"GPU_tuning_{model_name}_{class_task}_{metric}.csv"
    )
    df.to_csv(out_csv, index=False)
    log(f"Saved results: {out_csv}")

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
        results_path, f"GPU_tuning_{model_name}_{class_task}_{metric}_AGG.csv"
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
        description="GPU tuning for SVM/RF/LR over Top4 Fsel+Msel from ranking JSON."
    )
    p.add_argument("--model", choices=["svm", "rf", "lr", "all"], default="all")
    p.add_argument(
        "--task", required=True, help="healthy_vs_cancer, cancer_vs_lr_hr, MC"
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
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-group-seeds", type=int, default=10)
    p.add_argument("--n-trials", type=int, default=40)
    p.add_argument("--results-path", default="results")
    p.add_argument("--data-path", default="data")
    p.add_argument("--norm", default="freq")
    p.add_argument("--no-gpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    use_gpu = not args.no_gpu

    models = ["svm", "rf", "lr"] if args.model == "all" else [args.model]
    all_best_rows = []

    for m in models:
        agg = run_tuning(
            model_name=m,
            class_task=args.task,
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
            args.results_path,
            f"GPU_tuning_BEST_OVERALL_{args.task}_{args.metric}.csv",
        )
        final.to_csv(out_final, index=False)
        log("=== BEST OVERALL (across models) ===")
        log(final.head(10).to_string(index=False))
        log(f"Saved overall best: {out_final}")


if __name__ == "__main__":
    main()
