#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------
# GPU (cuML + CuPy) with CPU fallback (scikit-learn + NumPy)
# -----------------------
try:
    import cupy as cp
    from cuml.svm import SVC  # GPU

    try:
        _gpu_ok = cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        _gpu_ok = False

    if not _gpu_ok:
        raise RuntimeError("No CUDA device found")
except Exception:
    from sklearn.svm import SVC  # CPU fallback

    class _CPShim:
        @staticmethod
        def asarray(x):
            return np.asarray(x)

        @staticmethod
        def asnumpy(x):
            return np.asarray(x)

    cp = _CPShim()
    print("CuPy/cuML GPU not available -> falling back to CPU (scikit-learn).")

from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from toolbox import Masking
from toolbox.results import Results_data_loading


# -----------------------
# Mask selection dictionary (display name -> masking key)
# NOTE: We add "Z-threshold" as a special option handled in code.
# -----------------------
MASK_SEL = {
    "All": "all",
    "Z-threshold": "z_threshold",  # special-case handling
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


DEFAULT_Z_THRESHOLDS = {
    # fallbacks ! only for auc metric, and only if no tuned value is found in the provided z_thresholds_file
    "healthy_vs_cancer": 424,
    "cancer_vs_lr_hr": 314,
    "MC": 212,
}


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def shuffled_group_kfold(groups: np.ndarray, n_splits: int, seed: int):
    """Group-aware KFold with deterministic shuffling by groups."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)

    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)

    folds = np.array_split(uniq, n_splits)

    for fold_id, test_groups in enumerate(folds, start=1):
        test_mask = np.isin(groups, test_groups)
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(~test_mask)[0]
        yield fold_id, train_idx, test_idx


def compute_auc(y_true: np.ndarray, probs: np.ndarray, classes: np.ndarray) -> float:
    n_classes = len(classes)
    if n_classes == 2:
        return float(roc_auc_score(y_true, probs[:, 1]))
    y_bin = label_binarize(y_true, classes=classes)
    fpr, tpr, _ = roc_curve(y_bin.ravel(), probs.ravel())
    return float(auc(fpr, tpr))


def compute_primary_metric(
    metric: str, y_true: np.ndarray, y_pred: np.ndarray, probs, classes: np.ndarray
) -> float:
    if metric == "auc":
        return compute_auc(y_true, probs, classes)
    if metric == "acc":
        return float(accuracy_score(y_true, y_pred))
    if metric == "f1":
        if len(classes) == 2:
            mapping = {c: i for i, c in enumerate(classes)}
            y_true_idx = np.vectorize(mapping.get)(y_true)
            return float(f1_score(y_true_idx, y_pred, average="binary"))
        return float(f1_score(y_true, y_pred, average="macro"))
    raise ValueError(f"Unknown metric: {metric}")


def compute_extra_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray
) -> dict:
    """
    Extra metrics ONLY for logging/saving (not selectable as --metric).
    Binary: recall, sensitivity_sn, specificity_sp, precision, npv, balanced_acc, mcc
    Multi: recall_macro, precision_macro, f1_macro, specificity_sp_macro, balanced_acc, mcc
    """
    out: dict[str, float] = {}
    n_classes = len(classes)

    if n_classes == 2:
        mapping = {c: i for i, c in enumerate(classes)}
        yt = np.vectorize(mapping.get)(y_true).astype(int)
        yp = y_pred.astype(int)

        cm = confusion_matrix(yt, yp, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        sn = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        sp = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan

        out["recall"] = float(recall_score(yt, yp, average="binary", zero_division=0))
        out["sensitivity_sn"] = float(sn) if np.isfinite(sn) else np.nan
        out["specificity_sp"] = float(sp) if np.isfinite(sp) else np.nan
        out["precision"] = float(
            precision_score(yt, yp, average="binary", zero_division=0)
        )
        out["npv"] = float(npv) if np.isfinite(npv) else np.nan
        out["balanced_acc"] = float(balanced_accuracy_score(yt, yp))
        out["mcc"] = float(matthews_corrcoef(yt, yp))
        return out

    yt = y_true.astype(int)
    yp = y_pred.astype(int)

    out["recall_macro"] = float(recall_score(yt, yp, average="macro", zero_division=0))
    out["precision_macro"] = float(
        precision_score(yt, yp, average="macro", zero_division=0)
    )
    out["f1_macro"] = float(f1_score(yt, yp, average="macro", zero_division=0))
    out["balanced_acc"] = float(balanced_accuracy_score(yt, yp))
    out["mcc"] = float(matthews_corrcoef(yt, yp))

    cms = confusion_matrix(yt, yp, labels=np.unique(yt))
    specs = []
    for i in range(cms.shape[0]):
        tp = cms[i, i]
        fn = cms[i, :].sum() - tp
        fp = cms[:, i].sum() - tp
        tn = cms.sum() - (tp + fn + fp)
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        specs.append(spec)
    out["specificity_sp_macro"] = float(np.nanmean(specs)) if len(specs) else np.nan
    return out


def load_z_thresholds_map(z_thresholds_file: Path) -> dict:
    if not z_thresholds_file.exists():
        print("No z-thresholds file found, proceeding with defaults.")
        return {}
    try:
        return json.loads(z_thresholds_file.read_text(encoding="utf-8"))
    except Exception:
        return {}


def get_z_threshold_for_task_and_metric(
    class_task: str,
    metric: str,
    z_map: dict,
) -> int:
    """
    Prefer tuned value from z_map[task][metric], else fallback to DEFAULT_Z_THRESHOLDS[task].
    """
    try:
        v = z_map.get(class_task, {}).get(metric, None)
        if v is not None:
            return int(v)
    except Exception:
        pass
    return int(DEFAULT_Z_THRESHOLDS.get(class_task, 0))


def apply_mask_selection(
    *,
    mask_name: str,
    class_task: str,
    metric: str,
    X: np.ndarray,
    mask: Masking,
    avg_Z_fp: np.ndarray,
    z_map: dict,
) -> tuple[np.ndarray, int | None]:
    """
    Returns (flattened_X_tmp, used_z_threshold or None)
    """
    if mask_name == "All":
        return X.reshape(X.shape[0], -1), None

    if mask_name == "Z-threshold":
        z_thresh = get_z_threshold_for_task_and_metric(class_task, metric, z_map)
        if z_thresh <= 0:
            raise ValueError(
                f"No z-threshold configured for task='{class_task}', metric='{metric}'"
            )

        sel = avg_Z_fp[0, :] <= z_thresh
        X_tmp = X[:, :, sel]
        return X_tmp.reshape(X_tmp.shape[0], -1), int(z_thresh)

    # IIVV-based masking
    mask_key = MASK_SEL[mask_name]
    mask.IIVV_mask(mask_key, plot=False)
    X_tmp = X[:, :, mask.mask]
    return X_tmp.reshape(X_tmp.shape[0], -1), None


def run_single_fold(
    class_task: str,
    mask_name: str,
    rs: int,
    fold_id: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    X_tmp: np.ndarray,
    y_labels: np.ndarray,
    classes: np.ndarray,
    max_iter: int,
    metric: str,
    used_z_threshold: int | None,
):
    X_train = cp.asarray(X_tmp[train_idx])
    X_test = cp.asarray(X_tmp[test_idx])
    y_train = cp.asarray(y_labels[train_idx].astype(np.int32))
    y_test = cp.asarray(y_labels[test_idx].astype(np.int32))

    need_proba = metric == "auc"
    m = SVC(probability=need_proba, max_iter=max_iter)
    m.fit(X_train, y_train)

    def _get(a):
        return a.get() if hasattr(a, "get") else a

    y_pred_cpu = _get(m.predict(X_test)).astype(np.int64)
    y_test_cpu = cp.asnumpy(y_test).astype(np.int64)

    probs = None
    if need_proba:
        probs = _get(m.predict_proba(X_test))

    primary = compute_primary_metric(metric, y_test_cpu, y_pred_cpu, probs, classes)
    extras = compute_extra_metrics(y_test_cpu, y_pred_cpu, classes)

    row = {
        "Task": class_task,
        "Fold": int(fold_id),
        "IIVV": mask_name,
        "rs": int(rs),
        "metric": metric,
        "score": float(primary),
        "n_params": int(X_tmp.shape[1]),
        "need_proba": bool(need_proba),
        "z_threshold_used": (
            "None" if used_z_threshold is None else int(used_z_threshold)
        ),
    }
    # store extras (output only)
    for k, v in extras.items():
        row[k] = (
            None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)
        )

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        required=True,
        choices=["healthy_vs_cancer", "cancer_vs_lr_hr", "MC"],
        help="Classification task to run",
    )
    parser.add_argument(
        "--metric",
        default="auc",
        choices=["auc", "acc", "f1"],
        help="Metric used for scoring and ranking masks (extras are still saved in output)",
    )
    parser.add_argument("--n_trials", type=int, default=5)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--max_iter", type=int, default=5000)
    parser.add_argument("--results_dir", type=str, default="results")

    # NEW: load tuned z-thresholds
    parser.add_argument(
        "--z_thresholds_file",
        type=str,
        default="results/z_thresholds_best.json",
        help="JSON file created by tuning script 1_1iivv_zthresh.py.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    ensure_dir(results_dir)

    z_map = load_z_thresholds_map(Path(args.z_thresholds_file))

    # load masking patterns
    IIVV = np.loadtxt("source/IIVV_patterns.txt", dtype=int)
    mask = Masking(IIVV)
    mask.prints = True

    # load average impedances (provided)
    avg_Z_fp = np.load("source/Avg_Zs.npy")

    class_task = args.task
    metric = args.metric
    print(f"\n=== Running task: {class_task} | metric: {metric} ===")

    # load data
    X, y_labels, _, Case, _, _Y = Results_data_loading(
        class_task, data_augmentation=False
    )
    classes = np.unique(y_labels)

    stats = []
    for mask_name in MASK_SEL.keys():
        X_tmp, used_z = apply_mask_selection(
            mask_name=mask_name,
            class_task=class_task,
            metric=metric,
            X=X,
            mask=mask,
            avg_Z_fp=avg_Z_fp,
            z_map=z_map,
        )

        for rs in range(args.n_trials):
            for fold_id, train_idx, test_idx in shuffled_group_kfold(
                groups=Case, n_splits=args.n_folds, seed=rs
            ):
                stats.append(
                    run_single_fold(
                        class_task=class_task,
                        mask_name=mask_name,
                        rs=rs,
                        fold_id=fold_id,
                        train_idx=train_idx,
                        test_idx=test_idx,
                        X_tmp=X_tmp,
                        y_labels=y_labels,
                        classes=classes,
                        max_iter=args.max_iter,
                        metric=metric,
                        used_z_threshold=used_z,
                    )
                )

    df_stats = pd.DataFrame(stats)

    # Save raw fold stats (includes extra metrics)
    out_csv = results_dir / f"IIVV-analysis_{class_task}_{metric}_gpu.csv"
    df_stats.to_csv(out_csv, index=False)
    print(f"Saved fold stats: {out_csv}")

    # Ranking by mean score (good -> bad)
    df_rank = (
        df_stats.groupby(["IIVV"], as_index=False)
        .agg(
            mean_score=("score", "mean"),
            std_score=("score", "std"),
            n_runs=("score", "count"),
            n_params=("n_params", "first"),
            z_threshold_used=("z_threshold_used", "first"),
        )
        .sort_values(["mean_score", "std_score"], ascending=[False, True])
        .reset_index(drop=True)
    )

    rank_csv = results_dir / f"MaskSel_ranking_{class_task}_{metric}.csv"
    df_rank.to_csv(rank_csv, index=False)
    print(f"Saved ranking CSV: {rank_csv}")

    ordered_masks = df_rank["IIVV"].tolist()

    rank_txt = results_dir / f"MaskSel_ranking_{class_task}_{metric}.txt"
    rank_txt.write_text("\n".join(ordered_masks) + "\n", encoding="utf-8")
    print(f"Saved ranking TXT: {rank_txt}")

    rank_json = results_dir / f"MaskSel_ranking_{class_task}_{metric}.json"
    rank_json.write_text(
        json.dumps(
            {
                "task": class_task,
                "metric": metric,
                "order_good_to_bad": ordered_masks,
                "ranking_table": df_rank.to_dict(orient="records"),
                "z_thresholds_file": str(Path(args.z_thresholds_file)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved ranking JSON: {rank_json}")

    print(f"\nTop masks (good -> bad, metric: {metric}):")
    print(df_rank.head(len(MASK_SEL)).to_string(index=False))


if __name__ == "__main__":
    main()
