#!/usr/bin/env python3
"""
1_1iivv_zthres_tune.py

Tuning von z_threshold pro (Task, metric) via fold-level paired t-test vs Baseline (max z_threshold).
Speichert:
- results/z_thresholds_best.json               (Mapping task->metric->best_z_threshold)
- results/zthres_tuning_paired_ttest_*.csv     (t-test Tabelle)
- results/zthres_tuning_best_thresholds_*.csv  (best thresholds)
- results/zthres_tuning_df_stats_*.pkl/.csv    (df_stats im "snippet-format")
- results/zthres_tuning_fold_details_*.csv     (fold-level Details inkl. Recall/SN/SP/etc.)

Fixes:
- Filtert Kandidaten je Task so, dass mind. 1 Pattern selektiert wird
- Guard: wenn 0 Features -> skip
- cuML SVC: gamma='auto' um Division-by-zero bei gamma='scale' (Varianz=0) zu vermeiden
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

# -----------------------
# GPU (cuML + CuPy) with CPU fallback (scikit-learn + NumPy)
# -----------------------
USING_GPU = False
try:
    import cupy as cp
    from cuml.svm import SVC  # GPU

    try:
        USING_GPU = cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        USING_GPU = False

    if not USING_GPU:
        raise RuntimeError("No CUDA device found")
except Exception:
    USING_GPU = False
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

from toolbox.results import Results_data_loading


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
    *,
    metric: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray | None,
    classes: np.ndarray,
) -> float:
    if metric == "auc":
        if probs is None:
            raise ValueError("probs required for AUC")
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
    *, y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray
) -> dict:
    """
    Extra metrics (nur Output, NICHT als --metric):
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


def build_candidates(
    avg_Z_fp: np.ndarray, z_mode: str, n_quantiles: int, z_min, z_max
) -> np.ndarray:
    z_vals = np.asarray(avg_Z_fp[0, :]).astype(float)
    finite = np.isfinite(z_vals)
    if not np.any(finite):
        raise RuntimeError(
            "Avg_Zs.npy seems to contain no finite values in avg_Z_fp[0,:]."
        )

    if z_mode == "unique":
        candidates = np.unique(z_vals[finite]).astype(int)
    else:
        qs = np.linspace(0.0, 1.0, n_quantiles)
        candidates = np.unique(np.quantile(z_vals[finite], qs)).astype(int)

    if z_min is not None:
        candidates = candidates[candidates >= z_min]
    if z_max is not None:
        candidates = candidates[candidates <= z_max]

    candidates = np.unique(candidates).astype(int)
    if len(candidates) == 0:
        raise RuntimeError(
            "No z_threshold candidates left after filtering (z_min/z_max too strict?)."
        )
    return candidates


def filter_candidates_for_task(
    candidates: np.ndarray, avg_Z_fp: np.ndarray, X: np.ndarray
) -> np.ndarray:
    """
    Filter candidates to ensure they select at least one pattern.
    Also ensures the selection mask length matches X's pattern dimension.
    """
    z_vals = np.asarray(avg_Z_fp[0, :]).astype(float)
    finite = np.isfinite(z_vals)

    # Sanity: pattern count must match
    n_patterns_X = X.shape[2]
    n_patterns_Z = z_vals.shape[0]
    if n_patterns_X != n_patterns_Z:
        raise RuntimeError(
            f"Pattern dimension mismatch: X.shape[2]={n_patterns_X} but Avg_Zs has {n_patterns_Z} entries."
        )

    def n_selected(th: int) -> int:
        sel = finite & (z_vals <= th)
        return int(np.sum(sel))

    filtered = [int(th) for th in candidates if n_selected(int(th)) > 0]
    filtered = np.array(filtered, dtype=int)

    if len(filtered) == 0:
        mn = float(np.nanmin(z_vals))
        mx = float(np.nanmax(z_vals))
        raise RuntimeError(
            "All z_threshold candidates would select 0 patterns.\n"
            f"Avg_Zs range: min={mn:.6f}, max={mx:.6f}\n"
            "Try increasing z_min / using z_mode=unique / reducing quantile filtering."
        )
    return filtered


def run_fold_scores_for_threshold(
    *,
    class_task: str,
    metric: str,
    z_threshold: int,
    X: np.ndarray,
    y_labels: np.ndarray,
    Case: np.ndarray,
    classes: np.ndarray,
    avg_Z_fp: np.ndarray,
    n_folds: int,
    rs: int,
    max_iter: int,
) -> tuple[list[float], list[dict]]:
    """
    Returns:
      - fold_scores: list of primary metric per fold
      - fold_rows: detailed rows with extra metrics (for saving)
    """
    z_vals = np.asarray(avg_Z_fp[0, :]).astype(float)
    finite = np.isfinite(z_vals)
    sel = finite & (z_vals <= z_threshold)

    # Guard 1: no pattern selected
    if not np.any(sel):
        return [], []

    X_tmp = X[:, :, sel].reshape(X.shape[0], -1)

    # Guard 2: no features
    if X_tmp.shape[1] == 0:
        return [], []

    fold_scores: list[float] = []
    fold_rows: list[dict] = []

    for fold_id, train_idx, test_idx in shuffled_group_kfold(
        groups=Case, n_splits=n_folds, seed=rs
    ):
        X_train = cp.asarray(X_tmp[train_idx])
        X_test = cp.asarray(X_tmp[test_idx])
        y_train = cp.asarray(y_labels[train_idx].astype(np.int32))
        y_test = cp.asarray(y_labels[test_idx].astype(np.int32))

        need_proba = metric == "auc"

        # IMPORTANT FIX:
        # cuML default gamma='scale' can divide by 0 if X variance == 0.
        # Use gamma='auto' (depends only on n_features) + we already guard n_features>0.
        m = SVC(probability=need_proba, max_iter=max_iter, gamma="auto")

        try:
            m.fit(X_train, y_train)
        except ZeroDivisionError:
            # Ultra-robust fallback: if something still causes division by 0, skip this trial/threshold.
            return [], []
        except Exception:
            # Other errors should bubble up
            raise

        def _get(a):
            return a.get() if hasattr(a, "get") else a

        y_pred = _get(m.predict(X_test)).astype(np.int64)
        y_true = cp.asnumpy(y_test).astype(np.int64)

        probs = None
        if need_proba:
            probs = _get(m.predict_proba(X_test))

        primary = compute_primary_metric(
            metric=metric,
            y_true=y_true,
            y_pred=y_pred,
            probs=probs,
            classes=classes,
        )
        extras = compute_extra_metrics(y_true=y_true, y_pred=y_pred, classes=classes)

        fold_scores.append(float(primary))
        fold_rows.append(
            {
                "Task": class_task,
                "metric": metric,
                "z_threshold": int(z_threshold),
                "rs": int(rs),
                "Fold": int(fold_id),
                "score": float(primary),
                "n_params": int(X_tmp.shape[1]),
                **{
                    k: (
                        None
                        if (v is None or (isinstance(v, float) and np.isnan(v)))
                        else float(v)
                    )
                    for k, v in extras.items()
                },
            }
        )

    return fold_scores, fold_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="ALL",
        choices=["ALL", "healthy_vs_cancer", "cancer_vs_lr_hr", "MC"],
        help="Which task to tune (or ALL).",
    )
    parser.add_argument(
        "--metric",
        default="auc",
        choices=["auc", "acc", "f1"],
        help="Primary metric used for tuning z-threshold (paired t-test vs baseline).",
    )
    parser.add_argument("--n_trials", type=int, default=5)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--max_iter", type=int, default=5000)
    parser.add_argument("--results_dir", type=str, default="results")

    parser.add_argument(
        "--z_mode",
        default="quantiles",
        choices=["quantiles", "unique"],
        help="How to build candidate z_threshold list from Avg_Zs.npy.",
    )
    parser.add_argument(
        "--n_quantiles", type=int, default=40, help="Used if z_mode=quantiles."
    )
    parser.add_argument("--z_min", type=int, default=None)
    parser.add_argument("--z_max", type=int, default=None)

    parser.add_argument("--out_prefix", type=str, default="zthres_tuning")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    ensure_dir(results_dir)

    avg_Z_fp = np.load("source/Avg_Zs.npy")
    base_candidates = build_candidates(
        avg_Z_fp, args.z_mode, args.n_quantiles, args.z_min, args.z_max
    )

    tasks = (
        ["healthy_vs_cancer", "cancer_vs_lr_hr", "MC"]
        if args.task == "ALL"
        else [args.task]
    )
    metric = args.metric

    stats_rows = []
    detailed_fold_rows = []

    for task in tasks:
        print(
            f"\n=== Tuning z-threshold for task={task} | metric={metric} | GPU={USING_GPU} ==="
        )

        X, y_labels, _, Case, _, _Y = Results_data_loading(
            task, data_augmentation=False
        )
        classes = np.unique(y_labels)

        # Per-task candidate filtering so we never run with 0 selected patterns
        candidates = filter_candidates_for_task(base_candidates, avg_Z_fp, X)

        print(
            f"[{task}] candidates: {len(candidates)} (after filtering) | min={candidates.min()} max={candidates.max()}"
        )

        for z_thres in candidates:
            for rs in range(args.n_trials):
                fold_scores, fold_rows = run_fold_scores_for_threshold(
                    class_task=task,
                    metric=metric,
                    z_threshold=int(z_thres),
                    X=X,
                    y_labels=y_labels,
                    Case=Case,
                    classes=classes,
                    avg_Z_fp=avg_Z_fp,
                    n_folds=args.n_folds,
                    rs=rs,
                    max_iter=args.max_iter,
                )

                # Skip invalid runs cleanly
                if len(fold_scores) == 0:
                    continue

                stats_rows.append(
                    {
                        "Task": task,
                        "metric": metric,
                        "z_threshold": int(z_thres),
                        "rs": int(rs),
                        # keep key name to match your snippet structure
                        "fold auc": np.asarray(fold_scores, dtype=float),
                    }
                )
                detailed_fold_rows.extend(fold_rows)

    if len(stats_rows) == 0:
        raise RuntimeError(
            "No valid runs were produced. Check Avg_Zs / candidate filtering / data loading."
        )

    df_stats = pd.DataFrame(stats_rows)

    # Persist df_stats with arrays safely
    pkl_path = results_dir / f"{args.out_prefix}_df_stats_{args.task}_{metric}.pkl"
    df_stats.to_pickle(pkl_path)
    print(f"Saved df_stats (pickle): {pkl_path}")

    # CSV version with arrays as JSON strings
    df_stats_csv = df_stats.copy()
    df_stats_csv["fold auc"] = df_stats_csv["fold auc"].apply(
        lambda a: json.dumps([float(x) for x in a])
    )
    csv_path = results_dir / f"{args.out_prefix}_df_stats_{args.task}_{metric}.csv"
    df_stats_csv.to_csv(csv_path, index=False)
    print(f"Saved df_stats (csv): {csv_path}")

    # Save fold-level details with extra metrics
    df_folds = pd.DataFrame(detailed_fold_rows)
    folds_csv = results_dir / f"{args.out_prefix}_fold_details_{args.task}_{metric}.csv"
    df_folds.to_csv(folds_csv, index=False)
    print(f"Saved fold details: {folds_csv}")

    # -----------------------
    # Paired t-test vs Baseline (max z_threshold per task)
    # -----------------------
    results_list = []

    for task in df_stats["Task"].unique():
        task_stats = df_stats[df_stats["Task"] == task]

        # aggregate fold arrays from all trials per threshold
        agg_fold_scores = {}
        for z_thres, group_df in task_stats.groupby("z_threshold"):
            fold_values = np.concatenate(group_df["fold auc"].values)
            agg_fold_scores[int(z_thres)] = fold_values

        if len(agg_fold_scores) == 0:
            continue

        end_thresh = max(agg_fold_scores.keys())
        end_fold_values = agg_fold_scores[end_thresh]
        print(
            f"[{task}] Baseline (z={end_thresh}) avg {metric}: {np.mean(end_fold_values):.6f}"
        )

        for z_thres, fold_values in agg_fold_scores.items():
            if z_thres == end_thresh:
                results_list.append(
                    {
                        "Task": task,
                        "metric": metric,
                        "z_threshold": z_thres,
                        "T-stat": None,
                        "p-value": None,
                        "h-val": None,
                    }
                )
                continue

            n = min(len(fold_values), len(end_fold_values))
            if n == 0:
                continue

            t_stat, p_val = ttest_rel(fold_values[:n], end_fold_values[:n])
            h_val = int(p_val < 0.05)

            results_list.append(
                {
                    "Task": task,
                    "metric": metric,
                    "z_threshold": int(z_thres),
                    "T-stat": float(t_stat),
                    "p-value": float(p_val),
                    "h-val": int(h_val),
                    "mean_score": float(np.mean(fold_values)),
                    "baseline_mean": float(np.mean(end_fold_values)),
                }
            )

    df_ttest_stats = pd.DataFrame(results_list)
    df_ttest_sorted = df_ttest_stats.sort_values(
        by="T-stat", ascending=False, na_position="last"
    ).reset_index(drop=True)

    ttest_csv = results_dir / f"{args.out_prefix}_paired_ttest_{args.task}_{metric}.csv"
    df_ttest_sorted.to_csv(ttest_csv, index=False)
    print(f"Saved t-test table: {ttest_csv}")
    print("\nTop rows:\n", df_ttest_sorted.head(20).to_string(index=False))

    # -----------------------
    # Pick best threshold per task
    # -----------------------
    best_rows = []
    for task in df_ttest_sorted["Task"].unique():
        sub = df_ttest_sorted[
            (df_ttest_sorted["Task"] == task) & (df_ttest_sorted["T-stat"].notna())
        ].copy()
        sig = sub[sub["h-val"] == 1].copy()

        if len(sig) > 0:
            best = sig.sort_values(
                ["T-stat", "mean_score"], ascending=[False, False]
            ).iloc[0]
            reason = "best_significant_tstat"
        else:
            if len(sub) == 0:
                continue
            best = sub.sort_values(["mean_score"], ascending=[False]).iloc[0]
            reason = "best_mean_fallback"

        best_rows.append(
            {
                "Task": task,
                "metric": metric,
                "best_z_threshold": int(best["z_threshold"]),
                "T-stat": None if pd.isna(best["T-stat"]) else float(best["T-stat"]),
                "p-value": None if pd.isna(best["p-value"]) else float(best["p-value"]),
                "h-val": None if pd.isna(best["h-val"]) else int(best["h-val"]),
                "mean_score": float(best["mean_score"]),
                "baseline_mean": float(best["baseline_mean"]),
                "selection_reason": reason,
            }
        )

    df_best = pd.DataFrame(best_rows).sort_values(["Task"]).reset_index(drop=True)

    best_csv = (
        results_dir / f"{args.out_prefix}_best_thresholds_{args.task}_{metric}.csv"
    )
    df_best.to_csv(best_csv, index=False)
    print(f"Saved best thresholds (csv): {best_csv}")
    print("\nBest thresholds:\n", df_best.to_string(index=False))

    # Update JSON store: task -> metric -> threshold
    best_json_path = results_dir / "z_thresholds_best.json"
    best_map = {}
    if best_json_path.exists():
        try:
            best_map = json.loads(best_json_path.read_text(encoding="utf-8"))
        except Exception:
            best_map = {}

    for r in best_rows:
        best_map.setdefault(r["Task"], {})
        best_map[r["Task"]][r["metric"]] = int(r["best_z_threshold"])

    best_json_path.write_text(json.dumps(best_map, indent=2) + "\n", encoding="utf-8")
    print(f"Updated z-threshold store: {best_json_path}")


if __name__ == "__main__":
    main()
