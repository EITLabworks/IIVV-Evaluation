#!/usr/bin/env python3
"""
Frequency selection evaluation (GPU) with the same CLI/structure/output style
as 1_1iivv_gpu_eval.py, but iterating over top-N frequencies.

Outputs (per task + metric):
- results/FREQ-analysis_<task>_<metric>_gpu.csv              (raw fold stats, incl. selected freqs)
- results/FreqSel_ranking_<task>_<metric>.csv               (ranking table for N=1..max_freqs)
- results/FreqSel_ranking_<task>_<metric>.txt               (order good->bad, by N)
- results/FreqSel_ranking_<task>_<metric>.json              (order + table)

Additionally, for convenience (named "according to the frequency"):
- results/FreqSel_N<NN>_ranking_<task>_<metric>.{csv,txt,json}
  (same ranking content, but filename includes N)
"""

import argparse
import json
from pathlib import Path

import cupy as cp
import numpy as np
import pandas as pd
from cuml.decomposition import PCA
from cuml.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

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


def compute_metric_from_proba(
    metric: str, y_true: np.ndarray, proba: np.ndarray, n_classes: int
) -> float:
    if metric == "auc":
        if n_classes == 2:
            return float(roc_auc_score(y_true, proba[:, 1]))
        # multiclass ovr macro (like your script)
        if y_true.ndim == 1:
            y_bin = np.zeros((y_true.size, n_classes), dtype=np.int32)
            y_bin[np.arange(y_true.size), y_true.astype(int)] = 1
        else:
            y_bin = y_true
        return float(roc_auc_score(y_bin, proba, average="macro", multi_class="ovr"))

    y_pred = np.argmax(proba, axis=1)

    if metric == "acc":
        return float(accuracy_score(y_true, y_pred))

    if metric == "f1":
        if n_classes == 2:
            # binary f1 expects 0/1; y_pred is already 0/1 (class index)
            return float(f1_score(y_true, y_pred, average="binary"))
        return float(f1_score(y_true, y_pred, average="macro"))

    raise ValueError(f"Unknown metric: {metric}")


def top_freq_indices_via_pca(
    X_train: np.ndarray, max_freqs: int, top_components: int = 10
) -> np.ndarray:
    """
    Compute per-frequency scores using PCA loading energy from the first `top_components` components,
    then return indices sorted descending (best->worst).
    """
    # Shape: (train_samples, freqs, patterns)
    # Flatten patterns into samples for PCA: (train_samples*patterns, freqs)
    X_train_clean = X_train.transpose(0, 2, 1).reshape(-1, X_train.shape[1])

    X_gpu = cp.asarray(X_train_clean, dtype=cp.float32)

    n_components = min(max_freqs, X_gpu.shape[0], X_gpu.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(X_gpu)

    loading_matrix = pca.components_.get()  # (n_components, freqs) on CPU
    k = min(top_components, loading_matrix.shape[0])
    frequency_scores = np.sum(loading_matrix[:k] ** 2, axis=0)  # (freqs,)
    return np.argsort(frequency_scores)[::-1]


def eval_one_fold(
    *,
    class_task: str,
    rs: int,
    fold_id: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    case: np.ndarray,
    metric: str,
    max_iter: int,
    max_freqs: int,
):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Determine top frequencies for THIS fold
    fold_top_freq_indices = top_freq_indices_via_pca(
        X_train, max_freqs=max_freqs, top_components=10
    )

    n_classes = int(np.unique(y).size)

    # We only need proba for auc; for accuracy/f1 we can do probability=False and use predict
    need_proba = metric == "auc"

    scores_by_n = np.full(max_freqs, np.nan, dtype=np.float64)

    for n in range(1, max_freqs + 1):
        selected_freqs = fold_top_freq_indices[:n]

        X_train_filtered = X_train[:, selected_freqs, :]
        X_test_filtered = X_test[:, selected_freqs, :]

        X_train_flat = cp.asarray(
            X_train_filtered.reshape(X_train_filtered.shape[0], -1), dtype=cp.float32
        )
        X_test_flat = cp.asarray(
            X_test_filtered.reshape(X_test_filtered.shape[0], -1), dtype=cp.float32
        )
        y_train_gpu = cp.asarray(y_train.astype(np.int32))

        m = SVC(probability=need_proba, max_iter=max_iter)
        m.fit(X_train_flat, y_train_gpu)

        if need_proba:
            y_score = m.predict_proba(X_test_flat).get()  # CPU
            score = compute_metric_from_proba(metric, y_test, y_score, n_classes)
        else:
            # predict gives labels; we compute accuracy/f1 directly
            y_pred = m.predict(X_test_flat).get().astype(np.int64)
            if metric == "acc":
                score = float(accuracy_score(y_test, y_pred))
            elif metric == "f1":
                if n_classes == 2:
                    score = float(f1_score(y_test, y_pred, average="binary"))
                else:
                    score = float(f1_score(y_test, y_pred, average="macro"))
            else:
                raise ValueError(f"Unknown metric: {metric}")

        scores_by_n[n - 1] = score

    return {
        "Task": class_task,
        "rs": int(rs),
        "Fold": int(fold_id),
        "metric": metric,
        "need_proba": bool(need_proba),
        "max_freqs": int(max_freqs),
        "fold_top_freq_indices": fold_top_freq_indices.tolist(),
        "avg_fold_score": float(np.nanmean(scores_by_n)),
        # store per-N scores compactly
        "scores_by_n": scores_by_n.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", required=True, choices=["healthy_vs_cancer", "cancer_vs_lr_hr", "MC"]
    )
    parser.add_argument("--metric", default="auc", choices=["auc", "acc", "f1"])
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--max_iter", type=int, default=5000)
    parser.add_argument("--max_freqs", type=int, default=31)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--logs_dir", type=str, default="logs")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    logs_dir = Path(args.logs_dir)
    ensure_dir(results_dir)
    ensure_dir(logs_dir)

    class_task = args.task
    metric = args.metric
    max_freqs = int(args.max_freqs)

    print(
        f"\n=== Frequency selection | task: {class_task} | metric: {metric} | max_freqs: {max_freqs} ==="
    )

    X, y_labels, labels, Case, inverse_mapping, Y = Results_data_loading(
        class_task, data_augmentation=False
    )

    stats = []
    for rs in range(args.n_trials):
        for fold_id, train_idx, test_idx in shuffled_group_kfold(
            Case, n_splits=args.n_folds, seed=rs
        ):
            stats.append(
                eval_one_fold(
                    class_task=class_task,
                    rs=rs,
                    fold_id=fold_id,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    X=X,
                    y=y_labels,
                    case=Case,
                    metric=metric,
                    max_iter=args.max_iter,
                    max_freqs=max_freqs,
                )
            )

    df_stats = pd.DataFrame(stats)

    # Save raw fold stats
    out_csv = results_dir / f"FREQ-analysis_{class_task}_{metric}_gpu.csv"
    df_stats.to_csv(out_csv, index=False)
    print(f"Saved fold stats: {out_csv}")

    # ---- Build ranking across N (good -> bad) ----
    # Expand scores_by_n into long format: one row per (trial, fold, N)
    rows = []
    for r in stats:
        scores = r["scores_by_n"]
        for n in range(1, max_freqs + 1):
            rows.append(
                {
                    "Task": r["Task"],
                    "metric": r["metric"],
                    "rs": r["rs"],
                    "Fold": r["Fold"],
                    "N_freqs": n,
                    "score": float(scores[n - 1]),
                }
            )
    df_long = pd.DataFrame(rows)

    df_rank = (
        df_long.groupby(["N_freqs"], as_index=False)
        .agg(
            mean_score=("score", "mean"),
            std_score=("score", "std"),
            n_runs=("score", "count"),
        )
        .sort_values(["mean_score", "std_score"], ascending=[False, True])
        .reset_index(drop=True)
    )

    # Save ranking (like mask selection)
    rank_csv = results_dir / f"FreqSel_ranking_{class_task}_{metric}.csv"
    df_rank.to_csv(rank_csv, index=False)
    print(f"Saved ranking CSV: {rank_csv}")

    ordered = df_rank["N_freqs"].tolist()
    ordered_str = [int(n) for n in ordered]

    rank_txt = results_dir / f"FreqSel_ranking_{class_task}_{metric}.txt"
    rank_txt.write_text("\n".join(map(str, ordered_str)) + "\n", encoding="utf-8")
    print(f"Saved ranking TXT: {rank_txt}")

    rank_json = results_dir / f"FreqSel_ranking_{class_task}_{metric}.json"
    rank_json.write_text(
        json.dumps(
            {
                "task": class_task,
                "metric": metric,
                "order_good_to_bad": ordered_str,
                "ranking_table": df_rank.to_dict(orient="records"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved ranking JSON: {rank_json}")

    print(f"\nTop-N frequency counts (good -> bad, metric: {metric}):")
    print(df_rank.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
