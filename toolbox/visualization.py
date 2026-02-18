from collections import Counter
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix
import os
import ast

color_palette_full = {
    "Healthy": "#FFA600",
    "Non-malignant": "#58508D",
    "Dysplasia": "#7A5195",
    "Low risk": "#58508D",
    "High risk": "#FF6361",
    "Non-cancer": "#003F5C",
    "High-grade dysplasia": "#FF6361",
    "Cancer": "#DD5182",
}


custom_colors = list(color_palette_full.values())
custom_cmap = mcolors.ListedColormap(custom_colors, name="custom_disease_cmap")

cmap_gradient = mcolors.LinearSegmentedColormap.from_list(
    "custom_gradient",
    [
        color_palette_full["Non-cancer"],
        color_palette_full["Cancer"],
        color_palette_full["Healthy"],
    ],
)


def visualize_Category_hist(Category, s_name=None):
    """
    Visualizes a histogram of class/category distribution with category-specific color mapping.

    Parameters
    ----------
    Category : list of str
        List of class labels to visualize.
    s_name : str
        Name of the saved filed, default None

    Returns
    -------
    None
    """
    Category = [str(c) for c in Category]
    counts = Counter(Category)
    categories = list(counts.keys())
    values = list(counts.values())
    colors = [
        color_palette_full[cat] if cat in color_palette_full else "blue"
        for cat in categories
    ]
    # Plot
    plt.figure(figsize=(6, 4))
    plt.bar(categories, values, color=colors)
    plt.xlabel("Category")
    plt.ylabel("Number of Samples")
    plt.grid(True, axis="y")
    plt.xticks(rotation=45)
    plt.tight_layout()
    if s_name != None:
        plt.savefig(s_name, bbox_inches="tight")
    plt.show()


def plot_cv_heatmap(X_tmp, Y_labels, Case, n_trials, n_folds):
    n_patients = len(X_tmp)
    split_data = []
    trial_fold_labels = []
    for rs in range(n_trials):
        cv = GroupShuffleSplit(n_splits=n_folds, random_state=rs)
        fold = 1
        for train_idx, test_idx in cv.split(X_tmp, Y_labels, Case):
            patient_split = np.zeros(n_patients)  # 0 = train, 1 = test
            patient_split[test_idx] = 1
            split_data.append(patient_split)
            trial_fold_labels.append(f"RS{rs}_F{fold}")
            fold += 1
    # Convert to matrix (patients x trial-folds)
    split_matrix = np.array(split_data).T
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ["C0", "C1"]
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(colors)

    im = ax.imshow(split_matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    # Set labels
    ax.set_xlabel("Random state and trial")
    ax.set_ylabel("Sample index")
    # Set ticks
    ax.set_xticks(range(len(trial_fold_labels)))
    ax.set_xticklabels(trial_fold_labels, rotation=90, ha="right")
    ax.set_yticks(
        range(0, n_patients, max(1, n_patients // 20))
    )  # Show every nth patient
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix_custom(
    y_true,
    y_pred,
    class_names=None,
    normalize=False,
    weighted=False,
    sample_weights=None,
    path=None,
    cmap="Blues",
):

    if weighted and sample_weights is not None:
        cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weights)
    else:
        cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_display = cm * 100
    else:
        cm_display = cm

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm_display, cmap=cmap)

    # plt.colorbar(im)

    if class_names is None:
        class_names = np.unique(y_true)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    plt.xlabel("Predicted")
    plt.ylabel("True")

    thresh = cm_display.max() / 2

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize:
                text = f"{cm_display[i,j]:.1f}%"
            else:
                text = f"{int(cm_display[i,j])}"

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if cm_display[i, j] > thresh else "black",
            )

    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
    plt.show()


def plot_freq_rank_heatmap(
    class_task: str,
    metric: str,
    results_dir: str = "results",
    figures_dir: str = "figures",
    filename_prefix: str = "FREQ-analysis",
    cmap=None,  # e.g. plt.cm.viridis
    figsize=(9, 7),
    annotate: bool = True,
):
    """
    Loads results/{filename_prefix}_{class_task}_{metric}_gpu.csv,
    builds a (freq x fold) matrix of mean ranks across rs, sorts freqs by overall mean rank,
    and plots/saves a heatmap PDF.

    Parameters
    ----------
    class_task : str
        e.g. "MC"
    metric : str
        e.g. "f1"
    results_dir : str
        Folder where the CSV lives.
    figures_dir : str
        Folder to save the PDF into.
    filename_prefix : str
        Prefix used in the CSV filename.
    cmap : matplotlib colormap or None
        If None, defaults to plt.cm.viridis.
    figsize : tuple
        Figure size.
    annotate : bool
        Whether to draw integer rank numbers in each cell.

    Returns
    -------
    dict with keys:
        df, folds, rs_list, max_freqs, sorted_freq_ids, rank_matrix_sorted, mean_rank, fig, ax, out_path
    """
    if cmap is None:
        cmap = plt.cm.viridis

    in_path = os.path.join(
        results_dir, f"{filename_prefix}_{class_task}_{metric}_gpu.csv"
    )
    df = pd.read_csv(in_path)

    def parse_list(x):
        if isinstance(x, list):
            return x
        if pd.isna(x):
            return []
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except Exception:
                return []
        return []

    if "fold_top_freq_indices" not in df.columns:
        raise KeyError("CSV must contain a 'fold_top_freq_indices' column.")
    if "Fold" not in df.columns or "rs" not in df.columns:
        raise KeyError("CSV must contain 'Fold' and 'rs' columns.")

    df["fold_top_freq_indices"] = df["fold_top_freq_indices"].apply(parse_list)

    max_freqs = max((len(x) for x in df["fold_top_freq_indices"]), default=0)
    if max_freqs == 0:
        raise ValueError("No frequency indices found (all lists empty).")

    freq_ids = np.arange(max_freqs)

    def ranks_from_order(order, max_freqs_):
        r = np.full(max_freqs_, np.nan, dtype=float)
        for pos, f in enumerate(order, start=1):
            try:
                fi = int(f)
            except (TypeError, ValueError):
                continue
            if 0 <= fi < max_freqs_:
                r[fi] = pos
        return r

    folds = sorted(df["Fold"].unique())
    rs_list = sorted(df["rs"].unique())

    rank_by_rs_fold = {}  # (rs, fold) -> (max_freqs,) ranks
    for _, row in df.iterrows():
        rs = int(row["rs"])
        fold = int(row["Fold"])
        order = row["fold_top_freq_indices"]
        rank_by_rs_fold[(rs, fold)] = ranks_from_order(order, max_freqs)

    rank_matrix = np.zeros((max_freqs, len(folds)), dtype=float)

    for j, fold in enumerate(folds):
        all_rs_ranks = []
        for rs in rs_list:
            r = rank_by_rs_fold.get((rs, fold), None)
            # keep only complete rows (same as your original logic)
            if r is not None and np.isfinite(r).all():
                all_rs_ranks.append(r)

        if len(all_rs_ranks) == 0:
            rank_matrix[:, j] = np.nan
        else:
            all_rs_ranks = np.vstack(all_rs_ranks)  # (n_rs, max_freqs)
            rank_matrix[:, j] = np.nanmean(all_rs_ranks, axis=0)

    mean_rank = np.nanmean(rank_matrix, axis=1)
    sort_idx = np.argsort(mean_rank)  # best -> worst

    rank_matrix_sorted = rank_matrix[sort_idx, :]
    sorted_freq_ids = freq_ids[sort_idx]
    annot = np.rint(rank_matrix_sorted).astype(int)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(rank_matrix_sorted, aspect="auto", cmap=cmap)

    ax.tick_params(axis="both", labelsize=12)
    ax.set_xticks(np.arange(len(folds)))
    ax.set_xticklabels([f"Fold {f}" for f in folds])

    ax.set_yticks(np.arange(max_freqs))
    ax.set_yticklabels([f"{i+1}" for i in sorted_freq_ids])  # +1 for "Freq 1..N"

    ax.set_ylabel("Top Frequencies", fontsize=14)

    if annotate:
        for i in range(rank_matrix_sorted.shape[0]):
            for j in range(rank_matrix_sorted.shape[1]):
                if np.isfinite(rank_matrix_sorted[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{annot[i, j]}",
                        ha="center",
                        va="center",
                        fontsize=14,
                        color="white",
                    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Rank Position (lower is better)", fontsize=14)

    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, f"1_2_freq_{class_task}_{metric}.pdf")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()


def plot_mean_across_folds(
    class_task: str,
    metric: str,
    results_dir: str = "results",
    filename_prefix: str = "FREQ-analysis",
    figures_dir: str = "figures",
    save: bool = True,
    show: bool = True,
):
    """
    Reads a CSV like:
      results/FREQ-analysis_{class_task}_{metric}_gpu.csv

    Expects columns:
      - Fold
      - rs
      - scores_by_n  (list of metric values for n=1..max_freqs)

    Produces a plot:
      - one line per fold = mean(scores_by_n) across all rs for that fold
      - one line for overall mean across folds (mean of fold means)

    Returns (x, fold_means_dict, overall_mean, fig, ax, out_path)
    """

    def _parse_list(x):
        if isinstance(x, list):
            return x
        if pd.isna(x):
            return []
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except Exception:
                return []
        return []

    in_path = os.path.join(
        results_dir, f"{filename_prefix}_{class_task}_{metric}_gpu.csv"
    )
    df = pd.read_csv(in_path)

    if "Fold" not in df.columns or "scores_by_n" not in df.columns:
        raise KeyError("CSV must contain columns 'Fold' and 'scores_by_n'.")

    df["scores_by_n"] = df["scores_by_n"].apply(_parse_list)

    # Determine max length (n of top frequencies)
    max_n = max((len(v) for v in df["scores_by_n"]), default=0)
    if max_n == 0:
        raise ValueError("scores_by_n is empty for all rows.")

    x = np.arange(1, max_n + 1)

    # Fold mean curves (average across rs inside each fold)
    folds = sorted(df["Fold"].unique())
    fold_means = {}

    for fold in folds:
        sub = df[df["Fold"] == fold]["scores_by_n"].tolist()
        # pad to max_n with NaN just in case lengths differ
        arr = np.full((len(sub), max_n), np.nan, dtype=float)
        for i, seq in enumerate(sub):
            seq = seq[:max_n]
            arr[i, : len(seq)] = np.asarray(seq, dtype=float)
        fold_means[fold] = np.nanmean(arr, axis=0)

    # Overall mean across folds (mean of fold means, not weighted by sample count)
    fold_stack = np.vstack([fold_means[f] for f in folds])
    overall_mean = np.nanmean(fold_stack, axis=0)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(6, 3))

    # Mean line first (so it appears in legend nicely)
    ax.plot(
        x,
        overall_mean,
        marker="x",
        linewidth=2.5,
        color="black",
        label=f"Mean {metric.upper()}",
    )

    for fold in folds:
        ax.plot(
            x,
            fold_means[fold],
            marker="o",
            alpha=0.4,
            linewidth=1,
            label=f"Fold {fold}",
        )

    ax.set_xlabel(r"Number of top frequencies $\mathbf{f}_\mathrm{T}$")
    ax.set_ylabel(metric.upper())
    ax.grid()
    ax.legend(loc="lower right")

    out_path = None
    if save:
        os.makedirs(figures_dir, exist_ok=True)
        out_path = os.path.join(figures_dir, f"baselineaucs_{class_task}_{metric}.pdf")
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
    if show:
        plt.tight_layout()
        plt.show()
    else:
        plt.close(fig)


def plot_fold_means_with_error(
    class_task: str,
    metric: str,
    results_dir: str = "results",
    filename_prefix: str = "FREQ-analysis",
    figures_dir: str = "figures",
    value_col: str = "scores_by_n",  # column containing list over n=1..T
    fold_col: str = "Fold",
    rs_col: str = "rs",
    error: str = "sem",  # "sem" or "std"
    error_style: str = "band",  # "band" or "bars"
    bars_every: int = 2,  # only used for error_style="bars"
    capsize: int = 3,  # only used for error_style="bars"
    show_overall_mean: bool = True,  # mean across folds (of fold means)
    overall_error: str | None = "sem",  # None, "sem", "std" across folds
    save: bool = True,
    show: bool = True,
    figsize=(6, 3),
):
    """
    Plot, for each fold:
      - mean across all rs at each n
      - error indicators across rs (SEM or STD)
    Optionally plots an overall mean across folds (mean of fold means) with optional error across folds.

    STD ... Standard Deviation
    SEM ... Standard Error of the Mean

    CSV filename:
      {results_dir}/{filename_prefix}_{class_task}_{metric}_gpu.csv

    Expected columns:
      - Fold
      - rs
      - scores_by_n  (list of metric values for n=1..T)
    """

    def _parse_list(x):
        if isinstance(x, list):
            return x
        if pd.isna(x):
            return []
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except Exception:
                return []
        return []

    def _pad_to_max(seqs, max_n):
        arr = np.full((len(seqs), max_n), np.nan, dtype=float)
        for i, seq in enumerate(seqs):
            seq = list(seq)[:max_n]
            if len(seq):
                arr[i, : len(seq)] = np.asarray(seq, dtype=float)
        return arr

    def _err(arr, kind: str):
        # arr: (n_rs, T)
        if kind == "std":
            return np.nanstd(arr, axis=0, ddof=1)
        if kind == "sem":
            s = np.nanstd(arr, axis=0, ddof=1)
            n = np.sum(np.isfinite(arr), axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                return s / np.sqrt(n)
        raise ValueError("error must be 'sem' or 'std'")

    in_path = os.path.join(
        results_dir, f"{filename_prefix}_{class_task}_{metric}_gpu.csv"
    )
    df = pd.read_csv(in_path)

    for col in (fold_col, rs_col, value_col):
        if col not in df.columns:
            raise KeyError(f"CSV must contain column '{col}'.")

    df[value_col] = df[value_col].apply(_parse_list)

    max_n = max((len(v) for v in df[value_col]), default=0)
    if max_n == 0:
        raise ValueError(f"Column '{value_col}' is empty for all rows.")

    x = np.arange(1, max_n + 1)
    folds = sorted(df[fold_col].unique())

    fold_mean = {}
    fold_err = {}

    # ---- compute mean/error across rs for each fold ----
    for fold in folds:
        seqs = df.loc[df[fold_col] == fold, value_col].tolist()
        arr = _pad_to_max(seqs, max_n)  # (n_rs, T)
        fold_mean[fold] = np.nanmean(arr, axis=0)
        fold_err[fold] = _err(arr, error)

    # ---- overall mean across folds (mean of fold means) ----
    overall_mean = None
    overall_err_vals = None
    if show_overall_mean:
        stack = np.vstack([fold_mean[f] for f in folds])  # (n_folds, T)
        overall_mean = np.nanmean(stack, axis=0)
        if overall_error in ("sem", "std"):
            if overall_error == "std":
                overall_err_vals = np.nanstd(stack, axis=0, ddof=1)
            else:
                s = np.nanstd(stack, axis=0, ddof=1)
                n = np.sum(np.isfinite(stack), axis=0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    overall_err_vals = s / np.sqrt(n)

    # ---- plot ----
    fig, ax = plt.subplots(figsize=figsize)

    # overall mean first (like in your example legend)
    if show_overall_mean and overall_mean is not None:
        ax.plot(
            x,
            overall_mean,
            marker="x",
            linewidth=2.5,
            color="black",
            label=f"Mean {metric.upper()}",
        )
        if overall_err_vals is not None:
            if error_style == "band":
                ax.fill_between(
                    x,
                    overall_mean - overall_err_vals,
                    overall_mean + overall_err_vals,
                    alpha=0.15,
                )
            elif error_style == "bars":
                idx = np.arange(0, max_n, bars_every)
                ax.errorbar(
                    x[idx],
                    overall_mean[idx],
                    yerr=overall_err_vals[idx],
                    fmt="none",
                    capsize=capsize,
                )

    # fold curves + error indicators (across rs)
    for fold in folds:
        y = fold_mean[fold]
        e = fold_err[fold]
        ax.plot(x, y, marker="o", alpha=0.5, linewidth=1.5, label=f"Fold {fold}")

        if error_style == "band":
            ax.fill_between(x, y - e, y + e, alpha=0.12)
        elif error_style == "bars":
            idx = np.arange(0, max_n, bars_every)
            ax.errorbar(x[idx], y[idx], yerr=e[idx], fmt="none", capsize=capsize)
        else:
            raise ValueError("error_style must be 'band' or 'bars'")

    ax.set_xlabel(r"Number of top frequencies $\mathbf{f}_\mathrm{T}$")
    ax.set_ylabel(metric.upper())
    ax.grid()
    ax.legend(loc="lower right")

    out_path = None
    if save:
        os.makedirs(figures_dir, exist_ok=True)
        out_path = os.path.join(
            figures_dir, f"baselineaucs_{class_task}_{metric}_folds_err.pdf"
        )
        plt.tight_layout()
        plt.savefig(out_path)

    if show:
        plt.tight_layout()
        plt.show()
    else:
        plt.close(fig)
