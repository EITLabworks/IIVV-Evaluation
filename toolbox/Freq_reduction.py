from toolbox.visualization import cmap_gradient
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def select_random_state(df, random_state):
    df_rs = df[df["rs"] == random_state]
    return df_rs


def select_fold_rankings(df, random_state):
    df_rs = select_random_state(df, random_state)

    fold_rankings_baseline = []
    for _, subdf in df_rs.groupby("Fold"):
        s = subdf["fold_top_freq_indices"].iloc[0]
        arr = np.fromstring(s.strip("[]"), sep=" ", dtype=int)
        fold_rankings_baseline.append(arr)
    fold_rankings_baseline = np.array(fold_rankings_baseline)
    return fold_rankings_baseline


def select_aucs(df, random_state):
    df_rs = select_random_state(df, random_state)

    fold_auc_results = []
    for _, subdf in df_rs.groupby("Fold"):
        s = subdf["fold_auc_results"].iloc[0]
        arr = np.fromstring(s.strip("[]"), sep=" ", dtype=float)
        fold_auc_results.append(arr)
    fold_auc_results = np.array(fold_auc_results)
    return fold_auc_results


def select_composite_freq_indices(df, random_state):
    df_rs = select_random_state(df, random_state)

    composite_freq_indices = []
    for _, subdf in df_rs.groupby("Fold"):
        s = subdf["composite_freq_indices"].iloc[0]
        arr = np.fromstring(s.strip("[]"), sep=" ", dtype=int)
        composite_freq_indices.append(arr)
    composite_freq_indices = np.array(composite_freq_indices)
    return composite_freq_indices


def plot_fold_rankings_rework(df, random_state, class_task="unnamed", top_n_freqs=31):

    fold_rankings_baseline = select_fold_rankings(df, random_state)
    # create the baseline dict
    fold_rankings_dict = {
        "Baseline": fold_rankings_baseline,
    }

    for name, rankings in fold_rankings_dict.items():
        n_folds = len(rankings)
        n_freqs = len(rankings[0])

        # Composite ranking across folds for sorting
        score_accumulator = np.zeros(n_freqs)
        for fold in rankings:
            for rank, freq_idx in enumerate(fold):
                score_accumulator[freq_idx] += n_freqs - rank

        composite_ranking = np.argsort(score_accumulator)[::-1][:top_n_freqs]

        # Create rank matrix: rows = top freqs, cols = folds
        rank_matrix = np.full((top_n_freqs, n_folds), np.nan)
        for f_idx, fold in enumerate(rankings):
            for i, freq in enumerate(composite_ranking):
                pos = np.where(fold == freq)[0]
                if pos.size > 0:
                    rank_matrix[i, f_idx] = pos[0] + 1  # +1 for 1-based rank

        plt.figure(figsize=(9, 7))
        sns.heatmap(
            rank_matrix,
            annot=True,
            fmt=".0f",
            cmap=cmap_gradient,
            cbar_kws={"label": "Rank Position (lower is better)"},
            xticklabels=[f"Fold {i+1}" for i in range(n_folds)],
            yticklabels=[f"Freq {f+1}" for f in composite_ranking],
            annot_kws={"fontsize": 14},
        )
        plt.xlabel("Fold", fontsize=14)
        plt.ylabel("Top Frequencies", fontsize=14)

        plt.xticks(fontsize=12)  # rotate if needed
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(
            f"results/Fold-wise_frequency_impact_rankings_rs_{random_state}_{name}_{class_task}.pdf",
            bbox_inches="tight",
        )
        plt.show()


def plot_no_of_top_frequencies_rework(df, random_state, class_task):

    df_rs = select_random_state(df, random_state)
    print(df_rs)
    max_len = max(len(x) for x in df_rs["fold_auc_results"])
    fold_arrays = [
        pd.Series(x).reindex(range(max_len), fill_value=None)
        for x in df_rs["fold_auc_results"]
    ]

    avg_auc = np.mean(select_aucs(df, random_state), axis=0)
    plt.figure(figsize=(6, 3))
    plt.plot(
        np.arange(1, len(avg_auc) + 1),
        avg_auc,
        marker="x",
        color="steelblue",
        linewidth=2,
        label="Mean AUC",
    )

    for _, row in df_rs.iterrows():
        auc_str = row["fold_auc_results"]
        auc_arr = np.fromstring(auc_str.strip("[]"), sep=" ", dtype=float)
        print("HIER", range(1, len(row["fold_auc_results"]) + 1))
        plt.plot(
            np.arange(1, len(avg_auc) + 1),
            auc_arr,
            "o-",
            alpha=0.5,
            label=f"Fold {row['Fold']}",
        )

    plt.xlabel(r"Number of top frequencies $\mathbf{f}_\mathrm{T}$")
    plt.xticks(range(1, 31 + 1, 3))
    plt.ylabel("AUC")
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(
        f"results/baselineaucs_rs_{random_state}_{class_task}.pdf", bbox_inches="tight"
    )
    plt.show()
