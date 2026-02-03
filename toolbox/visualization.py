from collections import Counter
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

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
