from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd

color_palette_full = {
    "Healthy": "#003F5C",
    "Benign": "#58508D",
    "Dysplasia": "#7A5195",
    "Mild dysplasia": "#BC5090",
    "Non-cancer": "#DD5182",
    "Severe dysplasia": "#FF6361",
    "Cancer": "#FFA600",
}


def visualize_Category_hist(Category):
    """
    Visualizes a histogram of class/category distribution with category-specific color mapping.

    Parameters
    ----------
    Category : list of str
        List of class labels to visualize.

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
    plt.show()
