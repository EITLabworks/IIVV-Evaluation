"""Shared class colors and IEEE-oriented figure defaults (no scientific settings)."""

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
# The lesion display label includes normal/benign tissue and mild dysplasia.
CLASS_COLORS = dict(
    color_palette_full,
    **{
        "Low-grade dysplasia": "#58508D",
        "Benign": "#58508D",
        "Other": "#003F5C",
        "All": "#003F5C",
    }
)


def class_color(label):
    return CLASS_COLORS.get(str(label), "#003F5C")


def apply_style():
    import matplotlib as mpl
    from cycler import cycler
    from matplotlib.colors import LinearSegmentedColormap

    if "iivv" not in mpl.colormaps:
        mpl.colormaps.register(
            LinearSegmentedColormap.from_list("iivv", ["white", "#003F5C"])
        )
    mpl.rcParams.update(
        {
            "axes.prop_cycle": cycler(
                color=list(dict.fromkeys(color_palette_full.values()))
            ),
            "image.cmap": "iivv",
            "axes.titleweight": "normal",
            "font.family": "serif",
            "axes.facecolor": "white",
            "axes.grid": True,
            "axes.grid.which": "both",
            "grid.color": "#003F5C",
            "grid.alpha": 0.12,
            "grid.linewidth": 0.4,
            "axes.linewidth": 0.5,
            "legend.framealpha": 1,
            "legend.edgecolor": "#333333",
        }
    )
