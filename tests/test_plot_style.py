import matplotlib

matplotlib.use("Agg")

from revision.plot_style import apply_style, class_color, color_palette_full


def test_class_palette_and_lesion_alias():
    apply_style()
    assert class_color("Healthy") == "#FFA600"
    assert class_color("Low-grade dysplasia") == color_palette_full["Non-malignant"]
    assert class_color("High-grade dysplasia") == color_palette_full["High risk"]
    assert len(set(color_palette_full.values())) == 6
    assert matplotlib.rcParams["axes.titleweight"] == "normal"
    assert matplotlib.colormaps["iivv"](1.0)[:3] == matplotlib.colors.to_rgb("#003F5C")
