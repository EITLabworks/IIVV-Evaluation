"""Redraw saved predictions and descriptive coordinates; no training or PCA/MI fits."""

from pathlib import Path
import sys, json, shutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from revision.plot_style import apply_style, class_color
from revision.analysis import plot_outputs
from revision.data_quality import save_figure as original_save


def save_figure(fig, path, labels=()):
    target = Path("results/revision-palette") / Path(path).relative_to("results")
    target.parent.mkdir(parents=True, exist_ok=True)
    original_save(fig, target, labels)


def search_figures(root):
    for path in root.glob("*/inner_search.csv"):
        data = pd.read_csv(path)
        target = Path("results/revision-palette") / path.parent.relative_to("results")
        target.mkdir(parents=True, exist_ok=True)
        for stage, column, name in [
            ("frequency", "n_freq", "frequency_reduction"),
            ("mask", "mask", "iivv_selection"),
            ("quality", "cutoff", "quality_sensitivity"),
        ]:
            part = data[data.stage == stage]
            if part.empty:
                continue
            values = part.groupby(column).macro_auc.mean()
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(range(len(values)), values, "o-")
            ax.set(
                xticks=range(len(values)),
                xticklabels=values.index,
                xlabel=column,
                ylabel="Inner-selection macro AUC (descriptive)",
            )
            if stage == "mask":
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            original_save(fig, target / name)


def main():
    apply_style()
    root = Path("results/revision-three-tasks-v3")
    q = Path("results/revision-quality-v3")
    folder = q / "exploration"
    for path in folder.glob("*-pca.csv"):
        d = pd.read_csv(path)
        m = json.loads(path.with_suffix(".json").read_text())
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for name, g in d.groupby("class_name", sort=False):
            ax.scatter(
                g.PC1, g.PC2, s=20, alpha=0.65, color=class_color(name), label=name
            )
        ax.set(
            xlabel=f'PC1 ({m["explained_variance_ratio"][0]:.1%})',
            ylabel=f'PC2 ({m["explained_variance_ratio"][1]:.1%})',
        )
        ax.legend(fontsize=8)
        save_figure(fig, path, d.class_name)
    for path in folder.glob("*-mutual-information.csv"):
        d = pd.read_csv(path).sort_values("rank_site")
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(range(len(d)), d.mutual_information_site)
        if "patient_draw_mean" in d:
            ax.plot(
                range(len(d)),
                d.patient_draw_mean,
                "o",
                label="One-site-per-patient mean",
            )
        ax.set(
            xticks=range(len(d)),
            xticklabels=[f"{v:g}" for v in d.frequency_hz],
            xlabel="Frequency (Hz), ordered by exploratory MI",
            ylabel="Mutual information",
        )
        plt.setp(ax.get_xticklabels(), rotation=70, ha="right")
        save_figure(fig, path)
    d = pd.read_csv(folder / "task_quality_retention.csv")
    for task, g in d.groupby("task"):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for label, t in g.groupby("class_name"):
            ax.plot(
                t.cutoff,
                t.retained_sites / t.total_sites,
                "o-",
                color=class_color(label),
                label=label,
            )
        ax.set(
            xlabel="Strict acquisition-rejection cutoff (%)",
            ylabel="Retained site fraction",
            ylim=(0, 1.05),
        )
        ax.legend(fontsize=8)
        save_figure(fig, folder / f"{task}-quality-retention", g.class_name)
    d = pd.read_csv(folder / "mask_frequency_quality.csv").pivot(
        index="mask", columns="frequency_hz", values="nonfinite_percent"
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(d, aspect="auto", vmin=0, vmax=100)
    ax.grid(False)
    ax.set(yticks=range(len(d)), yticklabels=d.index, xlabel="Ordered frequency index")
    fig.colorbar(im, ax=ax, label="Non-finite coordinates (%)")
    save_figure(fig, folder / "mask-frequency-quality")
    d = pd.read_csv(q / "repeatability/icc_and_inter_site_variability.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, r in enumerate(d.itertuples()):
        ax.errorbar(
            i,
            r.icc_2_1_absolute,
            yerr=[
                [r.icc_2_1_absolute - r.icc_ci_low],
                [r.icc_ci_high - r.icc_2_1_absolute],
            ],
            fmt="o",
            color=class_color(r.class_name),
        )
    ax.set(
        xticks=range(len(d)),
        xticklabels=d.class_name,
        ylabel="ICC(2,1), absolute agreement",
    )
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    save_figure(fig, folder / "saved-test-mean-icc")
    a = pd.read_csv(folder / "site_quality.csv")
    b = pd.read_csv(folder / "frequency_quality.csv")
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].scatter(a.missing_percent, a.nonfinite_coordinate_percent, s=15, alpha=0.65)
    axs[0].set(
        xlabel="Mean acquisition rejection (%)",
        ylabel="Non-finite averaged coordinates (%)",
    )
    axs[1].semilogx(b.frequency_hz, b.median_magnitude_ohm)
    axs[1].fill_between(
        b.frequency_hz, b.magnitude_q25_ohm, b.magnitude_q75_ohm, alpha=0.25
    )
    axs[1].set(xlabel="Frequency (Hz)", ylabel="Magnitude median and IQR (ohm)")
    save_figure(fig, folder / "quality-overview")
    for task in [p for p in root.iterdir() if (p / "classes.json").is_file()]:
        classes = json.loads((task / "classes.json").read_text())["classes"]
        for folder in root.glob(f"plots_{task.name}_*"):
            for stream in [p for p in folder.iterdir() if p.is_dir()]:
                path = task / f"nested_cv_{stream.name}_site_predictions.csv"
                if not path.is_file():
                    continue
                target = Path("results/revision-palette") / stream.relative_to(
                    "results"
                )
                target.mkdir(parents=True, exist_ok=True)
                plot_outputs(
                    target, pd.read_csv(path), classes, f"{task.name}: {stream.name}"
                )
                for src in target.glob("*"):
                    if src.suffix not in {".png", ".pdf"} or src.stem not in {
                        "CM",
                        "CM_norm",
                        "ROC",
                    }:
                        continue
                    for dst in folder.glob(
                        src.stem.split("_")[0]
                        + "_*"
                        + stream.name.split("_")[-1]
                        + "*"
                        + src.suffix
                    ):
                        if src.stem == "ROC" or ("_norm" in src.stem) == (
                            "_norm" in dst.stem
                        ):
                            shutil.copyfile(src, target.parent / dst.name)
    search_figures(root)
    print("Redrawn saved figures only.")


if __name__ == "__main__":
    main()
