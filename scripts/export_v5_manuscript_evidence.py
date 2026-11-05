"""Export manuscript tables/figure data from frozen v5 results. Never fit a model.

Run from the repository root with the analysis environment. Existing manuscript
wording and original/deleted figures are deliberately left to editorial review.
"""

import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from revision.features import patient_weights
from revision.plot_style import apply_style, class_color

ROOT = Path("results/revision-three-tasks-v5")
QUALITY = Path("results/revision-quality-v5")
PAPER = Path("manuscript/Revision")
TABLES = PAPER / "revision-evidence"
FIGURES = PAPER / "figures_revision"
RESPONSE = Path("manuscript/Response_Letter/quality-evidence")
TASKS = ["healthy_vs_cancer", "highrisk_vs_lowrisk", "cancer_vs_lr_hr"]
COLORS = {"Healthy": "iivvHealthy", "Cancer": "iivvCancer", "Low risk": "iivvBenign",
          "High risk": "iivvRisk", "Low-grade dysplasia": "iivvBenign",
          "High-grade dysplasia": "iivvHighGrade"}
SOURCES = {}


def read(path):
    path = Path(path)
    SOURCES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text()) if path.suffix == ".json" else pd.read_csv(path)


def metric(task, stream="auc_selected", unit="patient_weighted_site"):
    rows = METRICS[task]
    row = rows[(rows.stream == stream) & (rows.unit == unit)]
    assert len(row) == 1
    return row.iloc[0]


def intervals(task, stream="auc_selected", unit="patient_weighted_site"):
    return read(ROOT / task / f"nested_cv_{stream}_{unit}_confidence_intervals.csv").set_index("metric")


def cell(value, ci=None):
    if not np.isfinite(value):
        return "--"
    if ci is None:
        return f"{value:.3f}"
    assert np.isfinite(ci.ci_low) and np.isfinite(ci.ci_high)
    return rf"\shortstack{{{value:.3f}\\({ci.ci_low:.3f}--{ci.ci_high:.3f})}}"


def table(path, caption, label, columns, header, rows, wide=True):
    env = "table*" if wide else "table"
    text = [rf"\begin{{{env}}}[t]", r"\centering", r"\scriptsize\setlength{\tabcolsep}{2.5pt}",
            rf"\caption{{\edit{{{caption}}}}}", rf"\label{{{label}}}",
            rf"\edit{{\begin{{tabular}}{{@{{}}{columns}@{{}}}}\toprule", header + r" \\\midrule"]
    text += [" & ".join(map(str, row)) + r" \\" for row in rows]
    text += [r"\bottomrule\end{tabular}}", rf"\end{{{env}}}"]
    path.write_text("\n".join(text) + "\n")


def diagnostic_tables():
    metrics = ["auc", "accuracy", "f1", "precision", "sensitivity", "specificity", "npv"]
    for stream, name, description in [("auc_selected", "primary", "AUC-selected"),
                                      ("sensitivity_selected", "sensitivity", "sensitivity-oriented")]:
        rows = []
        for i, task in enumerate(TASKS, 1):
            row, ci = metric(task, stream), intervals(task, stream)
            rows.append([i, *[cell(row[m], ci.loc[m]) for m in metrics],
                         rf"\shortstack{{{100*row.coverage:.1f}\%\\{int(row.n_covered)}/{int(row.n_sites)}}}"])
        table(TABLES / f"{name}-table.tex",
              f"Patient-equal site estimates for the {description} procedure after aggregation across five repetitions. "
              r"Parentheses give 95\% whole-patient bootstrap intervals. Precision equals PPV. "
              r"Coverage reports patient-equal percentage and evaluable/eligible sites. A site must pass quality in every repetition. "
              r"Task~3 reports macro one-versus-rest metrics. -- denotes an undefined metric, including macro precision when a class has no positive predictions.",
              "tab:revision_three_tasks" if name == "primary" else "tab:revision_sensitivity",
              "lcccccccc", r"Task & AUC & Accuracy & F1 & Precision/PPV & SN & SP & NPV & Coverage", rows)
    rows = []
    for i, task in enumerate(TASKS, 1):
        frame = read(ROOT / task / "per_class_metrics.csv")
        frame = frame[(frame.stream == "auc_selected") & (frame.unit == "site")]
        ci = read(ROOT / task / "nested_cv_auc_selected_site_per_class_confidence_intervals.csv")
        for row in frame.itertuples():
            c = ci[ci.class_index == row.class_index].set_index("metric")
            rows.append([i, row.class_name, int(row.support), int(row.tp), int(row.fn),
                         *[cell(getattr(row, m), c.loc[m]) for m in ["auc", "accuracy", "f1", "sensitivity", "specificity", "ppv", "npv"]]])
    table(TABLES / "class-table.tex",
          r"Unweighted site-level per-class results for the repeat-aggregated AUC-selected procedure, with 95\% whole-patient bootstrap intervals. "
          r"Accuracy and all diagnostic metrics use one-versus-rest classes. Precision equals PPV. Support excludes quality abstentions. Binary Task~2 alone uses risk labels. "
          r"In Task~3, Low-grade dysplasia includes normal/benign lesions and mild dysplasia, High-grade dysplasia includes moderate/severe dysplasia and CIS, and clinically healthy sites are excluded.",
          "tab:revision_class_errors", "lccccccccccc",
          r"Task & Class & Sites & TP & FN & AUC & OvR Acc. & F1 & SN & SP & PPV & NPV", rows)
    rows = []
    for i, task in enumerate(TASKS, 1):
        for model in ["svm", "rf", "lr"]:
            b, t = metric(task, f"baseline_{model}"), metric(task, f"auc_{model}")
            rows.append([i, model.upper(), cell(b.auc, intervals(task, f"baseline_{model}").loc["auc"]),
                         cell(t.auc, intervals(task, f"auc_{model}").loc["auc"]),
                         f"{int(b.n_covered)}/{int(b.n_sites)}", f"{int(t.n_covered)}/{int(t.n_sites)}"])
    table(TABLES / "model-table.tex",
          r"Patient-equal site AUC by classifier family, after aggregation across five repetitions, with patient-bootstrap 95\% intervals. "
          r"All/all uses all 31 frequencies and 7728 IIVV coordinates with the fixed reference preprocessing and $C_{th}=\SI{80}{\percent}$. "
          r"Tuned pipelines select inputs and settings within training. Paired CV tests use common coverage. No AUC comparison passes the global Holm correction.",
          "tab:revision_models", "lccccc", "Task & Model & All/all AUC & Tuned AUC & All/all sites & Tuned sites", rows, False)


def selected_tables():
    outputs, frames = [], []
    for i, task in enumerate(TASKS, 1):
        selected = [s for s in read(ROOT / task / "selections.json") if s["stream"] == "auc_selected"]
        assert len(selected) == 25 and len({(s["repeat"], s["outer_fold"]) for s in selected}) == 25
        rows = []
        for s in sorted(selected, key=lambda s: (s["repeat"], s["outer_fold"])):
            c = s["candidate"]; params = json.loads(c["params_json"])
            nf, npat = len(s["frequencies_hz"]), len(s["pattern_indices_0based"])
            components = 2 if c["representation"] == "magnitude_phase" else 1
            assert s["input_features"] == nf * npat * components
            mask = rf"$Z_{{q={c['mask'][3:]}}}$" if c["mask"].startswith("Zq:") else c["mask"]
            par = (rf"$m={params['max_features']:g}$" if c["model"] == "rf" else rf"$C={params['C']:g}$")
            if c["model"] == "svm":
                par += " RBF" if params["kernel"] == "rbf" else " Lin."
            par += " B" if c["class_weight"] == "balanced" else " U"
            rows.append([f"{s['repeat']+1}/{s['outer_fold']+1}", c["model"].upper(),
                         "PCA" if c["selector"] == "pca" else "MI", nf, mask, npat,
                         r"$M,\phi$" if components == 2 else "$M$",
                         {"freq": "Freq.", "global": "Global", "none": "None"}[c["normalization"]],
                         "Full" if c["cutoff"] > 100 else f"{c['cutoff']:g}", s["input_features"], par])
            frames.append(dict(task=task, repeat=s["repeat"], outer_fold=s["outer_fold"], **c,
                               frequencies_hz=json.dumps(s["frequencies_hz"]), n_iivv=npat,
                               pattern_indices_0based=json.dumps(s["pattern_indices_0based"]), input_features=s["input_features"]))
        path = TABLES / f"feature-table-task{i}.tex"
        table(path,
              rf"Task~{i}: AUC-selected configurations from the 25 outer training populations. Repeat/fold numbers are one-based. "
              r"$F$ is the selected frequency count, $P$ the IIVV count, $M$ magnitude, $\phi$ phase, and MI mutual information. "
              r"Numeric $C_{th}$ entries are rejection cutoffs in percent, applied as $r_i<C_{th}$. Full retains sites even at recorded 100\% rejection. "
              r"$Z_q$ denotes a training-fitted magnitude-quantile mask. RF uses 100 trees and depth 8 with feature fraction $m$. "
              r"SVM/LR show $C$. B/U denotes balanced/no class weighting. Patient-equal fitting weights apply throughout. The Inputs column counts scalar features, including both components when phase is used.",
              "tab:revision_selections" if i == 1 else f"tab:revision_selections_task{i}",
              "lcccccccccc", r"Repeat/fold & Model & Rank & $F$ & IIVV mask & $P$ & Signal & Scale & $C_{th}$ (\%) & Inputs & Parameter", rows)
        outputs.append(rf"\input{{revision-evidence/{path.name}}}")
    (TABLES / "feature-table.tex").write_text("\n".join(outputs) + "\n")
    pd.DataFrame(frames).to_csv(TABLES / "selected-configurations-v5.csv", index=False)


def figure_data():
    for i, task in enumerate(TASKS, 1):
        frame = read(ROOT / task / "nested_cv_auc_selected_site_predictions.csv")
        weights = patient_weights(frame.patient_id.to_numpy())
        good = frame.covered.to_numpy(bool); g = frame[good]
        classes = read(ROOT / task / "classes.json")["classes"]
        pc = read(ROOT / task / "per_class_metrics.csv")
        pc = pc[(pc.stream == "auc_selected") & (pc.unit == "patient_weighted_site")].set_index("class_index")
        roc_file = FIGURES / f"task{i}-roc.tikz.tex"; text = roc_file.read_text()
        aucs = []
        for k, name in enumerate(classes):
            area = roc_auc_score(g.y_true == k, g[f"p_{k}"], sample_weight=weights[good])
            assert np.isclose(area, pc.loc[k, "auc"], atol=1e-12)
            if len(classes) == 2 and k == 0:
                continue
            x, y, _ = roc_curve(g.y_true == k, g[f"p_{k}"], sample_weight=weights[good])
            pd.DataFrame({"x": x, "y": y}).to_csv(FIGURES / f"task{i}-roc-{k}.dat", index=False, sep=" ")
            text, count = re.subn(r"\\addlegendentry\{" + re.escape(name) + r" \([\d.]+\)\}",
                                  lambda _: rf"\addlegendentry{{{name} ({area:.3f})}}", text)
            assert count == 1
            aucs.append(area)
        assert np.isclose(np.mean(aucs), metric(task).auc, atol=1e-12)
        roc_file.write_text(text)
        cm = confusion_matrix(g.y_true, g.y_pred, labels=range(len(classes)))
        saved = read(ROOT / task / "nested_cv_auc_selected_site_confusion_counts.csv").iloc[:, 1:].to_numpy()
        assert np.array_equal(cm, saved) and cm.sum() == metric(task).n_covered
        path = FIGURES / f"task{i}-cm.tikz.tex"; text = path.read_text()
        values = iter(cm.ravel())
        text, n = re.subn(r"(at \(axis cs:[\d.]+,[\d.]+\) \{)\d+(\};)",
                          lambda m: m[1] + str(next(values)) + m[2], text)
        assert n == cm.size
        shades = iter((8 + 67 * cm / cm.sum(axis=1, keepdims=True)).astype(int).ravel())
        text, n = re.subn(r"iivvBlue!\d+", lambda _: f"iivvBlue!{next(shades)}", text)
        assert n == cm.size
        path.write_text(text)
        for kind in ["pca", "mi-pca"]:
            p = QUALITY / "exploration" / f"{task}-{kind}.csv"
            d = read(p)
            for k, name in enumerate(classes):
                d.loc[d.class_name == name, ["PC1", "PC2"]].to_csv(
                    FIGURES / f"task{i}-{kind}-{k}.dat", sep=" ", index=False)
            # Each panel's labels are replaced below together, in document order.
        if i <= 2:
            d = read(ROOT / task / "nested_cv_auc_selected_calibration.csv")
            d[d.class_name == classes[1]][["mean_probability", "observed_fraction"]].to_csv(
                FIGURES / f"task{i}-calibration.dat", sep=" ", index=False)
            d = read(ROOT / task / "nested_cv_auc_selected_decision_curve.csv")
            d = d[(d.class_name == classes[1]) & (d.threshold <= .5)]
            for col in ["net_benefit", "treat_all", "treat_none"]:
                d[["threshold", col]].to_csv(FIGURES / f"task{i}-decision-{col}.dat", sep=" ", index=False)
    path = FIGURES / "pca-comparison.tikz.tex"; text = path.read_text(); labels = []
    for kind in ["pca", "mi-pca"]:
        for task in TASKS:
            meta = read(QUALITY / "exploration" / f"{task}-{kind}.json")
            labels.extend(rf"PC{k+1} ({100*v:.1f}\%)" for k, v in enumerate(meta["explained_variance_ratio"]))
    labels = iter(labels)
    text, n = re.subn(r"PC[12] \([\d.]+\\%\)", lambda _: next(labels), text)
    assert n == 12
    path.write_text(text)
    path = FIGURES / "quality-repeatability.tikz.tex"; text = path.read_text()
    icc = read(QUALITY / "repeatability/icc_and_inter_site_variability.csv")
    values = iter(v for row in icc.itertuples() for v in (row.icc_ci_low, row.icc_ci_high, row.icc_2_1_absolute))
    a, b = text.split("title={(b) Saved test means}")
    b, n = re.subn(r"(\(\d,)[\d.]+(\))", lambda m: m[1] + f"{next(values):.10g}" + m[2], b)
    assert n == 15
    path.write_text(a + "title={(b) Saved test means}" + b)


def timing_figures():
    data = {t: read(ROOT / t / "measured_computation_times.csv") for t in TASKS}
    for filename, columns in [("timing-fit.tikz.tex", ["model_fit_seconds"]),
                              ("timing-inference.tikz.tex", ["test_transform_seconds", "test_predict_seconds"])]:
        points = []
        for column in columns:
            for i, task in enumerate(TASKS):
                for j, model in enumerate(["svm", "rf", "lr"]):
                    for stream, offset in [(f"baseline_{model}", -.12), (f"auc_{model}", .12)]:
                        values = data[task].loc[data[task].stream == stream, column]
                        assert len(values) == 25 and values.min() > 0
                        y = 3*i + j + 1 + offset
                        points += [f"({values.min():.10g},{y:g}) ({values.max():.10g},{y:g})",
                                   f"({values.median():.10g},{y:g})"]
        path = FIGURES / filename; text = path.read_text(); point_iter = iter(points)
        text, n = re.subn(r"coordinates \{[^}]*\}", lambda _: "coordinates {" + next(point_iter) + "}", text)
        assert n == len(points)
        path.write_text(text)


def quality_pngs():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from revision.data import TASK3_CAPTION

    apply_style(); RESPONSE.mkdir(exist_ok=True)

    def save(fig, name, labels=()):
        if "Low-grade dysplasia" in labels:
            fig.text(.5, .005, TASK3_CAPTION, ha="center", va="bottom", fontsize=7)
            fig.tight_layout(rect=(0, .12, 1, 1))
        else:
            fig.tight_layout()
        fig.savefig(RESPONSE / f"{name}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    for task in TASKS:
        for kind in ["pca", "mi-pca"]:
            d = read(QUALITY / "exploration" / f"{task}-{kind}.csv")
            meta = read(QUALITY / "exploration" / f"{task}-{kind}.json")
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            for label, g in d.groupby("class_name", sort=False):
                ax.scatter(g.PC1, g.PC2, s=20, alpha=.65, color=class_color(label), label=label)
            ax.set(xlabel=f"PC1 ({meta['explained_variance_ratio'][0]:.1%})",
                   ylabel=f"PC2 ({meta['explained_variance_ratio'][1]:.1%})")
            ax.legend(fontsize=8); save(fig, f"{task}-{kind}", list(d.class_name.unique()))
        d = read(QUALITY / "exploration" / f"{task}-mutual-information.csv").sort_values("rank_site")
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(range(len(d)), d.mutual_information_site, color="#003F5C", label="All-site MI")
        ax.plot(range(len(d)), d.patient_draw_mean, "o", color="#DD5182", label="One-site-per-patient mean")
        ax.set(xticks=range(len(d)), xticklabels=[f"{v:g}" for v in d.frequency_hz],
               xlabel="Frequency (Hz), ordered by exploratory MI", ylabel="Mutual information")
        plt.setp(ax.get_xticklabels(), rotation=70, ha="right"); ax.legend(fontsize=8)
        save(fig, f"{task}-mutual-information")
    d = read(QUALITY / "exploration/task_quality_retention.csv")
    for task, g in d.groupby("task"):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for label, part in g.groupby("class_name"):
            ax.plot(part.cutoff.clip(upper=100), part.retained_sites / part.total_sites,
                    "o-", color=class_color(label), label=label)
        ax.set(xlabel="Strict acquisition-rejection cutoff (%)", ylabel="Retained site fraction", ylim=(0, 1.05))
        ax.legend(fontsize=8); save(fig, f"{task}-quality-retention", list(g.class_name.unique()))
    d = read(QUALITY / "exploration/mask_frequency_quality.csv").pivot(index="mask", columns="frequency_hz", values="nonfinite_percent")
    fig, ax = plt.subplots(figsize=(10, 5)); im = ax.imshow(d, aspect="auto", vmin=0, vmax=100)
    ax.grid(False); ax.set(yticks=range(len(d)), yticklabels=d.index, xlabel="Ordered frequency index (0–30)")
    fig.colorbar(im, ax=ax, label="Non-finite coordinates (%)"); save(fig, "mask-frequency-quality")
    d = read(QUALITY / "repeatability/icc_and_inter_site_variability.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, row in enumerate(d.itertuples()):
        ax.errorbar(i, row.icc_2_1_absolute,
                    yerr=[[row.icc_2_1_absolute-row.icc_ci_low], [row.icc_ci_high-row.icc_2_1_absolute]],
                    fmt="o", color=class_color(row.class_name))
    ax.set(xticks=range(len(d)), xticklabels=d.class_name, ylabel="ICC(2,1), absolute agreement")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right"); save(fig, "saved-test-mean-icc")
    a = read(QUALITY / "exploration/site_quality.csv"); b = read(QUALITY / "exploration/frequency_quality.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(a.missing_percent, a.nonfinite_coordinate_percent, s=15, alpha=.65, color="#003F5C")
    axes[0].set(xlabel="Mean acquisition rejection (%)", ylabel="Non-finite averaged coordinates (%)")
    axes[1].semilogx(b.frequency_hz, b.median_magnitude_ohm, color="#003F5C")
    axes[1].fill_between(b.frequency_hz, b.magnitude_q25_ohm, b.magnitude_q75_ohm, color="#003F5C", alpha=.2)
    axes[1].set(xlabel="Frequency (Hz)", ylabel="Magnitude median and IQR (ohm)"); save(fig, "quality-overview")


def cutoff_table():
    rows = []
    for i, task in enumerate(TASKS, 1):
        for cutoff in [20, 40, 60, 80, 100.000001]:
            stream = f"ablation_quality_{cutoff}"
            row = metric(task, stream)
            ci = intervals(task, stream).loc["auc"]
            rows.append([i, "Full" if cutoff > 100 else cutoff,
                         f"{int(row.n_covered)}/{int(row.n_sites)}",
                         rf"{100*row.coverage:.1f}\%", cell(row.auc, ci)])
    table(RESPONSE / "cutoff-performance-v5.tex",
          r"Completed v5 fixed-reference SVM quality contrasts. AUC and coverage use patient-equal site weights after aggregation across five repetitions. "
          r"Parentheses give 95\% whole-patient bootstrap intervals. The strict rule is $r_i<C_{th}$. "
          r"Full retains all sites irrespective of recorded acquisition rejection. These contrasts change the evaluated population and do not establish a clinical cutoff.",
          "tab:response_cutoff_performance", "lcccc", r"Task & $C_{th}$ (\%) & Evaluable/eligible sites & Coverage & AUC", rows, False)
    path = RESPONSE / "cutoff-performance-v5.tex"
    path.write_text(path.read_text().replace(r"\begin{table}[t]", r"\begin{center}\begin{minipage}{.96\linewidth}")
                    .replace(r"\caption{", r"\captionof{table}{")
                    .replace(r"\end{table}", r"\end{minipage}\end{center}"))


def comparison():
    baseline = read(TABLES / "v3-reported-values.json")
    lines = ["# V3–v5 results comparison", "",
             "V5: `results/revision-three-tasks-v5`, with descriptive quality evidence from `results/revision-quality-v5`. "
             "The saved-result verifier passed all 75 outer folds with no unavailable streams. Full input-file verification is incomplete because "
             "four files in `results/revision-source-v3/` are absent locally (clinical metadata, test means, repeat index and extraction manifest). "
             "The 276 local site NPZ files, original MATLAB file and IIVV pattern file match the recorded hashes. No model was trained for this update.", "",
             "V3 raw metric CSVs and its run manifest are no longer present locally. Its values below are the rounded values reported in the "
             "pre-v5 manuscript, tables and native timing figures, captured in [v3-reported-values.json](Revision/revision-evidence/v3-reported-values.json) "
             "with source hashes. V3 is not re-verified here. Deltas use those rounded reported values.", "",
             "Seed, search choices, baseline cutoff and repetition count changed. These are descriptive comparisons, not a paired experiment or evidence that v5 improves on v3. "
             "Intervals are conditional patient-bootstrap intervals. Repeated predictions do not increase the cohort size. "
             "Site metrics exclude abstentions, so different coverage must be considered when comparing estimates.", "",
             "| Item / endpoint | V3 reported | V5 | Change in point estimate | V5 source |",
             "|---|---|---|---:|---|"]

    def plain(value):
        text = str(value).replace(r"\shortstack{", "").replace(r"\\", " ")
        return text.replace(r"\%", "%").replace("}", "").replace("--", "–").strip()

    def add(item, old, new, source, delta=None):
        if delta is None:
            try:
                a, b = float(re.match(r"^-?[\d.]+", plain(old))[0]), float(re.match(r"^-?[\d.]+", plain(new))[0])
                delta = f"{b-a:+.3f}"
            except (TypeError, ValueError):
                delta = "—"
        lines.append(f"| {item} | {plain(old)} | {plain(new)} | {delta} | `{source}` |")

    cfg = read(ROOT / "resolved_config.json")
    protocol = baseline["protocol"]
    for key in ["seed", "outer_folds", "outer_repeats", "inner_folds", "default_cutoff"]:
        add(f"Protocol: {key}", protocol[key], cfg[key], "resolved_config.json")
    add("Outer computations (all tasks)", 15, 75, "final_search_budget.csv")
    for key in ["frequency_counts", "normalizations", "cutoffs"]:
        current = ", ".join("Full" if isinstance(x, (int, float)) and x > 100 else str(x) for x in cfg[key])
        # Frequency counts never exceed 100, so only rejection sentinels are renamed.
        add(f"Search: {key}", protocol[key], current, "resolved_config.json", "—")
    budget = read(ROOT / "final_search_budget.csv")
    failures = [f for t in TASKS for f in read(ROOT / t / "candidate_failures.json")]
    tests = read(ROOT / "corrected_resampled_tests_holm.csv"); unique = tests[tests.alias_of.isna()]
    for key, value, source in [
        ("successful_candidates", int(budget.successful_unique_candidates.sum()), "final_search_budget.csv"),
        ("inner_fits", int(budget.inner_fits.sum()), "final_search_budget.csv"),
        ("candidate_failures", len(failures), "*/candidate_failures.json"),
        ("global_holm_significant", int((unique.p_holm_all_unique_metrics < .05).sum()), "corrected_resampled_tests_holm.csv"),
        ("global_holm_min", f"{unique.p_holm_all_unique_metrics.min():.8g}", "corrected_resampled_tests_holm.csv"),
        ("within_metric_holm_min", f"{unique.p_holm_within_metric.min():.8g}", "corrected_resampled_tests_holm.csv"),
    ]:
        add(key, protocol[key], value, source, "—" if "holm_min" in key else None)
    add("Elapsed analysis time (minutes; v3 approximate)", protocol["elapsed_minutes_approx"],
        f"{manifest['elapsed_seconds']/60:.2f}", "run_manifest.json", "—")
    headers = {
        "primary-table.tex": ["Task", "AUC", "Accuracy", "F1", "PPV/precision", "SN", "SP", "NPV", "Coverage"],
        "sensitivity-table.tex": ["Task", "AUC", "Accuracy", "F1", "PPV/precision", "SN", "SP", "NPV", "Coverage"],
        "class-table.tex": ["Task", "Class", "Sites", "TP", "FN", "AUC", "OvR accuracy", "F1", "SN", "SP", "PPV", "NPV"],
        "model-table.tex": ["Task", "Model", "All/all AUC", "AUC-tuned AUC", "All/all sites", "Tuned sites"],
        "cohort-table.tex": ["Group", "Patients", "Sites", "Retained at 60%", "Retained at 80%"],
    }
    for filename, labels in headers.items():
        current = []
        for line in (TABLES / filename).read_text().splitlines():
            if re.match(r"^[123] ", line) or (filename == "cohort-table.tex" and re.match(r"^(Clinically|Invasive|High-grade|Lower-grade|Unmapped)", line)):
                parts = [v.strip() for v in line.split("&")]
                parts[-1] = re.sub(r"\s+\\\\\s*$", "", parts[-1]); current.append(parts)
        old = baseline["tables"][filename]
        assert len(current) == len(old), filename
        identity = 2 if filename in ["class-table.tex", "model-table.tex"] else 1
        for before, after in zip(old, current):
            assert before[:identity] == after[:identity], (filename, before, after)
            assert len(after) == len(labels)
            for j in range(identity, len(labels)):
                endpoint = {"primary-table.tex": "AUC-selected, patient-equal sites", "sensitivity-table.tex": "Sensitivity-selected, patient-equal sites",
                            "class-table.tex": "AUC-selected, unweighted class", "model-table.tex": "Patient-equal sites", "cohort-table.tex": "Source cohort"}[filename]
                add(f"{endpoint}: {' / '.join(after[:identity])}: {labels[j]}", before[j], after[j], f"revision-evidence/{filename}",
                    "—" if labels[j] in ["Coverage", "All/all sites", "Tuned sites"] else None)
    reported = baseline["reported"]
    for key, stream, unit, field in [
        ("unweighted_auc", "auc_selected", "site", "auc"), ("patient_auc", "auc_selected", "patient", "auc"),
        ("brier", "auc_selected", "patient_weighted_site", "brier"),
        ("missingness_auc", "ablation_missingness", "patient_weighted_site", "auc"),
        ("pca_auc", "ablation_selector_pca", "patient_weighted_site", "auc"),
        ("mi_auc", "ablation_selector_mutual_info", "patient_weighted_site", "auc"),
        ("magnitude_auc", "ablation_representation_magnitude", "patient_weighted_site", "auc"),
        ("magnitude_phase_auc", "ablation_representation_magnitude_phase", "patient_weighted_site", "auc"),
        ("covariate_only_auc", "covariate_only", "patient_weighted_site", "auc"),
        ("covariate_eis_only_auc", "covariate_eis_only", "patient_weighted_site", "auc"),
        ("covariate_combined_auc", "covariate_combined", "patient_weighted_site", "auc"),
    ]:
        for i, task in enumerate(TASKS):
            add(f"Task {i+1}: {key}", f"{reported[key][i]:.3f}", f"{metric(task, stream, unit)[field]:.3f}", f"{task}/metrics_summary.csv")
    for key, stream, unit in [("hgd_weighted_sensitivity", "auc_selected", "patient_weighted_site"),
                               ("hgd_site_sensitivity", "auc_selected", "site"),
                               ("hgd_sensitivity_selected_weighted", "sensitivity_selected", "patient_weighted_site")]:
        pc = read(ROOT / TASKS[2] / "per_class_metrics.csv")
        pc = pc[(pc.stream == stream) & (pc.unit == unit) & (pc.class_name == "High-grade dysplasia")].iloc[0]
        ci = read(ROOT / TASKS[2] / f"nested_cv_{stream}_{unit}_per_class_confidence_intervals.csv")
        ci = ci[(ci.class_index == 2) & (ci.metric == "sensitivity")].iloc[0]
        before = reported[key]
        add(key, f"{before[0]:.3f} ({before[1]:.3f}–{before[2]:.3f})",
            f"{pc.sensitivity:.3f} ({ci.ci_low:.3f}–{ci.ci_high:.3f})", f"{TASKS[2]}/per_class_metrics.csv")
    for key, stream in [("task3_auc_cm", "auc_selected"), ("task2_sensitivity_cm", "sensitivity_selected")]:
        task = TASKS[2] if key.startswith("task3") else TASKS[1]
        cm = read(ROOT / task / f"nested_cv_{stream}_site_confusion_counts.csv").iloc[:, 1:].to_numpy(int)
        for i, row in enumerate(reported[key]):
            for j, value in enumerate(row):
                add(f"{key}: reference class {i} → predicted class {j}", value, cm[i, j], f"{task}/nested_cv_{stream}_site_confusion_counts.csv")
    selections = read(TABLES / "selected-configurations-v5.csv")
    fcounts = selections.frequencies_hz.map(lambda x: len(json.loads(x)))
    pairs = fcounts * selections.n_iivv
    for key, value in [("feature_scalar_min", selections.input_features.min()), ("feature_scalar_max", selections.input_features.max()),
                       ("iivv_min", selections.n_iivv.min()), ("iivv_max", selections.n_iivv.max()),
                       ("pairs_min", pairs.min()), ("pairs_max", pairs.max()), ("all_frequencies_refits", (fcounts == 31).sum()),
                       ("task1_100hz_refits", selections[selections.task == TASKS[0]].frequencies_hz.map(lambda x: json.loads(x) == [100]).sum())]:
        add(key, reported[key], int(value), "selected-configurations-v5.csv")
    for key, stream in [("task2_auc_prevalence_ppv_npv", "auc_selected"), ("task2_sensitivity_prevalence_ppv_npv", "sensitivity_selected")]:
        frame = read(ROOT / TASKS[1] / f"nested_cv_{stream}_prevalence_projections.csv").query("class_index == 1")
        for i, row in enumerate(frame.itertuples()):
            for j, name in enumerate(["ppv", "npv"]):
                add(f"Task 2 {stream}: {name.upper()} at prevalence {row.assumed_prevalence:g}",
                    f"{reported[key][i][j]:.3f}", f"{getattr(row, name):.3f}", f"{TASKS[1]}/nested_cv_{stream}_prevalence_projections.csv")
    icc = read(QUALITY / "repeatability/icc_and_inter_site_variability.csv")
    assert len(baseline["icc_groups"]) == 15
    for i, row in enumerate(icc.itertuples()):
        low, high, estimate = baseline["icc_groups"][3*i:3*i+3]
        add(f"Processed-test ICC: {row.class_name}", f"{estimate:.3f} ({low:.3f}–{high:.3f})",
            f"{row.icc_2_1_absolute:.3f} ({row.icc_ci_low:.3f}–{row.icc_ci_high:.3f})",
            "revision-quality-v5/repeatability/icc_and_inter_site_variability.csv")
    variances = iter(baseline["pca_axis_variance_percent"])
    for kind in ["pca", "mi-pca"]:
        for i, task in enumerate(TASKS, 1):
            meta = read(QUALITY / "exploration" / f"{task}-{kind}.json")
            for k, value in enumerate(meta["explained_variance_ratio"], 1):
                add(f"Task {i}: descriptive {kind}, PC{k} variance (%)", f"{next(variances):.1f}",
                    f"{100*value:.1f}", f"revision-quality-v5/exploration/{task}-{kind}.json")
    for i, task in enumerate(TASKS):
        pop = read(ROOT / task / "cv_population_summary.csv").query("population == 'unfiltered'")
        for part, before in [("outer_train", ["77–81", "78–82", "76–79"][i]),
                             ("outer_test", ["17–21", "18–22", "18–21"][i])]:
            values = pop.loc[pop.partition == part, "n_patients"]
            add(f"Task {i+1}: {part} patient range", before, f"{values.min()}–{values.max()}", f"{task}/cv_population_summary.csv", "—")
        selected = selections[selections.task == task]
        old_models = [["RF", "RF", "RF", "RF", "RF"], ["RF", "SVM", "RF", "RF", "SVM"],
                      ["RF", "LR", "LR", "SVM", "RF"]][i]
        for model in ["svm", "rf", "lr"]:
            add(f"Task {i+1}: AUC-selected {model.upper()} refits", old_models.count(model.upper()),
                int((selected.model == model).sum()), "selected-configurations-v5.csv")
    for filename, columns in [("timing-fit.tikz.tex", ["model_fit_seconds"]),
                              ("timing-inference.tikz.tex", ["test_transform_seconds", "test_predict_seconds"])]:
        old = iter(baseline[filename])
        for col in columns:
            for i, task in enumerate(TASKS, 1):
                data = read(ROOT / task / "measured_computation_times.csv")
                for model in ["svm", "rf", "lr"]:
                    for stream in [f"baseline_{model}", f"auc_{model}"]:
                        span, med = next(old), next(old)
                        values = data.loc[data.stream == stream, col]
                        add(f"Task {i} {stream}: {col}, min / median / max (s)",
                            " / ".join(f"{x:.6g}" for x in [span[0], med[0], span[1]]),
                            " / ".join(f"{x:.6g}" for x in [values.min(), values.median(), values.max()]), f"{task}/measured_computation_times.csv", "—")
    lines += ["", "All 75 exact AUC-selected configurations, including frequency and IIVV identities, are in "
              "[selected-configurations-v5.csv](Revision/revision-evidence/selected-configurations-v5.csv). "
              "The v3 and v5 folds have different seeds and must not be paired by fold number.", "",
              "The original reviewer comments and text inside `\\edits{}` retain historical numbers deliberately. "
              "Current accepted manuscript results, tables, native figure data, response answers, quotations and quality PNGs use v5."]
    Path("manuscript/v3-v5-comparison.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    manifest = read(ROOT / "run_manifest.json")
    assert manifest["status"] == "completed" and not manifest["synthetic"]
    cfg = manifest["config"]
    assert cfg["tasks"] == TASKS and (cfg["outer_repeats"], cfg["outer_folds"], cfg["inner_folds"]) == (5, 5, 4)
    METRICS = {t: read(ROOT / t / "metrics_summary.csv") for t in TASKS}
    diagnostic_tables(); selected_tables(); figure_data(); timing_figures(); quality_pngs(); cutoff_table(); comparison()
    (TABLES / "v5-evidence-provenance.json").write_text(json.dumps({
        "results": str(ROOT), "quality_results": str(QUALITY),
        "scope": "Exports from frozen predictions and descriptive coordinates only. No training or PDF compilation.",
        "source_sha256": SOURCES,
    }, indent=2) + "\n")
    print("Updated v5 tables, all 75 configurations, TikZ data and 15 response quality PNGs. No training or PDF compilation.")
