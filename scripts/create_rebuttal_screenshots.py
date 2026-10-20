#!/usr/bin/env python3
"""Render supplementary JPG evidence from the compiled marked manuscript only."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
THEMES = {
    "clinical": ("sub:Clinical_data", "Source-verified population and clinical data"),
    "protocol": ("sub:revision_protocol", "Revised patient-nested evaluation protocol"),
    "preprocessing": (
        "sub:revision_preprocessing",
        "Patient separation, train-fitted preprocessing and abstentions",
    ),
    "features": (
        "sub:revision_features",
        "Supervised selectors, complex representations and frozen geometry",
    ),
    "metrics": (
        "sub:revision_metrics",
        "Clinical objectives, calibration and class-specific metrics",
    ),
    "uncertainty": (
        "sub:revision_uncertainty",
        "Patient-cluster uncertainty, available confounding and test-mean repeatability",
    ),
    "reproducibility": (
        "sub:revision_reproducibility",
        "Source verification and reproducible execution",
    ),
}
START = "% BEGIN GENERATED MANUSCRIPT SCREENSHOTS"
END = "% END GENERATED MANUSCRIPT SCREENSHOTS"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Use current PDF, only if newer than the marked TeX source",
    )
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    tex = ROOT / "manuscript/Paper/Revision-1.tex"
    pdf = tex.with_suffix(".pdf")
    aux = tex.with_suffix(".aux")
    if not args.no_build:
        subprocess.run(
            ["make", "revision"], cwd=ROOT, check=True, stdout=subprocess.DEVNULL
        )
    if not pdf.is_file() or pdf.stat().st_mtime < tex.stat().st_mtime:
        raise RuntimeError(
            "Compile Revision-1.tex before creating evidence; stale PDF rejected"
        )
    labels = {
        m[1]: int(m[2])
        for m in re.finditer(
            r"\\newlabel\{([^}]+)\}\{\{[^}]*\}\{(\d+)\}", aux.read_text()
        )
    }
    labels["__first_page__"] = 1
    themes = {
        "abstract": ("__first_page__", "Marked abstract and analyzed cohort"),
        **THEMES,
    }
    for theme, label, title in [
        (
            "quality_results",
            "sub:revision_data_results",
            "Completed source/quality/repeatability audit results",
        ),
        (
            "quality_table",
            "tab:revision_quality",
            "Class quality and test-mean ICC estimates",
        ),
        (
            "nested_results",
            "sub:revision_results",
            "Completed patient-nested classification results",
        ),
        (
            "nested_table",
            "tab:revision_nested",
            "Nested AUC intervals and class-balanced metrics",
        ),
        (
            "referral_results",
            "tab:revision_referral",
            "Sensitivity-selected clinical operating points",
        ),
        (
            "hardware_results",
            "tab:revision_hardware",
            "Actual selected coordinates and hardware requirements",
        ),
        (
            "confusion_results",
            "fig:revision_CM",
            "Regenerated four-class confusion matrices",
        ),
        ("roc_results", "fig:revision_ROC", "Regenerated binary ROC curves"),
        (
            "clinical_curves",
            "fig:revision_clinical_curves",
            "Calibration and clinical decision curves",
        ),
    ]:
        if label in labels:
            themes[theme] = (label, title)
    pages = {}
    for theme, (label, title) in themes.items():
        if label not in labels:
            raise RuntimeError(f"Unresolved manuscript label {label}")
        pages.setdefault(labels[label], []).append(
            dict(theme=theme, label=label, title=title)
        )
    folder = ROOT / "manuscript/Response_Letter/figures_rebuttal"
    folder.mkdir(parents=True, exist_ok=True)
    evidence = []
    manifest = []
    for page, items in sorted(pages.items()):
        stem = folder / f"revision-page-{page:02d}"
        subprocess.run(
            [
                "pdftoppm",
                "-f",
                str(page),
                "-l",
                str(page),
                "-r",
                str(args.dpi),
                "-jpeg",
                "-singlefile",
                str(pdf),
                str(stem),
            ],
            check=True,
        )
        path = stem.with_suffix(".jpg")
        with Image.open(path) as im:
            if im.format != "JPEG":
                raise RuntimeError("Evidence must be JPEG")
            dimensions = list(im.size)
        title = "; ".join(v["title"] for v in items)
        evidence.append(
            "\\begin{figure}[p]\n\\centering\n"
            + "\\includegraphics[width=\\linewidth,height=0.80\\textheight,keepaspectratio]{figures_rebuttal/"
            + path.name
            + "}\n"
            + "\\caption{Marked revised manuscript, verified page "
            + str(page)
            + ". "
            + title
            + ".}\n"
            + "\n".join("\\label{evidence:" + v["theme"] + "}" for v in items)
            + "\n\\end{figure}\n\\clearpage"
        )
        manifest.append(
            dict(
                file=path.name,
                pdf_page_1based=page,
                dimensions_pixels=dimensions,
                anchors=items,
                jpg_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    info = dict(
        source_pdf=str(pdf.relative_to(ROOT)),
        source_pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
        source_tex_sha256=hashlib.sha256(tex.read_bytes()).hexdigest(),
        dpi=args.dpi,
        method="pdftoppm JPEG rendering; full compiled pages; no generated or reconstructed manuscript imagery",
        screenshots=manifest,
    )
    (folder / "screenshot-manifest.json").write_text(json.dumps(info, indent=2) + "\n")
    letter = ROOT / "manuscript/Response_Letter/resposeletter.tex"
    s = letter.read_text()
    appendix = (
        START
        + "\n\\clearpage\n\\appendix\n\\section{Marked manuscript excerpts}\n"
        + r"These JPGs are supplementary screenshots of the compiled marked revision. The response text provides precise section and label locations; screenshots do not replace those descriptions."
        + "\n\n"
        + "\n\n".join(evidence)
        + "\n"
        + END
    )
    if START in s:
        s = re.sub(
            re.escape(START) + ".*?" + re.escape(END), lambda _: appendix, s, flags=re.S
        )
    else:
        assert s.count(r"\end{document}") == 1
        s = s.replace(r"\end{document}", appendix + "\n\n" + r"\end{document}")
    letter.write_text(s)
    print(
        f"Rendered {len(pages)} JPG pages for {len(themes)} manuscript anchors; updated response appendix."
    )


if __name__ == "__main__":
    main()
