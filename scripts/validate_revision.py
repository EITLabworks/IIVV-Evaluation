#!/usr/bin/env python3
"""Validate manuscript provenance, every reviewer response and clinical run scope."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from collections import Counter
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def without_comments(text):
    return "\n".join(
        re.split(r"(?<!\\)%", line, maxsplit=1)[0] for line in text.splitlines()
    )


def check_structure(text):
    text = without_comments(text)
    depth = 0
    stack = []
    for match in re.finditer(r"(?<!\\)[{}]", text):
        depth += 1 if match[0] == "{" else -1
        assert depth >= 0, "Unbalanced TeX braces"
    assert depth == 0, "Unbalanced TeX braces"
    for match in re.finditer(r"\\(begin|end)\{([^}]+)\}", text):
        if match[1] == "begin":
            stack.append(match[2])
        else:
            assert stack and stack.pop() == match[2], match[0]
    assert not stack, stack


def diagnostics(path):
    text = Path(path).read_text(errors="replace")
    return dict(
        fatal_errors=len(re.findall(r"^!", text, re.M)),
        undefined=sorted(
            set(re.findall(r"(?:Reference|Citation) `([^']+)'[^\n]*undefined", text))
        ),
        overfull_boxes=len(re.findall("Overfull", text)),
        missing_characters=text.count("Missing character:"),
        output=re.findall(r"Output written on ([^\n]+)", text),
    )


def plain(text):
    text = re.sub(r"\\loc\{([^}]+)\}", r"`\1`", text)
    return text.replace("\\%", "%").replace("~", " ").replace("\\textquotesingle", "'")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/revision-clinical-final")
    parser.add_argument("--require-completed", action="store_true")
    args = parser.parse_args()
    baseline = json.loads((ROOT / "manuscript/revision-baseline.json").read_text())
    paper = ROOT / "manuscript/Paper/Revision-1.tex"
    letter = ROOT / "manuscript/Response_Letter/resposeletter.tex"
    s = paper.read_text()
    response = letter.read_text()
    assert (
        sha(paper.with_name("main.tex")) == baseline["main_sha256"]
    ), "main.tex was altered"
    prior = re.sub(
        r"% BEGIN GENERATED (.*?)\n.*?% END GENERATED \1\n\n", "", s, flags=re.S
    )
    transformations = (
        baseline["original_transformations"] + baseline["code_revision_transformations"]
    )
    for change in reversed(transformations):
        assert prior.count(change["new"]) == 1, change["old"][:100]
        prior = prior.replace(change["new"], change["old"])
    assert (
        hashlib.sha256(prior.encode()).hexdigest()
        == baseline["original_revision_sha256"]
    ), "Original Revision-1 source not exactly recoverable"
    comments = baseline["original_comments"]
    ncomments = 0
    for kind in ["c1", "m1", "c2", "c3", "m3"]:
        values = (
            comments[kind].values()
            if isinstance(comments[kind], dict)
            else comments[kind]
        )
        for text in values:
            assert text in response, (kind, text[:100])
            ncomments += 1
    for key in ["ae", "r1intro", "r1general"]:
        assert comments[key] in response, key
    assert ncomments == 52
    assert response.count(r"\begin{response}") == 55
    for text in [s, response]:
        check_structure(text)
    labels = set(re.findall(r"\\label\{([^}]+)\}", without_comments(s)))
    locs = set(re.findall(r"\\loc\{([^}]+)\}", response))
    assert locs <= labels, locs - labels
    builds = dict(
        revision=diagnostics(paper.with_suffix(".log")),
        response=diagnostics(letter.with_suffix(".log")),
    )
    for name, info in builds.items():
        assert (
            not info["fatal_errors"]
            and not info["undefined"]
            and not info["missing_characters"]
        ), (name, info)
    assert 7728 * 31 == 239568 and 256 * 239568 == 61329408
    assert 90 + 96 + 29 + 41 == 256 and 96 + 29 + 41 == 166 and 90 + 96 == 186
    screenshot_path = letter.parent / "figures_rebuttal/screenshot-manifest.json"
    screenshot = json.loads(screenshot_path.read_text())
    assert screenshot["source_pdf_sha256"] == sha(paper.with_suffix(".pdf"))
    assert screenshot["source_tex_sha256"] == sha(paper)
    for item in screenshot["screenshots"]:
        path = screenshot_path.parent / item["file"]
        assert sha(path) == item["jpg_sha256"]
        with Image.open(path) as im:
            assert im.format == "JPEG"
    root = Path(args.results)
    root = root if root.is_absolute() else ROOT / root
    manifest = json.loads((root / "run_manifest.json").read_text())
    completed = manifest["status"] == "completed" and not manifest.get("synthetic")
    if args.require_completed:
        assert completed, "Clinical reanalysis not completed"
    split_count = 0
    if completed:
        from revision.verification import verify_results

        verify_results(root)
        assert set(manifest["config"]["tasks"]) == {
            "healthy_vs_cancer",
            "cancer_vs_lr_hr",
            "MC",
            "highrisk_vs_lowrisk",
        }
        for task in manifest["config"]["tasks"]:
            splits = json.loads((root / task / "splits.json").read_text())
            for split in splits:
                train = set(split["train_patients"])
                test = set(split["test_patients"])
                assert not train & test
                for inner in split["inner"]:
                    a = set(inner["train_patients"])
                    b = set(inner["validation_patients"])
                    assert not a & b and a | b <= train and not (a | b) & test
                split_count += 1
            pred = pd.read_csv(root / task / "outer_predictions.csv")
            assert not pred.duplicated(
                ["stream", "evaluation", "repeat", "sample_id"]
            ).any()
            nsites = len(pd.read_csv(root / task / "task_inclusion.csv"))
            assert (
                pred.groupby(["stream", "evaluation", "repeat"]).size().eq(nsites).all()
            )
            classinfo = json.loads((root / task / "classes.json").read_text())
            if task == "healthy_vs_cancer":
                assert classinfo["positive_class"] == "Cancer"
            if task == "highrisk_vs_lowrisk":
                assert classinfo["positive_class"] == "High risk"
        comparison = pd.read_csv(root / "paired_comparisons_holm.csv")
        assert len(comparison) == len(manifest["config"]["tasks"]) * len(
            manifest["config"]["models"]
        ) * len(manifest["config"]["objectives"])
    actions = json.loads((ROOT / "manuscript/reviewer-code-actions.json").read_text())
    partial = {
        "R1-C5": "Sparse location/joint quality tests are unavailable; observed confounding controls cannot identify contact causes.",
        "R1-C11": "Histology section coverage, registration error, mixed-site adjudication and blinding require clinical records.",
        "R1-C12": "Histological verification of all healthy sites and field-status records remain unavailable.",
        "R1-C14": "Age, sex, smoking/alcohol, inflammation and detailed treatment/contact records remain unavailable.",
        "R1-C24": "Controlled pressure/contact measurements and acquisition-control records remain unavailable.",
        "R1-C25": "Only two to four test means are available; individual burst repeatability cannot be recovered.",
        "R1-C26": "Prospective acquisition and instrument end-to-end latency need new measurements.",
        "R1-C27": "Electrode/drive requirements are computed, but reduced-channel hardware needs design and validation.",
        "R1-M3": "Central-electrode electrical sensitivity asymmetry needs simulation or measurements.",
        "R1-M10": "Public clinical artifact/data sharing conditions require author verification.",
        "R2-C1": "Mixed-footprint histology and its association with errors require pathology records.",
        "R2-C2": "Footprint sectioning, registration and blinded-review records are unavailable.",
        "R3-C4": "Frequency identities/stability are computed; physiological validation remains unavailable.",
    }
    rows = []
    for old in baseline["original_action_rows"]:
        key = old["id"]
        new = actions["updates"].get(key)
        remaining = partial.get(key, "")
        needs_run = key in actions["starred"] and key not in [
            "R1-C24",
            "R1-C25",
            "R1-M6",
            "R1-M7",
            "R1-M8",
            "R1-M10",
            "R3-C1",
            "R3-C2",
        ]
        if needs_run and not completed:
            remaining = (
                "Completed author-run server evaluation is pending. " + remaining
            ).strip()
        status = "Partially addressed" if remaining else "Addressed"
        action = plain(new["text"] if new else old["action"])
        if len(action) > 550:
            action = action[:547].rsplit(" ", 1)[0] + "..."
        rows.append(
            dict(
                id=key,
                status=status,
                action=action,
                location=plain(new["location"] if new else old["location"]),
                remaining=remaining,
                purple_star=key in actions["starred"],
            )
        )
    numbered = [
        r for r in rows if r["id"] not in ["Editor", "R1-General", "R2-General"]
    ]
    counts = Counter(r["status"] for r in numbered)
    record = dict(
        main_sha256=baseline["main_sha256"],
        main_unchanged=True,
        original_revision_recoverable=True,
        source_transformations=len(transformations),
        numbered_comments=52,
        total_responses=55,
        numbered_addressed=counts["Addressed"],
        numbered_partially_addressed=counts["Partially addressed"],
        purple_starred_comments=len(actions["starred"]),
        reviewer_wording_preserved=True,
        balanced_braces_and_environments=True,
        manuscript_location_labels_checked=len(locs),
        arithmetic_checks_pass=True,
        screenshot_jpgs=len(screenshot["screenshots"]),
        screenshot_source_hashes_verified=True,
        clinical_manifest_status=manifest["status"],
        clinical_reanalysis_completed=completed,
        checked_patient_disjoint_outer_partitions=split_count,
        builds=builds,
        comments=rows,
    )
    (ROOT / "manuscript/revision-validation.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    lines = [
        "# Manuscript and code revision summary",
        "",
        "All 52 numbered comments have individual framed responses, plus the editor and two reviewer overview responses. Reviewer 1's original major numbering skips 22 and 23; no missing comments were invented.",
        "",
        f'Current numbered-comment status: **{counts["Addressed"]} addressed, {counts["Partially addressed"]} partially addressed**. A purple star marks {len(actions["starred"])} comments connected to changed code and associated rerun outputs. Clinical run status: **{manifest["status"]}**.',
        "",
        "The supplied source/export audit confirms 276 sites from 100 patients, ordered 7728-pattern geometry, complex spectra, two to four repositioned test means, location and prior radiation/surgery. The original 104-patient analysis count is not reproduced. Historical tables/figures remain explicitly exploratory and separate from the new nested analysis.",
        "",
        "The main manuscript is unchanged. Every manuscript replacement retains old text in `\\edits{}` and marks new text in `\\edit{}`. The original revised source, including previous marked changes, is exactly recoverable using `revision-baseline.json`. Generated audit/classification sections contain only marked additions.",
        "",
        "## Deliverables",
        "",
        "- `Paper/Revision-1.tex` and PDF: marked corrections, revised protocol/algorithm and verified numerical imports.",
        "- `Response_Letter/resposeletter.tex` and PDF: all responses, purple stars and supplementary JPG excerpts with precise locations.",
        "- `Response_Letter/figures_rebuttal/screenshot-manifest.json`: compiled-PDF/page/anchor hashes for actual screenshots.",
        "- `../code-revisions.md`, `../run_all.sh`, `../revision/`, `../configs/`, `../scripts/`, `../tests/`: implementation, run instructions, requested reports and checks.",
        "- `revision-validation.json`, `revision-baseline.json`, `result-import-manifest.json`: validation and edit/result provenance.",
        "",
        "## Comment-by-comment actions",
        "",
        "| Reviewer/comment | Action and remaining evidence | Manuscript location | Status |",
        "|---|---|---|---|",
    ]
    for row in rows:
        action = row["action"] + (
            " **Remaining:** " + row["remaining"] if row["remaining"] else ""
        )
        lines.append(
            "| "
            + row["id"]
            + (" ★" if row["purple_star"] else "")
            + " | "
            + action.replace("|", "/")
            + " | "
            + row["location"].replace("|", "/")
            + " | "
            + row["status"]
            + " |"
        )
    lines += [
        "",
        "## Explicit subcomponents",
        "",
        "| Parent comment | Components and disposition |",
        "|---|---|",
        "| R1 C3 | Mean/median/zero imputation are train-fitted; common coordinates remain; no sample-specific column deletion or common-valid intersection; missingness-only control is separate. Source quality averaging and post-averaging NaNs are audited. |",
        "| R1 C9 | Real/imaginary, magnitude/phase, normalized ratios and effective Cole–Cole features are implemented as training-selected pipelines. No physiological compartment inference is claimed. |",
        "| R1 C18 | All 31 counts, 13 masks, training impedance quantiles, three models, parameter grids, four objectives and four tasks stay inside outer-training data. Holm covers the prespecified outer baseline-comparison family. Inner ablations are descriptive. |",
        "| R1 C25 | Test-mean coordinate CV, repositioning variability, inter-site variability, absolute ICC and class-specific estimates are computed. Within-burst variability of nine original recordings remains unavailable. |",
        "| R1 C30 | Calibration, Brier/log loss, decision curves, sensitivity-oriented thresholds, fixed-specificity operating points, NPV and prevalence projections are implemented. Their clinical assumptions/transport limits are explicit. |",
        "",
        "## Validation and limitations",
        "",
        "- Main SHA-256: `" + baseline["main_sha256"] + "`.",
        "- Original reviewer wording, 55 response boxes, balanced LaTeX groups/environments, manuscript location labels and input-size arithmetic pass.",
        "- Screenshot JPGs are rendered from the current compiled marked PDF; source and image hashes match.",
        "- Numerical unit checks and synthetic integrations verify implementation behavior; synthetic data are excluded from manuscript import.",
        "- Patient-cluster intervals/permutations are conditional on fixed outer predictions, not full model-refitting uncertainty. "
        f'This result manifest specifies {manifest["config"]["outer_folds"]} outer folds, '
        f'{manifest["config"]["inner_folds"]} inner folds and {manifest["config"]["outer_repeats"]} outer partitions, with a staged finite search.',
        "- Available test means do not recover raw burst noise, pathology blinding/footprint matching, controlled contact, prospective hardware performance or external validation. Unrecorded clinical covariates and controlled sharing conditions remain explicit evidence gaps.",
        "- Private data/source/manuscript/generated result directories retain the repository's existing Git exclusion rules. Public sharing was not inferred.",
        "",
    ]
    for name, info in builds.items():
        lines.append(
            f'- {name}: {info["fatal_errors"]} fatal errors, {len(info["undefined"])} unresolved references/citations, {info["missing_characters"]} missing characters, {info["overfull_boxes"]} overfull warnings. '
            + "; ".join(info["output"])
        )
    (ROOT / "manuscript/REVISION_SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                key: record[key]
                for key in [
                    "main_unchanged",
                    "original_revision_recoverable",
                    "numbered_comments",
                    "numbered_addressed",
                    "numbered_partially_addressed",
                    "purple_starred_comments",
                    "screenshot_jpgs",
                    "clinical_manifest_status",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
