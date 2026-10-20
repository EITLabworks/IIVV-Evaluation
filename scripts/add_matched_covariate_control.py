#!/usr/bin/env python3
"""Complete matched covariate controls/reports after a previously completed run."""

import argparse
from pathlib import Path
import sys, json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from revision.data import load_dataset, digest
from revision.covariate_control import add_matched_control
from revision.analysis import report_task
from revision.pipeline import write_json
from revision.metrics import holm


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="results/revision-clinical-final")
    a = p.parse_args()
    root = Path(a.results)
    manifest = json.loads((root / "run_manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("Finish the original run before completing its reports")
    args = manifest["arguments"]
    config = manifest["config"]
    dataset = load_dataset(
        args["data"],
        patterns_path=args["patterns"],
        metadata_path=args.get("metadata"),
        label_map_path=args.get("label_map"),
        frequencies=config.get("frequencies_hz"),
        trust_object_arrays=args.get("trust_object_arrays", False),
    )
    for name, value in dataset.input_hashes.items():
        if manifest["input_hashes"].get(name) != value:
            raise ValueError(f"Input changed: {name}")
    import shutil

    source_hashes = {}
    for path in [*sorted(Path("revision").glob("*.py")), Path(__file__)]:
        name = str(path.relative_to(Path.cwd())) if path.is_absolute() else str(path)
        source_hashes[name] = digest(path)
        target = root / "source-at-matched-control" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    summaries = []
    comparisons = []
    for task in config["tasks"]:
        folder = root / task
        predictions = pd.read_csv(
            folder / "outer_predictions.csv",
            dtype={"patient_id": str, "sample_id": str},
            float_precision="round_trip",
        )
        selections = json.loads((folder / "selections.json").read_text())
        predictions, selections = add_matched_control(
            dataset, task, config, root, predictions, selections
        )
        summary, comparison = report_task(
            dataset, task, predictions, selections, config, root
        )
        summaries.append(summary)
        comparisons.append(comparison)
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(root / "all_metrics.csv", index=False)
    summary[summary.stream.str.startswith("baseline_")].to_csv(
        root / "BaselineResults.csv", index=False
    )
    summary[~summary.stream.str.startswith("baseline_")].to_csv(
        root / "FinalResults.csv", index=False
    )
    paired = pd.concat(comparisons, ignore_index=True)
    if len(paired):
        paired["p_holm_all_tasks_models_objectives"] = holm(paired.p_value)
    paired.to_csv(root / "paired_comparisons_holm.csv", index=False)
    write_json(
        root / "matched-control-postprocessing.json",
        dict(
            status="completed",
            script_sha256=digest(__file__),
            module_sha256=digest("revision/covariate_control.py"),
            source_hashes=source_hashes,
            input_hashes=dataset.input_hashes,
            scope="Training-only matched LR counterpart; original primary predictions/selections unchanged",
        ),
    )
    manifest["matched_covariate_control_completed"] = True
    write_json(root / "run_manifest.json", manifest)
    print("Matched covariate controls and complete reports finished")


if __name__ == "__main__":
    main()
