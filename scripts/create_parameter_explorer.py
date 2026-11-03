"""Build an offline interactive HTML view from saved CSVs; no model imports."""

import argparse
import csv
import json
import math
from pathlib import Path
import re


def read_rows(path):
    if not path.is_file() or not path.stat().st_size:
        return []
    with path.open() as stream:
        return list(csv.DictReader(stream))


def point(candidate, row, task, mode):
    params = json.loads(candidate.get("params_json", "{}"))
    result = dict(candidate, **{f"param_{k}": v for k, v in params.items()})
    result.update(
        task=task,
        mode=mode,
        fold=row.get("outer_fold", row.get("fold")),
        repeat=row["repeat"],
        stream=row.get("stream", candidate.get("stage")),
        unit=row.get("unit", "patient_weighted_site"),
        status=(
            "MC: 4/5 folds, exploratory"
            if task == "MC"
            else "adaptive follow-up analysis"
        ),
    )
    for key in [
        "auc",
        "accuracy",
        "f1",
        "sensitivity",
        "specificity",
        "precision",
        "npv",
        "coverage",
        "objective_auc",
        "objective_acc",
        "objective_f1",
        "objective_sensitivity",
    ]:
        value = row.get(key)
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = math.nan
        result[key] = number if math.isfinite(number) else None
    result["C"] = params.get("C")
    result["params"] = params
    return result


def load_points(root):
    root = Path(root)
    manifest = json.loads((root / "run_manifest.json").read_text())
    if manifest["status"] not in {"completed", "completed_partial"}:
        raise ValueError("Expected a finished search run (partial allowed)")
    config = manifest["config"]
    if not (config.get("fine_tuning") or config.get("final_selection")):
        raise ValueError("Expected a finished fine-tuning or final-selection run")
    tasks = manifest["config"]["tasks"]
    points = []
    for task in tasks:
        for row in read_rows(root / task / "inner_search.csv"):
            if row["evaluation"] == "nested_cv":
                points.append(point(row, row, task, "inner"))
        path = root / task / "candidate_failures.json"
        for row in json.loads(path.read_text()) if path.is_file() else []:
            match = re.fullmatch(r"nested_cv_repeat(\d+)_fold(\d+)", row["fold"])
            if match:
                value = point(
                    row["candidate"],
                    dict(repeat=match[1], outer_fold=match[2]),
                    task,
                    "failed",
                )
                value["reason"] = row["reason"]
                points.append(value)
    metrics = {}
    for task in tasks:
        for row in read_rows(root / task / "outer_fold_metrics.csv"):
            if row["evaluation"] != "nested_cv":
                continue
            key = (task, row["repeat"], row["outer_fold"], row["stream"], row["unit"])
            if key in metrics:
                raise ValueError(f"Duplicate outer metric identity: {key}")
            metrics[key] = row
    for row in read_rows(root / "review_selected_parameters.csv"):
        for unit in ["site", "patient_weighted_site", "patient"]:
            key = (row["task"], row["repeat"], row["fold"], row["stream"], unit)
            if key not in metrics:
                continue  # Never attach another fold's/model's outer result.
            value = point(
                json.loads(row["candidate"]), metrics[key], row["task"], "outer"
            )
            value.update(
                features=row.get("features"),
                fit_seconds=row.get("fit_seconds"),
                frequencies_hz=row.get("frequencies_hz"),
            )
            points.append(value)
    counts = (
        {task: len(config["fine_tuning"]["folds"][task]) for task in tasks}
        if config.get("fine_tuning")
        else {
            task: sum(
                row["evaluation"] == "nested_cv"
                for row in json.loads((root / task / "splits.json").read_text())
            )
            for task in tasks
        }
    )
    expected = config["outer_folds"] * config.get("outer_repeats", 1)
    for value in points:
        value["status"] = (
            f"EXPLORATORY: {counts[value['task']]}/{expected} folds"
            if counts[value["task"]] < expected
            else "complete original partition; adaptive follow-up analysis"
        )
    return points, tasks


def build_explorer(results_dir, output=None):
    root = Path(results_dir)
    points, tasks = load_points(root)
    template = Path(__file__).with_name("parameter_explorer.html").read_text()
    payload = json.dumps(
        dict(rows=points, tasks=tasks), ensure_ascii=False, allow_nan=False
    ).replace("<", "\\u003c")
    destination = Path(output) if output else root / "parameter-explorer.html"
    destination.write_text(template.replace("/*DATA*/", payload))
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/revision-finetuning-20261102-122955"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(build_explorer(args.results_dir, args.output))
