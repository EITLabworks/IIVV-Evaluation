"""Fold-specific training-winner neighbourhoods; never rank outer-test results."""

from dataclasses import replace
import csv
import json
import math
from pathlib import Path

from .features import Candidate
from .pipeline import write_json


def fold_id(task, repeat, fold, evaluation="nested_cv"):
    return f"{task}/{evaluation}_repeat{repeat}_fold{fold}"


def neighbours(candidate, config):
    settings = config["fine_tuning"]
    params = json.loads(candidate.params_json)
    variants = [params]
    if candidate.model in {"svm", "lr"}:
        bounds = config["model_grids"][candidate.model]["C"]
        variants = [
            dict(
                params,
                C=min(max(float(params["C"]) * factor, min(bounds)), max(bounds)),
            )
            for factor in settings["c_factors"]
        ]
    if candidate.model == "svm":
        variants = [dict(p, max_iter=settings["svm_max_iter"]) for p in variants]
    return list(
        dict.fromkeys(
            replace(candidate, params_json=json.dumps(p, sort_keys=True))
            for p in variants
        )
    )


def prepare(config, input_hashes, output):
    from .parallel import _check_complete

    source = Path(config["fine_tuning"]["source"])
    settings = config["fine_tuning"]
    if (
        not settings["c_factors"]
        or 1.0 not in settings["c_factors"]
        or any(not math.isfinite(x) or x <= 0 for x in settings["c_factors"])
        or not isinstance(settings["svm_max_iter"], int)
        or settings["svm_max_iter"] < 1
    ):
        raise ValueError("Invalid C factors or SVM iteration limit")
    manifest = json.loads((source / "run_manifest.json").read_text())
    if manifest.get("synthetic") or manifest.get("input_hashes") != input_hashes:
        raise ValueError("Fine tuning requires identical clinical input hashes")
    execution = {
        "task_workers",
        "fold_workers",
        "rf_jobs",
        "blas_threads",
        "feature_cache_bytes",
        "verbose_candidates",
        "save_models",
        "plots",
    }
    for key, value in manifest["config"].items():
        if key not in execution and config.get(key) != value:
            raise ValueError(f"Fine tuning changed original scientific setting: {key}")
    if config["outer_repeats"] != 1 or config.get("external_cohort_column"):
        raise ValueError(
            "This shortlist supports the original single nested partition only"
        )
    plan = {}
    for task in config["tasks"]:
        folds = config["fine_tuning"]["folds"][task]
        if len(set(folds)) != len(folds) or len(folds) < 2:
            raise ValueError(f"At least two unique folds required: {task}")
        for fold in folds:
            if not 0 <= fold < config["outer_folds"]:
                raise ValueError(f"Unknown fold: {task}/{fold}")
            identity = fold_id(task, 0, fold)
            folder = source / "fold-checkpoints" / identity
            if not _check_complete(folder, task, (0, fold, "nested_cv")):
                raise ValueError(f"Missing source checkpoint: {folder}")
            rows = json.loads((folder / "selections.json").read_text())
            splits = json.loads((folder / "splits.json").read_text())
            if len(splits) != 1:
                raise ValueError(f"Expected exactly one source split: {folder}")
            winners = {}
            for objective in config["objectives"]:
                candidates = []
                for model in config["models"]:
                    matches = [
                        row for row in rows if row["stream"] == f"{objective}_{model}"
                    ]
                    if len(matches) != 1 or matches[0]["candidate"]["model"] != model:
                        raise ValueError(
                            f"Missing/ambiguous training winner: {identity}/{objective}/{model}"
                        )
                    row = matches[0]
                    if (
                        row["task"],
                        row["repeat"],
                        row["outer_fold"],
                        row["evaluation"],
                    ) != (task, 0, fold, "nested_cv"):
                        raise ValueError(
                            f"Incorrect source selection identity: {identity}"
                        )
                    candidates.extend(
                        c.dictionary()
                        for c in neighbours(Candidate(**row["candidate"]), config)
                    )
                winners[objective] = candidates
            unique = {
                json.dumps(c, sort_keys=True)
                for choices in winners.values()
                for c in choices
            }
            with (folder / "inner_search.csv").open() as stream:
                original_count = sum(1 for _ in csv.DictReader(stream))
            plan[identity] = dict(
                split=splits[0],
                candidates=winners,
                source_successful_candidates=original_count,
                unique_candidates_including_baselines=len(unique)
                + len(config["models"]),
                source_checkpoint=json.loads((folder / "complete.json").read_text()),
            )
    path = Path(output) / "fine_tuning_plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("Frozen fine-tuning plan changed; use a new output directory")
    write_json(path, plan)
    return plan


def validate_split(entry, current):
    old = entry["split"]
    for field in (
        "train_sample_ids",
        "test_sample_ids",
        "train_patients",
        "test_patients",
    ):
        if set(old[field]) != set(current[field]):
            raise ValueError(f"Fine-tuning source/current split differ: {field}")
    if old["inner"] != current["inner"]:
        raise ValueError("Fine-tuning source/current inner partitions differ")


def annotate(output, config, plan):
    import pandas as pd

    output = Path(output)
    counts = {
        task: len(config["fine_tuning"]["folds"][task]) for task in config["tasks"]
    }
    expected = config["outer_folds"]
    for task in config["tasks"]:
        write_json(
            output / task / "fine_tuning_status.json",
            dict(
                folds_available=counts[task],
                folds_expected=expected,
                status=(
                    "complete_partition"
                    if counts[task] == expected
                    else "PARTIAL_EXPLORATORY"
                ),
                note="Reuses the original partition; not independent confirmatory validation",
            ),
        )
    for name in (
        "all_metrics.csv",
        "classification_outputs.csv",
        "BaselineResults.csv",
        "FinalResults.csv",
        "corrected_resampled_tests_holm.csv",
        "paired_comparisons_holm.csv",
    ):
        path = output / name
        frame = pd.read_csv(path)
        frame["folds_available"] = frame.task.map(counts)
        frame["folds_expected"] = expected
        frame["evidence_status"] = frame.task.map(
            lambda task: (
                "complete_partition"
                if counts[task] == expected
                else "PARTIAL_EXPLORATORY"
            )
        )
        frame.to_csv(path, index=False)
    raw = {
        "outer_predictions.csv",
        "inner_search.csv",
        "task_inclusion.csv",
        "task_label_audit.csv",
    }
    for task, count in counts.items():
        for path in (output / task).glob("*.csv"):
            if path.name in raw or not path.stat().st_size:
                continue
            try:
                frame = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                continue
            frame["evidence_status"] = (
                "complete_partition" if count == expected else "PARTIAL_EXPLORATORY"
            )
            frame["folds_available"] = count
            frame["folds_expected"] = expected
            frame.to_csv(path, index=False)
    rows = []
    for task in config["tasks"]:
        for row in json.loads((output / task / "selections.json").read_text()):
            if row["objective"] in config["objectives"]:
                rows.append(
                    dict(
                        task=task,
                        repeat=row["repeat"],
                        fold=row["outer_fold"],
                        stream=row["stream"],
                        candidate=json.dumps(row["candidate"], sort_keys=True),
                        fit_seconds=row.get("model_fit_seconds"),
                        features=row["input_features"],
                        frequencies_hz=json.dumps(row["frequencies_hz"]),
                        folds_available=counts[task],
                        folds_expected=expected,
                    )
                )
    pd.DataFrame(rows).to_csv(output / "review_selected_parameters.csv", index=False)
    budget = [
        dict(
            fold=identity,
            source_successful_candidates=entry["source_successful_candidates"],
            unique_candidates_including_baselines=entry[
                "unique_candidates_including_baselines"
            ],
            inner_fits_excluding_controls=entry["unique_candidates_including_baselines"]
            * config["inner_folds"],
        )
        for identity, entry in plan.items()
    ]
    pd.DataFrame(budget).to_csv(output / "fine_tuning_budget.csv", index=False)
    lines = [
        "# Reduced fold-specific fine tuning",
        "",
        "Original inner-training winners only; original patient partition reused.",
        "This is an additional adaptive analysis, not independent validation.",
        "",
        "| Task | Folds | Evidence |",
        "|---|---:|---|",
    ]
    lines += [
        f"| {task} | {n}/{expected} | {'complete partition' if n == expected else 'PARTIAL: exclude from final claims'} |"
        for task, n in counts.items()
    ]
    lines += [
        "",
        f"{len(plan)} fold jobs. No source classifier results copied as new results.",
        "",
        "See classification_outputs.csv for AUC/F1/precision/SN/SP/PPV/NPV and patient-bootstrap CIs;",
        "corrected_resampled_tests_holm.csv for corrected CV tests; review_selected_parameters.csv for settings.",
        "MC statistics/intervals condition on the four available folds; they do not represent the full cohort.",
        "Ablations were not rerun; unavailable_streams.json records their absence.",
    ]
    (output / "REVIEW_RESULTS.md").write_text("\n".join(lines) + "\n")
