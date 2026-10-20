"""Verify completed result structure and patient separation without fitting models."""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import CLASS_NAMES, digest
from .metrics import holm


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_results(root, *, verify_inputs=False):
    root = Path(root)
    manifest = json.loads((root / "run_manifest.json").read_text())
    require(manifest.get("status") == "completed", "Run is not completed")
    cfg = manifest["config"]
    checked_folds = 0
    require(manifest.get("input_hashes"), "Missing input inventory")
    if verify_inputs:
        repository = Path(__file__).resolve().parents[1]
        inventory = {
            **manifest["input_hashes"],
            **manifest.get("source_input_hashes", {}),
        }
        for name, expected in inventory.items():
            path = Path(name)
            path = path if path.is_absolute() else repository / path
            require(
                path.is_file() and digest(path) == expected,
                f"Input missing or changed: {name}",
            )
    for task in cfg["tasks"]:
        folder = root / task
        required = [
            "outer_predictions.csv",
            "inner_search.csv",
            "splits.json",
            "selections.json",
            "classes.json",
            "task_inclusion.csv",
            "per_class_metrics.csv",
            "metrics_summary.csv",
            "frequency_and_pattern_identities.json",
            "measured_computation_times.csv",
            "hardware_requirements.csv",
            "split_population_counts.csv",
            "ranking_stability.csv",
            "candidate_failures.json",
            "unavailable_streams.json",
            "location_stratified_metrics.csv",
        ]
        # Location is optional for inputs outside the study's full source profile.
        for name in required:
            if (
                name == "location_stratified_metrics.csv"
                and not (folder / name).exists()
            ):
                continue
            require((folder / name).is_file(), f"{task}: missing {name}")
        classes = json.loads((folder / "classes.json").read_text())
        require(
            classes["classes"] == list(CLASS_NAMES[task]),
            f"{task}: incorrect class order",
        )
        if len(classes["classes"]) == 2:
            require(
                classes["positive_class"] == CLASS_NAMES[task][1],
                f"{task}: incorrect positive class",
            )
        inclusion = pd.read_csv(
            folder / "task_inclusion.csv", dtype={"patient_id": str, "sample_id": str}
        )
        require(inclusion.sample_id.is_unique, f"{task}: duplicate inclusion IDs")
        lookup = inclusion.set_index("sample_id")
        splits = json.loads((folder / "splits.json").read_text())
        selections = json.loads((folder / "selections.json").read_text())
        predictions = pd.read_csv(
            folder / "outer_predictions.csv",
            dtype={"patient_id": str, "sample_id": str},
        )
        keys = ["evaluation", "repeat", "outer_fold"]
        require(
            not predictions.duplicated([*keys, "stream", "sample_id"]).any(),
            f"{task}: duplicate predictions",
        )
        selection_frame = pd.DataFrame(selections)
        require(
            not selection_frame.duplicated([*keys, "stream"]).any(),
            f"{task}: duplicate selections",
        )
        nested = [s for s in splits if s["evaluation"] == "nested_cv"]
        require(
            len(nested) == cfg["outer_folds"] * cfg["outer_repeats"],
            f"{task}: incomplete outer folds",
        )
        split_keys = [tuple(s[key] for key in keys) for s in splits]
        require(len(set(split_keys)) == len(split_keys), f"{task}: duplicate splits")
        require(
            set(map(tuple, predictions[keys].drop_duplicates().to_numpy()))
            == set(split_keys),
            f"{task}: unexpected/missing prediction folds",
        )
        for repeat in range(cfg["outer_repeats"]):
            rows = [s for s in nested if s["repeat"] == repeat]
            require(
                {s["outer_fold"] for s in rows} == set(range(cfg["outer_folds"])),
                f"{task}: wrong fold IDs",
            )
            sites = [v for s in rows for v in s["test_sample_ids"]]
            development = set(rows[0]["train_sample_ids"]) | set(
                rows[0]["test_sample_ids"]
            )
            require(
                len(sites) == len(set(sites)) and set(sites) == development,
                f"{task}: invalid outer coverage",
            )
        expected_streams = {f"baseline_{m}" for m in cfg["models"]}
        expected_streams |= {
            f"{o}_{m}" for o, m in itertools.product(cfg["objectives"], cfg["models"])
        }
        expected_streams |= {f"{o}_selected" for o in cfg["objectives"]}
        for split in splits:
            train = set(split["train_patients"])
            test = set(split["test_patients"])
            require(not train & test, f"{task}: outer patient overlap")
            require(
                len(split["inner"]) == cfg["inner_folds"],
                f"{task}: wrong inner fold count",
            )
            for inner in split["inner"]:
                a = set(inner["train_patients"])
                b = set(inner["validation_patients"])
                require(
                    not a & b and a | b == train and not (a | b) & test,
                    f"{task}: invalid inner patient partition",
                )
            require(
                set(lookup.loc[split["train_sample_ids"]].patient_id) == train,
                f"{task}: training ID mismatch",
            )
            require(
                set(lookup.loc[split["test_sample_ids"]].patient_id) == test,
                f"{task}: test ID mismatch",
            )
            part = predictions
            chosen = selection_frame
            for key in keys:
                part = part[part[key] == split[key]]
                chosen = chosen[chosen[key] == split[key]]
            require(
                expected_streams <= set(part.stream),
                f"{task}: missing baseline/tuned streams",
            )
            require(
                set(chosen.stream) == set(part.stream),
                f"{task}: prediction/selection streams disagree",
            )
            expected_sites = set(split["test_sample_ids"])
            for stream, group in part.groupby("stream"):
                require(
                    set(group.sample_id) == expected_sites,
                    f"{task}/{stream}: omitted outer test sites",
                )
                truth = lookup.loc[group.sample_id]
                require(
                    np.array_equal(
                        truth.patient_id.to_numpy(), group.patient_id.to_numpy()
                    )
                    and np.array_equal(
                        truth.class_index.to_numpy(), group.y_true.to_numpy()
                    ),
                    f"{task}: prediction label mismatch",
                )
                require(
                    group.covered.isin([True, False]).all(),
                    f"{task}: invalid coverage flags",
                )
                covered = group.covered.to_numpy(bool)
                p = group[[f"p_{i}" for i in range(len(classes["classes"]))]].to_numpy(
                    float
                )
                require(
                    np.isfinite(p[covered]).all()
                    and ((p[covered] >= 0) & (p[covered] <= 1)).all()
                    and np.allclose(p[covered].sum(1), 1),
                    f"{task}: invalid probabilities",
                )
                require(
                    np.isnan(p[~covered]).all()
                    and group.loc[~covered, "y_pred"].eq(-1).all(),
                    f"{task}: invalid abstentions",
                )
                selected = chosen[chosen.stream == stream].iloc[0]
                require(
                    set(selected.train_patients) <= train
                    and set(selected.test_patients) == test,
                    f"{task}: fitted patient IDs disagree with split",
                )
            checked_folds += 1
        if "covariate_combined" in set(predictions.stream):
            control = json.loads(
                (folder / "matched_covariate_control.json").read_text()
            )
            require(
                control.get("status") == "completed",
                f"{task}: matched control incomplete",
            )
            for _, reference in selection_frame[
                selection_frame.stream == "covariate_combined"
            ].iterrows():
                matched = selection_frame[
                    selection_frame.stream == "covariate_eis_only"
                ]
                for key in keys:
                    matched = matched[matched[key] == reference[key]]
                require(len(matched) == 1, f"{task}: missing matched LR fold")
                other = matched.iloc[0]
                a = dict(reference.candidate)
                b = dict(other.candidate)
                a.pop("stage")
                b.pop("stage")
                require(a == b, f"{task}: matched EIS configuration differs")
                for key in [
                    "frequency_indices_0based",
                    "pattern_indices_0based",
                    "train_patients",
                    "test_patients",
                ]:
                    require(
                        reference[key] == other[key],
                        f"{task}: matched LR {key} differs",
                    )
    comparisons = pd.read_csv(root / "paired_comparisons_holm.csv")
    expected = set(itertools.product(cfg["tasks"], cfg["models"], cfg["objectives"]))
    observed = list(zip(comparisons.task, comparisons.model, comparisons.objective))
    require(
        len(observed) == len(expected) and set(observed) == expected,
        "Incomplete prespecified comparison family",
    )
    require(
        np.allclose(
            comparisons.p_holm_all_tasks_models_objectives,
            holm(comparisons.p_value),
            equal_nan=True,
        ),
        "Incorrect global Holm adjustment",
    )
    for name in [
        "all_metrics.csv",
        "BaselineResults.csv",
        "FinalResults.csv",
        "analysis_status.json",
        "environment-freeze.txt",
    ]:
        require((root / name).is_file(), f"Missing {name}")
    return dict(
        status="verified",
        synthetic=bool(manifest.get("synthetic")),
        checked_outer_folds=checked_folds,
        comparison_family=len(expected),
        input_hashes_checked=verify_inputs,
    )
