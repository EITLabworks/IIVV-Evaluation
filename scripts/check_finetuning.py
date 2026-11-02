"""Read-only structural checks; a verified partial run remains partial."""

import argparse
import json
from pathlib import Path
import sys

# Support direct execution from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from revision.data import digest
from revision.finetuning import fold_id, validate_split
from revision.parallel import _check_complete


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results/revision-finetuning")
    )
    args = parser.parse_args(argv)
    root = args.results_dir
    manifest = json.loads((root / "run_manifest.json").read_text())
    if manifest["status"] not in {"completed", "completed_partial"}:
        raise ValueError("Fine-tuning computations have not finished")
    config = manifest["config"]
    plan_path = root / "fine_tuning_plan.json"
    if digest(plan_path) != manifest["fine_tuning_plan_sha256"]:
        raise ValueError("Frozen shortlist changed")
    for name, expected in {
        **manifest["input_hashes"],
        **manifest.get("source_input_hashes", {}),
    }.items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Changed input: {name}")
    plan = json.loads(plan_path.read_text())
    wanted = {
        fold_id(task, 0, fold)
        for task in config["tasks"]
        for fold in config["fine_tuning"]["folds"][task]
    }
    if set(plan) != wanted:
        raise ValueError("Incorrect shortlist fold inventory")
    for identity, entry in plan.items():
        task, name = identity.split("/")
        fold = entry["split"]["outer_fold"]
        folder = root / "fold-checkpoints" / identity
        if not _check_complete(folder, task, (0, fold, "nested_cv")):
            raise ValueError(f"Missing completed fine-tuning fold: {identity}")
        splits = json.loads((folder / "splits.json").read_text())
        if len(splits) != 1:
            raise ValueError(f"Incorrect split inventory: {identity}")
        validate_split(entry, splits[0])
        for split in [splits[0], *splits[0]["inner"]]:
            validation = split.get(
                "test_patients", split.get("validation_patients", [])
            )
            if set(split["train_patients"]) & set(validation):
                raise ValueError(f"Patient leakage: {identity}")
    for name in [
        "all_metrics.csv",
        "classification_outputs.csv",
        "review_selected_parameters.csv",
        "corrected_resampled_tests_holm.csv",
        "REVIEW_RESULTS.md",
    ]:
        if not (root / name).is_file():
            raise ValueError(f"Missing review artifact: {name}")
    print(
        f"Structural/input checks passed: {len(plan)} folds; status={manifest['status']}."
    )
    print(
        "MC remains partial/exploratory; this check does not validate a complete 20-fold result."
    )


if __name__ == "__main__":
    main()
