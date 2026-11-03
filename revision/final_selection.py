"""Review exports for a fresh reduced search using the existing nested loop."""

import json
from pathlib import Path
import shutil

import pandas as pd

from .pipeline import write_json
from .data import digest


def annotate(output, config):
    output = Path(output)
    repository = Path(__file__).resolve().parents[1]
    rationale_files = {
        "FINAL_SELECTION.md": repository / "FINAL_SELECTION.md",
        "revision-final-evidence.json": repository
        / "configs/revision-final-evidence.json",
    }
    for name, source in rationale_files.items():
        shutil.copyfile(source, output / name)
    selected, unavailable, budget = [], [], []
    counts = {}
    for task in config["tasks"]:
        folder = output / task
        splits = json.loads((folder / "splits.json").read_text())
        counts[task] = len(splits)
        for row in json.loads((folder / "selections.json").read_text()):
            if row["objective"] not in config["objectives"]:
                continue
            selected.append(
                dict(
                    task=task,
                    repeat=row["repeat"],
                    fold=row["outer_fold"],
                    stream=row["stream"],
                    candidate=json.dumps(row["candidate"], sort_keys=True),
                    fit_seconds=row.get("model_fit_seconds"),
                    model_max_iter=row.get("model_max_iter"),
                    features=row["input_features"],
                    frequencies_hz=json.dumps(row["frequencies_hz"]),
                    folds_available=len(splits),
                    folds_expected=config["outer_folds"] * config["outer_repeats"],
                )
            )
        unavailable.extend(
            dict(row, task=task)
            for row in json.loads((folder / "unavailable_streams.json").read_text())
        )
        search = pd.read_csv(folder / "inner_search.csv")
        for (repeat, fold, evaluation), part in search.groupby(
            ["repeat", "outer_fold", "evaluation"]
        ):
            marker = (
                output
                / "fold-checkpoints"
                / task
                / f"{evaluation}_repeat{repeat}_fold{fold}"
                / "complete.json"
            )
            budget.append(
                dict(
                    task=task,
                    repeat=repeat,
                    outer_fold=fold,
                    evaluation=evaluation,
                    successful_unique_candidates=len(part),
                    inner_fits=config["inner_folds"] * len(part),
                    inner_evaluation_seconds=part["inner_evaluation_seconds"].sum(),
                    outer_job_elapsed_seconds=(
                        json.loads(marker.read_text())["elapsed_seconds"]
                        if marker.is_file()
                        else None
                    ),
                    note="Successful search/baseline/covariate fits; matched control and outer refits excluded",
                )
            )
    pd.DataFrame(selected).to_csv(
        output / "review_selected_parameters.csv", index=False
    )
    pd.DataFrame(
        unavailable,
        columns=[
            "task",
            "evaluation",
            "stream",
            "expected_predictions",
            "observed_predictions",
            "reason",
        ],
    ).to_csv(output / "unavailable_results.csv", index=False)
    budget_frame = pd.DataFrame(budget)
    budget_frame.to_csv(output / "final_search_budget.csv", index=False)
    print(
        "FINAL FOLD TIMES (seconds; successful candidate counts):\n"
        + budget_frame[
            [
                "task",
                "outer_fold",
                "successful_unique_candidates",
                "outer_job_elapsed_seconds",
            ]
        ].to_string(index=False),
        flush=True,
    )
    write_json(
        output / "final_selection_protocol.json",
        dict(
            config=config,
            task_fold_counts=counts,
            selection_scope="Per-fold scoring and winners use current inner-training/validation data only. Runtime reads no prior classifier results; the search-space design is informed by earlier cohort analyses.",
            interpretation="Adaptive reduced follow-up on the original cohort/partition; no independent confirmation or noninferiority claim",
            omitted_parameter_rationale="FINAL_SELECTION.md",
            rationale_artifact_hashes={
                name: digest(output / name) for name in rationale_files
            },
            unavailable_streams=len(unavailable),
        ),
    )
    lines = [
        "# Final reduced selection",
        "",
        "Fresh computations for every scheduled task/fold; previous classifier outputs are not reused.",
        "Selection, scaling, imputation, frequency ranking, masks, calibration and thresholds remain training-only.",
        "The design follows earlier experience on this cohort; it is adaptive, not independent confirmation.",
        "",
        "| Task | Outer computations |",
        "|---|---:|",
        *[f"| {task} | {n} |" for task, n in counts.items()],
        "",
        f"Unavailable streams: {len(unavailable)}. Failed fits are rejected; partial streams have no pooled metric.",
        "Ablation outputs cover retained alternatives only; omitted selectors/representations are not newly compared.",
        "",
        "- classification_outputs.csv: AUC, accuracy, F1, precision, SN, SP, PPV, NPV and patient-bootstrap intervals.",
        "- corrected_resampled_tests_holm.csv: corrected CV tests with the existing global Holm family.",
        "- review_selected_parameters.csv: selected model, hyperparameters, frequency identities and input counts.",
        "- final_search_budget.csv and fold-checkpoints/*/*/inner_search.csv: evaluated candidates and measured search times.",
        "- Task folders retain demographics, pathology splits, confusion matrices, ROC, calibration, controls and timings.",
        "- FINAL_SELECTION.md explains every excluded search option; narrowing the search does not prove equal performance.",
    ]
    (output / "REVIEW_RESULTS.md").write_text("\n".join(lines) + "\n")
