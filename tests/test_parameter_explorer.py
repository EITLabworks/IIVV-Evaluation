"""Saved-data identity checks only; no models or evaluation functions imported."""

import csv
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "parameter_explorer",
    Path(__file__).resolve().parents[1] / "scripts/create_parameter_explorer.py",
)
explorer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(explorer)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_outer_metrics_join_exact_fold_stream_and_unit(tmp_path):
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            dict(
                status="completed_partial",
                config={
                    "tasks": ["MC"],
                    "outer_folds": 5,
                    "fine_tuning": {"folds": {"MC": [0, 1, 3, 4]}},
                },
            )
        )
    )
    write_csv(
        tmp_path / "MC/inner_search.csv",
        [
            dict(
                model="lr",
                repeat="0",
                outer_fold="0",
                evaluation="nested_cv",
                stage="tuning",
                params_json='{"C":2}',
                auc="0.6",
            )
        ],
    )
    write_csv(
        tmp_path / "MC/outer_fold_metrics.csv",
        [
            dict(
                task="MC",
                repeat="0",
                outer_fold="0",
                stream="auc_lr",
                evaluation="nested_cv",
                unit="patient_weighted_site",
                auc="0.8",
            )
        ],
    )
    candidate = json.dumps(dict(model="lr", n_freq=3, params_json='{"C":1}'))
    (tmp_path / "MC/candidate_failures.json").write_text(
        json.dumps(
            [
                dict(
                    fold="nested_cv_repeat0_fold0",
                    reason="iteration limit",
                    candidate=dict(
                        model="svm", params_json='{"C":3,"kernel":"</script>"}'
                    ),
                )
            ]
        )
    )
    write_csv(
        tmp_path / "review_selected_parameters.csv",
        [
            dict(task="MC", repeat="0", fold="0", stream="auc_lr", candidate=candidate),
            dict(task="MC", repeat="0", fold="1", stream="auc_lr", candidate=candidate),
        ],
    )
    rows, tasks = explorer.load_points(tmp_path)
    assert tasks == ["MC"]
    inner = [row for row in rows if row["mode"] == "inner"]
    outer = [row for row in rows if row["mode"] == "outer"]
    assert len(inner) == len(outer) == 1
    assert inner[0]["C"] == 2 and inner[0]["auc"] == 0.6
    assert outer[0]["C"] == 1 and outer[0]["auc"] == 0.8
    assert outer[0]["status"].startswith("EXPLORATORY: 4/5")
    failed = [row for row in rows if row["mode"] == "failed"]
    assert len(failed) == 1 and failed[0]["auc"] is None
    assert failed[0]["reason"] == "iteration limit"
    output = explorer.build_explorer(tmp_path)
    assert "/*DATA*/" not in output.read_text()
    assert '<html lang="en">' in output.read_text()
    assert "Reset filters" in output.read_text()
    assert "\\u003c/script>" in output.read_text()


def test_missing_metrics_and_html_payload_are_safe():
    row = explorer.point(
        dict(model="svm", params_json='{"kernel":"</script>"}'),
        dict(repeat="0", auc="NaN"),
        "MC",
        "inner",
    )
    assert row["auc"] is None and row["f1"] is None
    assert "NaN" not in json.dumps(row, allow_nan=False)


def test_active_run_is_rejected(tmp_path):
    (tmp_path / "run_manifest.json").write_text('{"status":"running"}')
    with pytest.raises(ValueError, match="finished"):
        explorer.load_points(tmp_path)


def test_final_selection_uses_actual_mc_partition_without_partial_label(tmp_path):
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            dict(
                status="completed",
                config=dict(
                    tasks=["MC"],
                    outer_folds=5,
                    outer_repeats=1,
                    final_selection={"scope": "adaptive"},
                ),
            )
        )
    )
    (tmp_path / "MC").mkdir()
    (tmp_path / "MC/splits.json").write_text(
        json.dumps([dict(evaluation="nested_cv", outer_fold=i) for i in range(5)])
    )
    write_csv(
        tmp_path / "MC/inner_search.csv",
        [
            dict(
                model="svm",
                repeat="0",
                outer_fold="2",
                evaluation="nested_cv",
                stage="tuning",
                params_json='{"C":1}',
                auc="0.6",
            )
        ],
    )
    rows, tasks = explorer.load_points(tmp_path)
    assert tasks == ["MC"] and len(rows) == 1
    assert rows[0]["fold"] == "2"
    assert (
        rows[0]["status"] == "complete original partition; adaptive follow-up analysis"
    )
