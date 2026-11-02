"""Prepared checks for reduced search; no fitting in these unit tests."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from revision.features import Candidate
from revision.finetuning import neighbours, prepare, validate_split


@pytest.fixture
def fine_config():
    return json.loads(Path("configs/revision-finetuning.json").read_text())


def test_neighbourhood_is_bounded_and_rf_unchanged(fine_config):
    svm = Candidate(model="svm", params_json='{"C":0.001,"kernel":"linear"}')
    choices = neighbours(svm, fine_config)
    assert len(choices) == 2
    assert {json.loads(c.params_json)["C"] for c in choices} == {0.001, 0.002}
    assert all(json.loads(c.params_json)["max_iter"] == 100000 for c in choices)
    assert all(replace(c, params_json=svm.params_json) == svm for c in choices)
    rf = Candidate(model="rf", params_json='{"n_estimators":800,"max_depth":8}')
    assert len(neighbours(rf, fine_config)) == 1
    assert json.loads(neighbours(rf, fine_config)[0].params_json) == json.loads(
        rf.params_json
    )


def test_split_guard_rejects_other_outer_and_inner_patients():
    split = dict(
        train_sample_ids=["a"],
        test_sample_ids=["b"],
        train_patients=["1"],
        test_patients=["2"],
        inner=[{"train_sample_ids": ["a"]}],
    )
    validate_split({"split": split}, split)
    with pytest.raises(ValueError, match="test_patients"):
        validate_split({"split": split}, dict(split, test_patients=["3"]))
    with pytest.raises(ValueError, match="inner partitions"):
        validate_split({"split": split}, dict(split, inner=[]))


def test_preparation_hashes_and_freezes_training_winners(tmp_path, fine_config):
    from revision.data import digest

    source, output = tmp_path / "source", tmp_path / "new"
    source.mkdir()
    cfg = dict(fine_config)
    cfg["tasks"] = ["MC"]
    cfg["fine_tuning"] = dict(
        cfg["fine_tuning"], source=str(source), folds={"MC": [0, 1]}
    )
    original = {k: v for k, v in cfg.items() if k != "fine_tuning"}
    hashes = {"data/a.npz": "saved-hash"}
    (source / "run_manifest.json").write_text(
        json.dumps(dict(config=original, synthetic=False, input_hashes=hashes))
    )
    required = [
        "splits.json",
        "selections.json",
        "candidate_failures.json",
        "outer_predictions.csv",
        "inner_search.csv",
        "task_inclusion.csv",
        "task_label_audit.csv",
        "classes.json",
    ]
    for fold in [0, 1]:
        folder = source / "fold-checkpoints/MC" / f"nested_cv_repeat0_fold{fold}"
        folder.mkdir(parents=True)
        rows = []
        for objective in cfg["objectives"]:
            for model in cfg["models"]:
                params = (
                    {"C": 1, "kernel": "rbf"}
                    if model == "svm"
                    else ({"C": 1} if model == "lr" else {"n_estimators": 100})
                )
                rows.append(
                    dict(
                        task="MC",
                        repeat=0,
                        outer_fold=fold,
                        evaluation="nested_cv",
                        stream=f"{objective}_{model}",
                        outer_auc=0.999,
                        candidate=Candidate(
                            model=model, params_json=json.dumps(params)
                        ).dictionary(),
                    )
                )
        for name in required:
            value = (
                rows
                if name == "selections.json"
                else [{}] if name == "splits.json" else []
            )
            (folder / name).write_text(json.dumps(value))
        (folder / "complete.json").write_text(
            json.dumps(
                dict(
                    task="MC",
                    fold_key=[0, fold, "nested_cv"],
                    artifacts={name: digest(folder / name) for name in required},
                )
            )
        )
    plan = prepare(cfg, hashes, output)
    assert len(plan) == 2
    assert all("outer_auc" not in str(entry["candidates"]) for entry in plan.values())
    assert prepare(cfg, hashes, output) == plan
    with pytest.raises(ValueError, match="input hashes"):
        prepare(cfg, {"data/a.npz": "modified"}, output)
    changed = dict(cfg, fine_tuning=dict(cfg["fine_tuning"], c_factors=[1.0]))
    with pytest.raises(ValueError, match="Frozen"):
        prepare(changed, hashes, output)
    folder = source / "fold-checkpoints/MC/nested_cv_repeat0_fold0"
    (folder / "selections.json").write_text("tampered")
    with pytest.raises(ValueError, match="Changed completed"):
        prepare(cfg, hashes, output)


def test_checker_does_not_accept_running_or_changed_plan(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "check_finetuning", "scripts/check_finetuning.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "run_manifest.json"
    path.write_text(json.dumps(dict(status="running")))
    with pytest.raises(ValueError, match="not finished"):
        module.main(["--results-dir", str(tmp_path)])
    (tmp_path / "fine_tuning_plan.json").write_text("{}")
    path.write_text(
        json.dumps(
            dict(status="completed_partial", config={}, fine_tuning_plan_sha256="wrong")
        )
    )
    with pytest.raises(ValueError, match="shortlist changed"):
        module.main(["--results-dir", str(tmp_path)])


def test_fine_mode_rejects_convergence_warning(tmp_path, monkeypatch):
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    import revision.pipeline as pipeline
    from revision.data import load_dataset
    from run_revision import synthetic_fixture

    class Nonconverged:
        def fit(self, *args, **kwargs):
            warnings.warn("test iteration limit", ConvergenceWarning)
            return self

    data, patterns = synthetic_fixture(tmp_path / "fixture")
    dataset = load_dataset(data, patterns_path=patterns)
    indices, y, _ = dataset.task("healthy_vs_cancer")
    groups = dataset.table.iloc[indices].patient_id.to_numpy()
    train, test = pipeline.group_splits(y, groups, 3, 1729)[0]
    monkeypatch.setattr(pipeline, "make_model", lambda *args, **kwargs: Nonconverged())
    with pytest.raises(ValueError, match="test iteration limit"):
        pipeline.fit_predict(
            dataset,
            indices,
            y,
            Candidate(model="lr"),
            train,
            test,
            1729,
            {"fine_tuning": {"svm_max_iter": 100000}},
        )
