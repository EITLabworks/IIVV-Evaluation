"""Prepared for server execution; numerical exploration is not run during editing."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from revision.data import CLASS_NAMES, TASK3_CAPTION, TASK_PROTOCOL, Dataset
from revision.data_quality import freeze_quality, verify_quality, weighted_projection
from revision.pipeline import group_splits, make_split_record, task_outer_splits


def config():
    return json.loads(Path("configs/revision.json").read_text())


def test_three_tasks_keep_disjoint_lesions_and_distinguish_binary_risk():
    labels = ["Healthy", "Cancer", "High-grade dysplasia", "Non-malignant", None]
    table = pd.DataFrame({"pathology": labels})
    dataset = Dataset(
        np.zeros((5, 2, 1)), table, np.array([100, 1000]), None, False, {}
    )
    assert list(CLASS_NAMES) == config()["tasks"]
    indices, y, classes = dataset.task("cancer_vs_lr_hr")
    assert indices.tolist() == [1, 2, 3]
    assert [classes[k] for k in y] == [
        "Cancer",
        "High-grade dysplasia",
        "Low-grade dysplasia",
    ]
    indices, y, classes = dataset.task("highrisk_vs_lowrisk")
    assert [classes[k] for k in y] == ["Low risk", "High risk", "High risk", "Low risk"]
    assert "normal/benign" in TASK3_CAPTION and "excluded" in TASK3_CAPTION
    assert config()["task_protocol"] == TASK_PROTOCOL
    with pytest.raises(ValueError, match="Unknown task"):
        dataset.task("MC")


def test_patient_balanced_pca_is_invariant_to_duplicating_a_patients_only_site():
    x = np.array(
        [[1.0, 2.0, np.nan], [3.0, 8.0, 4.0], [5.0, 3.0, 6.0], [9.0, 7.0, 2.0]]
    )
    scores, loadings, ratio, observed = weighted_projection(x, ["a", "b", "c", "d"])
    duplicated = weighted_projection(np.vstack([x, x[0]]), ["a", "b", "c", "d", "a"])
    np.testing.assert_allclose(duplicated[0][:4], scores, atol=1e-12)
    np.testing.assert_allclose(duplicated[1], loadings, atol=1e-12)
    np.testing.assert_allclose(duplicated[2], ratio, atol=1e-12)
    assert observed[2] == 0.75
    with pytest.raises(ValueError, match="constant or missing"):
        weighted_projection(np.ones((4, 3)), ["a", "b", "c", "d"])


def test_quality_manifest_rejects_changed_artifacts_inputs_and_task_protocol(tmp_path):
    cfg = config()
    required = [
        "quality/clinical_metadata_status.json",
        "exploration/interpretation.json",
    ]
    required += [f"planned-splits/{task}/splits.json" for task in cfg["tasks"]]
    for name in required:
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("{}")
    dataset = Dataset(None, None, None, None, False, {"data/site.npz": "original"})
    freeze_quality(tmp_path, dataset, cfg, {"source.mat": "original"})
    verify_quality(tmp_path, cfg, dataset.input_hashes, {"source.mat": "original"})
    with pytest.raises(ValueError, match="different input"):
        verify_quality(
            tmp_path, cfg, {"data/site.npz": "changed"}, {"source.mat": "original"}
        )
    changed = copy.deepcopy(cfg)
    changed["task_protocol"] = "obsolete"
    with pytest.raises(ValueError, match="protocol differs"):
        verify_quality(tmp_path, changed)
    (tmp_path / required[0]).write_text("changed")
    with pytest.raises(ValueError, match="artifact missing, changed"):
        verify_quality(tmp_path, cfg)


def test_quality_and_training_share_the_exact_partition_inventory(tmp_path):
    from run_revision import synthetic_fixture
    from revision.data import load_dataset

    data, patterns = synthetic_fixture(tmp_path / "fixture")
    dataset = load_dataset(data, patterns_path=patterns)
    cfg = dict(config(), outer_folds=3, inner_folds=2)
    for task in cfg["tasks"]:
        indices, y, _ = dataset.task(task)
        groups = dataset.table.iloc[indices].patient_id.to_numpy()
        seen = []
        for repeat, fold, tr, te, evaluation in task_outer_splits(dataset, task, cfg):
            inner = group_splits(y[tr], groups[tr], 2, cfg["seed"] + fold * 100 + 1)
            row = make_split_record(
                dataset, task, indices, tr, te, inner, repeat, fold, evaluation
            )
            assert not set(row["train_patients"]) & set(row["test_patients"])
            assert all(
                not set(s["train_patients"]) & set(s["validation_patients"])
                for s in row["inner"]
            )
            seen.extend(row["test_sample_ids"])
        assert sorted(seen) == sorted(dataset.table.iloc[indices].sample_id)


def test_quality_only_exports_every_task_without_diagnostic_model_fitting(
    tmp_path, monkeypatch
):
    import run_revision
    import revision.pipeline as pipeline

    def forbidden(*args, **kwargs):
        pytest.fail("Quality-only mode must not fit diagnostic classifiers")

    monkeypatch.setattr(run_revision, "run_task", forbidden)
    monkeypatch.setattr(pipeline, "make_model", forbidden)
    output = tmp_path / "synthetic-quality"
    assert (
        run_revision.main(
            [
                "--synthetic",
                "--data-quality-only",
                "--bootstrap-samples",
                "2",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["status"] == "quality_completed"
    frozen = verify_quality(output, manifest["config"])
    assert frozen["synthetic"] is True
    for task in manifest["config"]["tasks"]:
        for suffix in ("pca", "mi-pca", "mutual-information", "quality-retention"):
            assert (output / "exploration" / f"{task}-{suffix}.pdf").is_file()
        assert (
            output / "planned-splits" / task / "cv_pathology_breakdown.csv"
        ).is_file()
    assert (output / "exploration/saved-test-mean-icc.pdf").is_file()
    assert not (output / "classification_outputs.csv").exists()
