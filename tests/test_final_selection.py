"""Run on the server; includes tiny numerical parity checks, never clinical data."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from revision.features import Candidate
from revision.pipeline import screen_candidates, tune_candidates
from run_revision import validate_config
from sklearn.model_selection import ParameterGrid


@pytest.fixture
def config():
    return json.loads(Path("configs/revision.json").read_text())


def test_final_search_budget_and_original_patient_protocol(config):
    validate_config(config)
    assert config["tasks"] == [
        "healthy_vs_cancer",
        "highrisk_vs_lowrisk",
        "cancer_vs_lr_hr",
    ]
    assert config["schema_version"] == 3
    assert "fine_tuning" not in config
    screened = screen_candidates(config, 31, True)
    dimensions = {
        "frequency": ("n_freq", config["frequency_counts"]),
        "mask": ("mask", config["masks"]),
        "z_threshold": ("mask", [f"Zq:{q}" for q in config["z_quantiles"]]),
        "selector": ("selector", config["selectors"]),
        "representation": ("representation", config["representations"]),
        "quality": ("cutoff", config["cutoffs"]),
        "normalization": ("normalization", config["normalizations"]),
        "imputation": ("imputation", config["imputations"]),
        "weighting": ("class_weight", config["class_weights"]),
        "missingness": ("representation", ["missingness"]),
    }
    assert {c.stage for c in screened} == set(dimensions)
    for stage, (field, values) in dimensions.items():
        assert {getattr(c, field) for c in screened if c.stage == stage} == set(values)
    assert len(screened) == sum(len(set(values)) for _, values in dimensions.values())
    assert {c.selector for c in screened if c.stage == "selector"} == {
        "pca",
        "mutual_info",
    }
    assert len([c for c in screened if c.stage == "z_threshold"]) == 5
    records = [
        dict(candidate=c, objectives={goal: 0.5 for goal in config["objectives"]})
        for c in screened
    ]
    expected_frequencies = min(config["top_k"], len(config["frequency_counts"]))
    expected_masks = min(config["top_k"], len(config["masks"])) + 1
    parameter_count = sum(
        len(ParameterGrid(config["model_grids"][model])) for model in config["models"]
    )
    for objective in config["objectives"]:
        choices = tune_candidates(records, config, objective, 31)
        assert len(choices) == expected_frequencies * expected_masks * parameter_count
        assert {c.model for c in choices} == {"svm", "lr", "rf"}
        assert len({c.mask for c in choices}) == expected_masks
        assert len({c.n_freq for c in choices}) == expected_frequencies
        assert all(
            c.normalization in config["normalizations"]
            and c.cutoff in config["cutoffs"]
            for c in choices
        )
    assert config["fold_workers"] * max(config["rf_jobs"], config["blas_threads"]) == 32


@pytest.mark.parametrize("repeats", [1, 5])
@pytest.mark.parametrize("seed", [42, 1729])
def test_repeated_patient_schedule_is_complete_and_deterministic(
    config, repeats, seed, monkeypatch
):
    import revision.pipeline as pipeline
    from revision.data import Dataset
    from revision.data_quality import quality_protocol

    config = dict(config, outer_repeats=repeats, seed=seed)
    table = pd.DataFrame(
        [
            dict(patient_id=str(patient), pathology=pathology)
            for patient in range(20)
            for pathology in (
                "Healthy",
                "Cancer",
                "Non-malignant",
                "High-grade dysplasia",
            )
        ]
    )
    dataset = Dataset(
        np.zeros((len(table), 1, 1)), table, np.array([100]), None, False, {}
    )
    original = pipeline.group_splits
    seeds = []

    def record_seeds(y, groups, folds, seed):
        seeds.append(seed)
        return original(y, groups, folds, seed)

    monkeypatch.setattr(pipeline, "group_splits", record_seeds)
    assert quality_protocol(config)["outer_repeats"] == repeats
    for task in config["tasks"]:
        seeds.clear()
        indices, y, _ = dataset.task(task)
        groups = table.iloc[indices].patient_id.to_numpy()
        schedule = pipeline.task_outer_splits(dataset, task, config)
        assert seeds == [config["seed"] + repeat * 1000 for repeat in range(repeats)]
        assert len(schedule) == config["outer_folds"] * repeats
        for repeat in range(repeats):
            rows = [row for row in schedule if row[0] == repeat]
            assert {row[1] for row in rows} == set(range(config["outer_folds"]))
            np.testing.assert_array_equal(
                np.sort(np.concatenate([row[3] for row in rows])),
                np.arange(len(indices)),
            )
            for _, fold, train, test, evaluation in rows:
                assert evaluation == "nested_cv"
                assert not set(groups[train]) & set(groups[test])
                inner = original(
                    y[train],
                    groups[train],
                    config["inner_folds"],
                    config["seed"] + repeat * 1000 + fold * 100 + 1,
                )
                for fit, validation in inner:
                    assert not set(groups[train[fit]]) & set(groups[train[validation]])
                    assert not set(groups[test]) & set(groups[train[fit]])
        for first, second in zip(
            schedule, pipeline.task_outer_splits(dataset, task, config)
        ):
            assert first[:2] == second[:2] and first[4] == second[4]
            np.testing.assert_array_equal(first[2], second[2])
            np.testing.assert_array_equal(first[3], second[3])


@pytest.mark.parametrize("repeats", [0, -1, 1.5, True, "5"])
def test_config_rejects_invalid_outer_repeats(config, repeats):
    with pytest.raises(ValueError, match="outer_repeats"):
        validate_config(dict(config, outer_repeats=repeats))


@pytest.mark.parametrize("kernel", ["linear", "rbf"])
def test_cached_training_gram_preserves_decisions(kernel):
    from revision.kernel_svm import VectorizedSVC

    rng = np.random.default_rng(17)
    x = rng.normal(size=(18, 40))
    y = np.repeat(np.arange(3), 6)
    weights = np.linspace(0.7, 1.3, len(y))
    parameters = dict(
        kernel=kernel, C=0.3, class_weight="balanced", random_state=19, tol=1e-9
    )
    gram = x @ x.T
    saved = gram.copy()
    reference = VectorizedSVC(**parameters).fit(x, y, sample_weight=weights)
    cached = VectorizedSVC(**parameters).fit(
        x, y, sample_weight=weights, training_gram=gram
    )
    test = rng.normal(size=(7, 40))
    np.testing.assert_allclose(
        cached.decision_function(test),
        reference.decision_function(test),
        rtol=1e-11,
        atol=1e-11,
    )
    np.testing.assert_array_equal(gram, saved)
    with pytest.raises(ValueError, match="Gram matrix"):
        VectorizedSVC(**parameters).fit(x, y, training_gram=np.ones((2, 2)))


def test_final_config_rejects_conflicting_modes_and_unsafe_cache(config):
    bad = copy.deepcopy(config)
    bad["fine_tuning"] = {"source": "not-used"}
    with pytest.raises(ValueError, match="fresh search"):
        validate_config(bad)
    bad = copy.deepcopy(config)
    bad["model_grids"]["svm"]["kernel"] = ["poly"]
    with pytest.raises(ValueError, match="Gram caching"):
        validate_config(bad)
    bad = copy.deepcopy(config)
    bad["screen_defaults"]["normalization"] = "per_coordinate"
    with pytest.raises(ValueError, match="screen_defaults.normalization"):
        validate_config(bad)


@pytest.mark.parametrize("repeat", [0, 4])
def test_progress_tracks_worker_and_phase_and_skips_stale_snapshots(
    tmp_path, capsys, repeat
):
    import time
    from revision.progress import update_progress, heartbeat

    folder = (
        tmp_path / f"fold-checkpoints/cancer_vs_lr_hr/nested_cv_repeat{repeat}_fold2"
    )
    config = {
        "progress_interval_seconds": 60,
        "_progress_path": str(folder / "progress.json"),
    }
    update_progress(
        config, "model-fit", candidates_completed=3, candidates_planned=20, inner_fold=2
    )
    state = json.loads((folder / "progress.json").read_text())
    assert state["pid"] > 0 and state["phase"] == "model-fit"
    heartbeat([("cancer_vs_lr_hr", (repeat, 2, "nested_cv"), tmp_path)], 1, 20)
    assert (
        f"cancer_vs_lr_hr/r{repeat}/f2: model-fit candidate=3/20 inner=2"
        in capsys.readouterr().out
    )
    heartbeat(
        [("cancer_vs_lr_hr", (repeat, 2, "nested_cv"), tmp_path)],
        1,
        20,
        since_epoch=time.time() + 1,
    )
    assert f"cancer_vs_lr_hr/r{repeat}/f2" not in capsys.readouterr().out
    heartbeat([("cancer_vs_lr_hr", str(tmp_path))], 0, 4, label="reports")
    assert "reports=0/4" in capsys.readouterr().out


@pytest.mark.parametrize("repeats", [1, 5])
def test_review_exports_retain_parameters_and_honest_scope(tmp_path, config, repeats):
    from revision.final_selection import annotate

    config = dict(
        config, tasks=["cancer_vs_lr_hr"], outer_folds=2, outer_repeats=repeats
    )
    folder = tmp_path / "cancer_vs_lr_hr"
    folder.mkdir()
    expected_folds = 2 * repeats
    (folder / "splits.json").write_text(json.dumps([{}] * expected_folds))
    selection = dict(
        objective="auc",
        repeat=0,
        outer_fold=0,
        stream="auc_svm",
        candidate=Candidate(model="svm").dictionary(),
        model_fit_seconds=0.1,
        input_features=30,
        frequencies_hz=[100, 1000, 10000],
    )
    (folder / "selections.json").write_text(json.dumps([selection]))
    (folder / "unavailable_streams.json").write_text(
        json.dumps(
            [
                dict(
                    evaluation="nested_cv",
                    stream="auc_lr",
                    reason="test rejection",
                    observed_predictions=0,
                )
            ]
        )
    )
    pd.DataFrame(
        [
            dict(
                repeat=repeat,
                outer_fold=fold,
                evaluation="nested_cv",
                inner_evaluation_seconds=1.5,
            )
            for repeat in range(repeats)
            for fold in (0, 1)
        ]
    ).to_csv(folder / "inner_search.csv", index=False)
    annotate(tmp_path, config)
    table = pd.read_csv(tmp_path / "review_selected_parameters.csv")
    assert json.loads(table.iloc[0].candidate)["model"] == "svm"
    assert (
        table.iloc[0].folds_available == table.iloc[0].folds_expected == expected_folds
    )
    assert len(pd.read_csv(tmp_path / "unavailable_results.csv")) == 1
    protocol = json.loads((tmp_path / "final_selection_protocol.json").read_text())
    assert protocol["task_fold_counts"] == {"cancer_vs_lr_hr": expected_folds}
    assert "no independent confirmation" in protocol["interpretation"]
    assert protocol["omitted_parameter_rationale"] == "README.md"
    assert set(protocol["rationale_artifact_hashes"]) == {"README.md"}
    assert (tmp_path / "README.md").read_text() == Path("README.md").read_text()


def test_final_synthetic_pipeline_exports_and_verifies_all_models(tmp_path):
    from run_revision import main
    from revision.verification import verify_results

    output = tmp_path / "synthetic-final"
    assert (
        main(
            [
                "--config",
                "configs/revision.json",
                "--synthetic",
                "--tasks",
                "healthy_vs_cancer",
                "--models",
                "svm",
                "rf",
                "lr",
                "--fold-workers",
                "1",
                "--rf-jobs",
                "1",
                "--blas-threads",
                "1",
                "--bootstrap-samples",
                "2",
                "--no-plots",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert verify_results(output)["checked_outer_folds"] == 3
    assert (output / "review_selected_parameters.csv").is_file()
    assert (output / "final_search_budget.csv").is_file()
    parameters = pd.read_csv(output / "review_selected_parameters.csv")
    assert {json.loads(value)["model"] for value in parameters.candidate} == {
        "svm",
        "rf",
        "lr",
    }


def test_final_mode_rejects_convergence_warning(tmp_path, monkeypatch, config):
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

    def make_nonconverged(*args, **kwargs):
        assert kwargs["lr_max_iter"] == config["final_selection"]["lr_max_iter"]
        return Nonconverged()

    monkeypatch.setattr(pipeline, "make_model", make_nonconverged)
    with pytest.raises(ValueError, match="test iteration limit"):
        pipeline.fit_predict(
            dataset,
            indices,
            y,
            Candidate(model="lr"),
            train,
            test,
            1729,
            config,
        )


def test_lesion_pipeline_exports_the_confirmed_class_order(tmp_path):
    from run_revision import main
    from revision.verification import verify_results

    output = tmp_path / "synthetic-lesion"
    assert (
        main(
            [
                "--synthetic",
                "--tasks",
                "cancer_vs_lr_hr",
                "--models",
                "lr",
                "--objectives",
                "auc",
                "sensitivity",
                "--fold-workers",
                "1",
                "--rf-jobs",
                "1",
                "--blas-threads",
                "1",
                "--bootstrap-samples",
                "2",
                "--no-plots",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert verify_results(output)["checked_outer_folds"] == 3
    labels = json.loads((output / "cancer_vs_lr_hr/classes.json").read_text())
    assert labels["classes"] == [
        "Cancer",
        "Low-grade dysplasia",
        "High-grade dysplasia",
    ]
    assert labels["positive_class"] is None
    inclusion = pd.read_csv(output / "cancer_vs_lr_hr/task_inclusion.csv")
    assert "Healthy" not in set(inclusion.pathology)
    results = pd.read_csv(output / "cancer_vs_lr_hr/per_class_metrics.csv")
    assert set(results.class_name) == set(labels["classes"])


@pytest.mark.parametrize("fail_baseline", [False, True])
def test_failed_lr_candidates_is_recorded_and_other_models_finish(
    tmp_path, monkeypatch, config, fail_baseline
):
    import revision.pipeline as pipeline
    from revision.data import load_dataset
    from run_revision import synthetic_fixture

    data, patterns = synthetic_fixture(tmp_path / "fixture")
    dataset = load_dataset(data, patterns_path=patterns)
    cfg = dict(
        config,
        models=["lr", "svm"],
        outer_folds=3,
        inner_folds=2,
        clinical_covariates=[],
    )
    cfg["_fold_key"] = (0, 0, "nested_cv")
    monkeypatch.setattr(
        pipeline,
        "screen_candidates",
        lambda *args: [Candidate(stage="frequency", model="svm")],
    )

    def evaluate(_data, _indices, _y, candidate, *args):
        if candidate.model == "lr" and (candidate.stage == "tuning" or fail_baseline):
            raise ValueError("test LR iteration limit")
        return dict(
            candidate=candidate,
            metrics={},
            objectives={goal: 0.5 for goal in cfg["objectives"]},
            coverage=1,
            metadata=[],
        )

    monkeypatch.setattr(pipeline, "inner_evaluate", evaluate)

    class Identity:
        def transform(self, values):
            return values

    import numpy as np

    monkeypatch.setattr(
        pipeline,
        "fit_training_rules",
        lambda *args, **kwargs: (Identity(), 0.5, {}, {}),
    )
    monkeypatch.setattr(
        pipeline,
        "fit_predict",
        lambda d, i, y, c, tr, te, *args: (
            np.full((len(te), 2), 0.5),
            {"input_features": 1, "frequencies_hz": [1]},
            (None, None, None),
        ),
    )
    output = tmp_path / "output"
    predictions, selections = pipeline.run_task(
        dataset, "healthy_vs_cancer", cfg, output
    )
    streams = set(predictions.stream)
    assert "auc_svm" in streams and "auc_selected" in streams
    assert "auc_lr" not in streams
    assert ("baseline_lr" in streams) == (not fail_baseline)
    errors = json.loads(
        (output / "healthy_vs_cancer/candidate_failures.json").read_text()
    )
    assert any(row["reason"] == "test LR iteration limit" for row in errors)
    assert not (output / "healthy_vs_cancer/complete.json").exists()
