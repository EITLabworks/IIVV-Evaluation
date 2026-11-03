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


@pytest.fixture
def config():
    return json.loads(Path("configs/revision-final.json").read_text())


def test_final_search_budget_and_original_patient_protocol(config):
    original = json.loads(Path("configs/revision.json").read_text())
    validate_config(config)
    for key in (
        "tasks",
        "models",
        "objectives",
        "seed",
        "outer_folds",
        "inner_folds",
        "outer_repeats",
        "clinical_covariates",
        "calibration",
        "bootstrap_samples",
        "permutation_samples",
    ):
        assert config[key] == original[key]
    assert "fine_tuning" not in config
    screened = screen_candidates(config, 31, True)
    assert len(screened) == 35
    assert {c.n_freq for c in screened if c.stage == "frequency"} == {1, 3, 5, 10, 31}
    assert {c.selector for c in screened if c.stage == "selector"} == {
        "pca",
        "mutual_info",
    }
    assert len([c for c in screened if c.stage == "z_threshold"]) == 5
    records = [
        dict(candidate=c, objectives={goal: 0.5 for goal in config["objectives"]})
        for c in screened
    ]
    for objective in config["objectives"]:
        choices = tune_candidates(records, config, objective, 31)
        assert len(choices) == 34
        assert {c.model for c in choices} == {"svm", "lr", "rf"}
        assert len({c.mask for c in choices}) == 2
        assert len({c.n_freq for c in choices}) == 1
        assert all(c.normalization != "none" and c.cutoff >= 80 for c in choices)
    assert 35 + 4 * 34 + 3 + 3 == 177
    assert config["fold_workers"] * max(config["rf_jobs"], config["blas_threads"]) == 32


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
    bad["screen_defaults"]["normalization"] = "none"
    with pytest.raises(ValueError, match="screen_defaults.normalization"):
        validate_config(bad)


def test_progress_tracks_worker_and_phase_and_skips_stale_snapshots(tmp_path, capsys):
    import time
    from revision.progress import update_progress, heartbeat

    folder = tmp_path / "fold-checkpoints/MC/nested_cv_repeat0_fold2"
    config = {
        "progress_interval_seconds": 60,
        "_progress_path": str(folder / "progress.json"),
    }
    update_progress(
        config, "model-fit", candidates_completed=3, candidates_planned=20, inner_fold=2
    )
    state = json.loads((folder / "progress.json").read_text())
    assert state["pid"] > 0 and state["phase"] == "model-fit"
    heartbeat([("MC", (0, 2, "nested_cv"), tmp_path)], 1, 20)
    assert "MC/f2: model-fit candidate=3/20 inner=2" in capsys.readouterr().out
    heartbeat(
        [("MC", (0, 2, "nested_cv"), tmp_path)], 1, 20, since_epoch=time.time() + 1
    )
    assert "MC/f2" not in capsys.readouterr().out
    heartbeat([("MC", str(tmp_path))], 0, 4, label="reports")
    assert "reports=0/4" in capsys.readouterr().out


def test_review_exports_retain_parameters_and_honest_scope(tmp_path, config):
    from revision.final_selection import annotate

    config = dict(config, tasks=["MC"], outer_folds=2)
    folder = tmp_path / "MC"
    folder.mkdir()
    (folder / "splits.json").write_text(json.dumps([{}, {}]))
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
                repeat=0,
                outer_fold=fold,
                evaluation="nested_cv",
                inner_evaluation_seconds=1.5,
            )
            for fold in (0, 1)
        ]
    ).to_csv(folder / "inner_search.csv", index=False)
    annotate(tmp_path, config)
    table = pd.read_csv(tmp_path / "review_selected_parameters.csv")
    assert json.loads(table.iloc[0].candidate)["model"] == "svm"
    assert table.iloc[0].folds_available == table.iloc[0].folds_expected == 2
    assert len(pd.read_csv(tmp_path / "unavailable_results.csv")) == 1
    protocol = json.loads((tmp_path / "final_selection_protocol.json").read_text())
    assert protocol["task_fold_counts"] == {"MC": 2}
    assert "not independent confirmation" in protocol["interpretation"]


def test_final_synthetic_pipeline_exports_and_verifies_all_models(tmp_path):
    from run_revision import main
    from revision.verification import verify_results

    output = tmp_path / "synthetic-final"
    assert (
        main(
            [
                "--config",
                "configs/revision-final.json",
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
