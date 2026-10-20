"""Scientific regression checks: leakage boundaries, labels, missingness and endpoints."""

import json
import numpy as np
import pandas as pd
import pytest
from revision.data import load_dataset
from revision.features import (
    Candidate,
    FeatureBuilder,
    TrainScaler,
    patient_weights,
    circuit_features,
)
from revision.pipeline import group_splits, fit_predict
from revision.metrics import (
    scores,
    threshold_for_sensitivity,
    select_threshold,
    holm,
    cluster_indices,
)
from revision.analysis import aggregate_repeats, patient_frame, icc_absolute
from run_revision import synthetic_fixture


@pytest.fixture
def dataset(tmp_path):
    folder, patterns = synthetic_fixture(tmp_path / "fixture")
    return load_dataset(folder, patterns_path=patterns)


def test_raw_loader_preserves_complex_missing_entries(dataset):
    assert dataset.complex_available
    assert np.isnan(dataset.z).any()
    assert np.nanmean(np.abs(dataset.z)) > 1
    assert dataset.repeats.shape[-1] == 9
    assert dataset.table.sample_id.is_unique


def test_cancer_and_high_risk_have_explicit_positive_class(dataset):
    indices, y, classes = dataset.task("healthy_vs_cancer")
    assert classes == ("Healthy", "Cancer")
    assert (
        dataset.table.iloc[indices[y == 1]].raw_pathology == "invasive carcinoma"
    ).all()
    indices, y, classes = dataset.task("highrisk_vs_lowrisk")
    assert (
        y[dataset.table.iloc[indices].raw_pathology.to_numpy() == "moderate dysplasia"]
        == 1
    ).all()


def test_raw_label_disagreements_are_not_silently_accepted(tmp_path):
    np.savez(
        tmp_path / "site.npz",
        Zdata=np.ones((31, 3)),
        patient_no=1,
        patient_sample=1,
        LesionAssessment="moderate dysplasia",
        labels_MC="Non-malignant",
        Avg_Cutpats=0,
    )
    with pytest.raises(ValueError, match="conflicting"):
        load_dataset(tmp_path)


def test_no_frequency_coordinate_guess_for_nonstandard_dimensions(tmp_path):
    np.savez(
        tmp_path / "site.npz",
        Zdata=np.ones((4, 3)),
        patient_no=1,
        patient_sample=1,
        LesionAssessment="healthy",
        Avg_Cutpats=0,
    )
    with pytest.raises(ValueError, match="frequencies"):
        load_dataset(tmp_path)


def test_train_scaling_ignores_extreme_heldout_values_and_missing_entries():
    train = np.array([1.0, 3.0, np.nan, 5.0]).reshape(2, 1, 2, 1)
    scaler = TrainScaler().fit(train)
    mean = scaler.mean_.copy()
    fill = scaler.fill_.copy()
    result = scaler.transform(np.full((1, 1, 2, 1), 1e12))
    np.testing.assert_array_equal(mean, scaler.mean_)
    np.testing.assert_array_equal(fill, scaler.fill_)
    assert scaler.mean_.item() == 3
    assert np.isfinite(scaler.transform(np.full((1, 1, 2, 1), np.nan))).all()
    assert result.min() > 1000


def test_all_missing_training_columns_do_not_use_test_means():
    x = np.ones((4, 2, 3, 1))
    x[:, :, 0] = np.nan
    scaler = TrainScaler(imputation="median").fit(x)
    assert scaler.all_missing_columns_[:, 0].all()
    assert (scaler.fill_[:, 0] == 0).all()
    assert (scaler.transform(x)[:, :, 0] == 0).all()


def test_group_splits_never_separate_a_patients_sites(dataset):
    ix, y, _ = dataset.task("MC")
    groups = dataset.table.iloc[ix].patient_id.to_numpy()
    splits = group_splits(y, groups, 5, 77)
    seen = []
    for tr, te in splits:
        assert not set(groups[tr]) & set(groups[te])
        seen.extend(te)
        assert set(y[tr]) == set(range(4))
    assert sorted(seen) == list(range(len(y)))


def test_insufficient_class_patients_fails_instead_of_sample_splitting():
    with pytest.raises(ValueError, match="fewer than two patients"):
        group_splits(np.array([0, 0, 1, 1]), np.array(["a", "b", "c", "c"]), 2, 0)


def test_selected_frequencies_are_identities_not_subset_sizes():
    rng = np.random.default_rng(22)
    z = rng.normal(0, 0.1, (30, 7, 4))
    y = np.repeat([0, 1], 15)
    z[:, 5, :] += y[:, None] * 10
    c = Candidate(selector="anova", n_freq=1, normalization="none")
    b = FeatureBuilder(c, np.arange(1, 8) * 100, None).fit(z, y, np.arange(30))
    assert b.metadata()["frequency_indices_0based"] == [5]
    assert b.metadata()["frequencies_hz"] == [600]
    assert b.transform(z).shape == (30, 4)


def test_z_mask_is_frozen_from_training_only():
    z = np.broadcast_to([10.0, 20.0, 30.0, 40.0], (8, 3, 4)).copy()
    y = np.tile([0, 1], 4)
    b = FeatureBuilder(Candidate(mask="Zq:0.5"), [100, 1000, 10000], None).fit(
        z, y, np.arange(8)
    )
    assert b.z_threshold_ == 25
    np.testing.assert_array_equal(b.pattern_mask_, [True, True, False, False])
    heldout = z[:2].copy()
    heldout[:, :, 0] = 1e9
    b.transform(heldout)
    assert b.z_threshold_ == 25
    np.testing.assert_array_equal(b.pattern_mask_, [True, True, False, False])


def test_quality_rejection_is_an_abstention_not_dropped_test_patient(dataset):
    ix, y, _ = dataset.task("healthy_vs_cancer")
    groups = dataset.table.iloc[ix].patient_id.to_numpy()
    tr, te = group_splits(y, groups, 3, 0)[0]
    dataset.table.loc[ix[te[0]], "missing_percent"] = 90
    p, meta, _ = fit_predict(
        dataset,
        ix,
        y,
        Candidate(model="lr"),
        tr,
        te,
        0,
        {"clinical_covariates": [], "max_pca_rows": 10000},
    )
    assert p.shape == (len(te), 2)
    assert np.isnan(p[0]).all()
    assert set(meta["train_patients"]).isdisjoint(meta["test_patients"])


def test_patient_weights_equalize_total_contributions():
    groups = np.array(["a", "b", "b", "b"])
    w = patient_weights(groups)
    assert np.isclose(w[groups == "a"].sum(), w[groups == "b"].sum())


def test_named_binary_metrics_do_not_swap_sensitivity_and_specificity():
    y = np.array([1, 1, 1, 0, 0])
    p = np.array([[0.1, 0.9], [0.8, 0.2], [0.7, 0.3], [0.2, 0.8], [0.9, 0.1]])
    m, classes, cm = scores(y, p)
    assert m["sensitivity"] == pytest.approx(1 / 3)
    assert m["specificity"] == pytest.approx(0.5)
    assert m["npv"] == pytest.approx(1 / 3)
    assert m["precision"] == pytest.approx(0.5)
    assert m["ppv"] == m["precision"]
    assert cm.tolist() == [[1, 1], [2, 1]]
    assert m["brier"] == pytest.approx(np.mean((p[:, 1] - y) ** 2))


def test_requested_multiclass_outputs_are_macro_ovr_with_precision_ppv_alias():
    from revision.metrics import CLASSIFICATION_METRICS

    y = np.array([0, 0, 1, 1, 2, 2])
    p = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.6, 0.2],
            [0.1, 0.7, 0.2],
            [0.3, 0.1, 0.6],
            [0.6, 0.1, 0.3],
        ]
    )
    summary, perclass, _ = scores(y, p)
    assert set(CLASSIFICATION_METRICS) <= set(summary)
    for metric in (
        "auc",
        "f1",
        "precision",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
    ):
        assert summary[metric] == pytest.approx(perclass[metric].mean())
    np.testing.assert_array_equal(perclass.ppv, perclass.precision)


def test_corrected_cv_test_uses_overlap_and_patient_split_ratio():
    from scipy.stats import t
    from revision.metrics import corrected_resampled_test

    result = corrected_resampled_test([0.1, 0.2, -0.1, 0.3, 0], [80] * 5, [20] * 5)
    assert result["mean_difference"] == pytest.approx(0.1)
    assert result["sample_variance"] == pytest.approx(0.025)
    assert result["corrected_se"] == pytest.approx(np.sqrt(0.01125))
    assert result["degrees_of_freedom"] == 4
    assert result["p_value"] == pytest.approx(2 * t.sf(0.1 / np.sqrt(0.01125), 4))
    assert result["corrected_se"] > np.sqrt(0.025 / 5)


def test_corrected_cv_test_withholds_partial_and_degenerate_nonzero_comparisons():
    from revision.metrics import corrected_resampled_test

    assert np.isnan(
        corrected_resampled_test([0.1, np.nan, 0.2], [80] * 3, [20] * 3)["p_value"]
    )
    assert np.isnan(corrected_resampled_test([0.1] * 5, [80] * 5, [20] * 5)["p_value"])
    assert corrected_resampled_test([0] * 5, [80] * 5, [20] * 5)["p_value"] == 1


def test_postprocessing_refuses_outer_test_patients_before_fitting():
    from revision.pipeline import fit_training_rules

    with pytest.raises(ValueError, match="outer-test patients"):
        fit_training_rules(
            {"predictions": np.array([[0.8, 0.2], [0.2, 0.8]])},
            np.array([0, 1]),
            np.array(["p1", "p2"]),
            ["s1", "s2"],
            ["Low", "High"],
            {"calibration": True},
            test_groups=["p2"],
        )


def test_outer_fold_worker_count_respects_scheduler_cpu_budget(monkeypatch):
    import revision.parallel as parallel

    monkeypatch.setattr(parallel.os, "sched_getaffinity", lambda _: set(range(256)))
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "16")
    monkeypatch.delenv("PBS_NP", raising=False)
    monkeypatch.delenv("NSLOTS", raising=False)
    monkeypatch.setattr(parallel.Path, "is_file", lambda _: False)
    assert parallel.worker_count(20, 20, 4) == 4
    assert parallel.worker_count(20, 2, 4) == 2


def test_serial_and_parallel_outer_folds_and_resume_preserve_predictions(
    tmp_path, monkeypatch
):
    import multiprocessing
    from run_revision import main
    import revision.parallel as parallel

    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("Outer-fold process parity requires fork")
    # Two tiny LR workers exercise actual process isolation even on a quota-limited CI host.
    monkeypatch.setattr(
        parallel,
        "worker_count",
        lambda requested, jobs, threads=1: min(requested, jobs),
    )
    common = [
        "--synthetic",
        "--tasks",
        "healthy_vs_cancer",
        "--models",
        "lr",
        "--objectives",
        "auc",
        "--bootstrap-samples",
        "2",
        "--no-plots",
        "--rf-jobs",
        "1",
    ]
    serial, concurrent = tmp_path / "serial", tmp_path / "parallel"
    assert main([*common, "--fold-workers", "1", "--output", str(serial)]) == 0
    assert main([*common, "--fold-workers", "2", "--output", str(concurrent)]) == 0

    def predictions(root):
        frame = pd.read_csv(root / "healthy_vs_cancer/outer_predictions.csv")
        return frame.sort_values(
            ["repeat", "outer_fold", "stream", "sample_id"]
        ).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        predictions(serial),
        predictions(concurrent),
        check_exact=False,
        atol=1e-12,
        rtol=1e-12,
    )
    checkpoints = sorted(
        (concurrent / "fold-checkpoints/healthy_vs_cancer").glob("*/complete.json")
    )
    assert len(checkpoints) == 3
    original = {path: path.read_bytes() for path in checkpoints[1:]}
    checkpoints[0].unlink()
    (checkpoints[0].parent / "outer_predictions.csv").unlink()
    manifest = json.loads((concurrent / "run_manifest.json").read_text())
    manifest["status"] = "interrupted"
    (concurrent / "run_manifest.json").write_text(json.dumps(manifest))
    assert (
        main([*common, "--fold-workers", "2", "--output", str(concurrent), "--resume"])
        == 0
    )
    pd.testing.assert_frame_equal(
        predictions(serial),
        predictions(concurrent),
        check_exact=False,
        atol=1e-12,
        rtol=1e-12,
    )
    assert all(path.read_bytes() == value for path, value in original.items())
    from revision.verification import verify_results

    assert verify_results(concurrent)["checked_outer_folds"] == 3


def test_cv_statistics_keep_scheduled_pairs_and_precision_ppv_one_hypothesis(tmp_path):
    from revision.cv_statistics import export_cv_statistics, combine_corrected_tests

    folder = tmp_path / "healthy_vs_cancer"
    folder.mkdir()
    rows, splits = [], []
    patients = [f"p{i}" for i in range(20)]
    samples = [f"s{i}" for i in range(20)]
    for fold in range(5):
        test = list(range(fold * 4, fold * 4 + 4))
        train = sorted(set(range(20)) - set(test))
        splits.append(
            dict(
                evaluation="nested_cv",
                repeat=0,
                outer_fold=fold,
                train_patients=[patients[i] for i in train],
                test_patients=[patients[i] for i in test],
                train_sample_ids=[samples[i] for i in train],
                test_sample_ids=[samples[i] for i in test],
            )
        )
        for stream in ("baseline_lr", "auc_lr"):
            for i in test:
                truth = i % 2
                rows.append(
                    dict(
                        stream=stream,
                        evaluation="nested_cv",
                        repeat=0,
                        outer_fold=fold,
                        sample_id=samples[i],
                        patient_id=patients[i],
                        y_true=truth,
                        y_pred=truth,
                        covered=True,
                        p_0=0.1 if truth else 0.9,
                        p_1=0.9 if truth else 0.1,
                    )
                )
    config = {"models": ["lr"], "objectives": ["auc"]}
    frame = pd.DataFrame(rows)
    result = export_cv_statistics(frame, splits, ["Healthy", "Cancer"], config, folder)
    assert len(result) == 7
    assert result.n_scheduled_folds.eq(5).all()
    assert result.p_value.eq(1).all()
    adjusted = combine_corrected_tests(tmp_path, [folder.name])
    assert adjusted.n_unique_hypotheses.eq(6).all()
    assert adjusted.p_holm_all_unique_metrics.eq(1).all()
    # Missing common coverage in one fold must not produce a four-fold test.
    frame.loc[(frame.outer_fold == 2) & (frame.stream == "baseline_lr"), "covered"] = (
        False
    )
    result = export_cv_statistics(frame, splits, ["Healthy", "Cancer"], config, folder)
    assert result.n_valid_folds.eq(4).all()
    assert result.p_value.isna().all()


def test_confirmed_risk_groups_retain_moderate_and_benign(dataset):
    from revision.data import PATHOLOGY, RISK_CLASS

    expected = {
        "healthy": "Low risk",
        "normal": "Low risk",
        "benign": "Low risk",
        "mild dysplasia": "Low risk",
        "moderate dysplasia": "High risk",
        "severe dysplasia": "High risk",
        "cis": "High risk",
        "invasive carcinoma": "High risk",
    }
    for raw, label in expected.items():
        assert RISK_CLASS[PATHOLOGY[raw]] == label
    indices, y, classes = dataset.task("highrisk_vs_lowrisk")
    assert classes == ("Low risk", "High risk")
    assert set(y) == {0, 1}


def test_macro_auc_is_undefined_when_a_target_class_is_missing():
    m, _, _ = scores(np.array([0, 0]), np.array([[0.8, 0.1, 0.1], [0.6, 0.2, 0.2]]))
    assert np.isnan(m["macro_auc"])


def test_operating_thresholds_are_selected_from_training_predictions():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.6, 0.3, 0.7, 0.9])
    threshold = threshold_for_sensitivity(y, p, 1.0)
    assert threshold == 0.3
    assert (p[y == 1] >= threshold).all()
    threshold = select_threshold(y, p, 1.0, kind="specificity")
    assert (p[y == 0] < threshold).all()


def test_holm_controls_the_prespecified_family():
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert np.isnan(holm([np.nan])[0])


def test_holm_does_not_drop_unavailable_prespecified_comparisons():
    actual = holm([0.01, np.nan, 0.04])
    np.testing.assert_allclose(actual[[0, 2]], [0.03, 0.08])
    assert np.isnan(actual[1])


def test_bootstrap_resamples_whole_patients():
    groups = np.array(["a", "a", "b", "b", "b"])
    rng = np.random.default_rng(0)
    ix, ids = cluster_indices(groups, rng)
    for value in np.unique(ids):
        original = groups[ix[ids == value]]
        assert len(set(original)) == 1
        assert len(original) == sum(groups == original[0])


def test_repeat_predictions_are_not_treated_as_independent_patients():
    frame = pd.DataFrame(
        [
            dict(
                task="x",
                stream="s",
                objective="auc",
                model="lr",
                evaluation="nested_cv",
                repeat=r,
                sample_id="p:s",
                patient_id="p",
                y_true=1,
                y_pred=1,
                covered=True,
                p_0=0.1,
                p_1=0.9,
                raw_p_0=0.2,
                raw_p_1=0.8,
            )
            for r in range(10)
        ]
    )
    result = aggregate_repeats(frame, 2)
    assert len(result) == 1
    assert result.p_1.iloc[0] == pytest.approx(0.9)
    frame.loc[0, "covered"] = False
    assert not aggregate_repeats(frame, 2).covered.iloc[0]


def test_binary_patient_endpoint_and_mixed_multiclass_exclusion_are_explicit():
    frame = pd.DataFrame(
        dict(
            patient_id=["a", "a", "b"],
            sample_id=["a:1", "a:2", "b:1"],
            y_true=[0, 1, 0],
            y_pred=[0, 1, 0],
            covered=True,
            p_0=[0.9, 0.2, 0.8],
            p_1=[0.1, 0.8, 0.2],
            p_2=[0, 0, 0],
        )
    )
    p, excluded = patient_frame(frame, 2)
    assert p.set_index("patient_id").loc["a", "y_true"] == 1
    assert not excluded
    _, excluded = patient_frame(frame, 3)
    assert excluded == ["a"]


def test_icc_absolute_penalizes_systematic_repositioning_offset():
    x = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], float)
    assert icc_absolute(x) == pytest.approx(1.0)
    x[:, 2] += 10
    assert icc_absolute(x) < 1


def test_circuit_fit_retains_physical_frequencies_and_recovers_known_spectrum():
    f = np.geomspace(100, 100000, 20)
    z = 50 + 200 / (1 + (2j * np.pi * f * 1e-4) ** 0.8)
    vals = circuit_features(z[None, :, None], f)
    np.testing.assert_allclose(vals[0, 0, :4, 0], [50, 200, 1e-4, 0.8], rtol=0.02)


def test_revision_reporting_module_exposes_supported_commands():
    import subprocess, sys

    completed = subprocess.run(
        [sys.executable, "-m", "revision.report_cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "check" in completed.stdout
    assert "evaluation" in completed.stdout
    assert "tuning" in completed.stdout


@pytest.mark.parametrize(
    "selector", ["pca", "anova", "mutual_info", "sparse", "rfe", "permutation"]
)
def test_all_ranking_methods_fit_only_training_shapes(selector):
    rng = np.random.default_rng(2)
    z = np.abs(rng.normal(size=(20, 6, 3)))
    y = np.repeat([0, 1], 10)
    z[y == 1, 4, :] += 4
    b = FeatureBuilder(
        Candidate(selector=selector, n_freq=2), np.arange(1, 7) * 100, None, seed=0
    ).fit(z, y, np.arange(20))
    assert len(b.metadata()["frequency_indices_0based"]) == 2
    assert np.isfinite(b.transform(z[:2])).all()


def test_cluster_wald_rejects_singular_covariance_from_sparse_levels():
    from revision.analysis import cluster_wald

    rng = np.random.default_rng(41)
    frame = pd.DataFrame(
        dict(
            patient_id=np.repeat(np.arange(30), 2),
            quality=rng.normal(size=60),
            location=["common"] * 58 + ["rare"] * 2,
        )
    )
    result = cluster_wald(frame, "quality", ["location"])
    # A single rare-level patient has no residual cluster variance in that cell.
    if result["status"] == "computed":
        assert result["wald_chi2"] >= 0
        assert 0 <= result["p_value"] <= 1
    else:
        assert "covariance" in result["reason"]


def test_cluster_wald_regular_design_has_nonnegative_statistic():
    from revision.analysis import cluster_wald

    rng = np.random.default_rng(82)
    frame = pd.DataFrame(
        dict(
            patient_id=np.repeat(np.arange(40), 2),
            quality=rng.normal(size=80) + np.tile([0, 2], 40),
            location=np.tile(["a", "b"], 40),
        )
    )
    result = cluster_wald(frame, "quality", ["location"])
    assert result["status"] == "computed"
    assert result["wald_chi2"] >= 0
    assert result["p_value"] < 0.05


@pytest.mark.parametrize("classes", [2, 3])
def test_training_span_logistic_preserves_l2_fit_and_frozen_test_projection(classes):
    from revision.linear_span import TrainingSpanLogisticRegression
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(91)
    train = rng.normal(size=(36, 4200))
    test = rng.normal(size=(7, 4200))
    y = np.arange(36) % classes
    weights = rng.uniform(0.5, 2, 36)
    params = dict(C=0.3, tol=1e-9, max_iter=3000, random_state=0)
    direct = LogisticRegression(**params).fit(train, y, sample_weight=weights)
    projected = TrainingSpanLogisticRegression(**params).fit(
        train, y, sample_weight=weights
    )
    assert projected.optimized_dimensions_ <= len(train)
    np.testing.assert_allclose(
        projected.predict_proba(test), direct.predict_proba(test), rtol=2e-4, atol=2e-5
    )
    before = projected.predict_proba(test)
    projected.predict_proba(test * 1000)
    np.testing.assert_array_equal(before, projected.predict_proba(test))


@pytest.mark.parametrize("classes", [2, 3, 4])
def test_fast_resampling_metrics_match_sklearn_with_ties_and_weights(classes):
    from revision.metrics import weighted_auc, score_arrays
    from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, f1_score

    rng = np.random.default_rng(19)
    y = np.arange(80) % classes
    p = rng.integers(1, 8, (80, classes)).astype(float)
    p /= p.sum(1, keepdims=True)
    weights = rng.uniform(0.2, 2, 80)
    pred = p.argmax(1)
    result, perclass, cm = score_arrays(y, p, pred, weights)
    for k in range(classes):
        assert weighted_auc(y == k, p[:, k], weights) == pytest.approx(
            roc_auc_score(y == k, p[:, k], sample_weight=weights)
        )
    assert result["accuracy"] == pytest.approx(
        accuracy_score(y, pred, sample_weight=weights)
    )
    assert result["macro_f1"] == pytest.approx(
        f1_score(y, pred, average="macro", sample_weight=weights)
    )
    assert result["log_loss"] == pytest.approx(log_loss(y, p, sample_weight=weights))
    assert result["micro_auc"] == pytest.approx(
        roc_auc_score(
            np.eye(classes)[y].ravel(),
            p.ravel(),
            sample_weight=np.repeat(weights, classes),
        )
    )


def test_coverage_bootstrap_preserves_pre_abstention_weights():
    from revision.metrics import bootstrap_metrics

    # One patient lost two of three sites to abstentions; retain their original weight.
    y = np.array([0, 1, 0, 1])
    p = np.array([[0.8, 0.2], [0.4, 0.6], [0.1, 0.9], [0.7, 0.3]])
    groups = np.array(["a", "b", "c", "c"])
    weights = np.array([1 / 3, 1, 1 / 2, 1 / 2])
    first = bootstrap_metrics(
        y, p, p.argmax(1), groups, n_boot=20, seed=0, base_weights=weights
    )
    second = bootstrap_metrics(
        y, p, p.argmax(1), groups, n_boot=20, seed=0, base_weights=weights * 10
    )
    keep = ~first.metric.isin(["tp", "fp", "tn", "fn"])
    np.testing.assert_allclose(
        first.loc[keep, ["ci_low", "ci_high"]],
        second.loc[keep, ["ci_low", "ci_high"]],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        first.loc[~keep, ["ci_low", "ci_high"]] * 10,
        second.loc[~keep, ["ci_low", "ci_high"]],
        equal_nan=True,
    )


@pytest.mark.parametrize("kernel", ["linear", "rbf", "poly", "sigmoid"])
@pytest.mark.parametrize("classes", [2, 3])
def test_vectorized_svc_matches_standard_kernels_and_label_order(kernel, classes):
    from revision.kernel_svm import VectorizedSVC
    from sklearn.svm import SVC

    rng = np.random.default_rng(131)
    x = rng.normal(size=(36, 90))
    test = rng.normal(size=(7, 90))
    y = np.arange(36) % classes
    weights = rng.uniform(0.5, 1.5, 36)
    params = dict(
        kernel=kernel, C=0.3, class_weight="balanced", random_state=12, tol=1e-8
    )
    expected = SVC(**params).fit(x, y, sample_weight=weights)
    actual = VectorizedSVC(**params).fit(x, y, sample_weight=weights)
    np.testing.assert_array_equal(actual.classes_, expected.classes_)
    np.testing.assert_allclose(
        actual.decision_function(test),
        expected.decision_function(test),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("status", ["running", "interrupted", "failed", "completed"])
def test_preflight_preserves_existing_analysis_manifest(tmp_path, status):
    from run_revision import main

    root = tmp_path / "existing-analysis"
    root.mkdir()
    path = root / "run_manifest.json"
    original = (
        json.dumps(
            dict(status=status, synthetic=False, scientific_evidence="retain"), indent=2
        )
        + "\n"
    )
    path.write_text(original)
    with pytest.raises(ValueError, match="already contains a run"):
        main(["--preflight", "--output", str(root)])
    assert path.read_text() == original
    assert sorted(p.name for p in root.iterdir()) == ["run_manifest.json"]


def test_preflight_cannot_be_combined_with_resume(tmp_path):
    from run_revision import main

    root = tmp_path / "must-not-create"
    with pytest.raises(ValueError, match="cannot be combined"):
        main(["--preflight", "--resume", "--output", str(root)])
    assert not root.exists()


def test_required_matlab_source_fails_before_loading_or_fitting(tmp_path):
    from run_revision import main

    root = tmp_path / "missing-source"
    assert (
        main(
            [
                "--prepare-source",
                "--source-mat",
                str(tmp_path / "absent.mat"),
                "--preflight",
                "--output",
                str(root),
            ]
        )
        == 2
    )
    manifest = json.loads((root / "run_manifest.json").read_text())
    assert manifest["status"] == "preflight_failed"
    assert "--prepare-source requires" in manifest["error"]
    assert not (root / "all_metrics.csv").exists()


def test_resume_reuses_complete_folds_and_reproduces_outer_predictions(tmp_path):
    from run_revision import main

    root = tmp_path / "synthetic-resume"
    args = [
        "--synthetic",
        "--tasks",
        "healthy_vs_cancer",
        "--output",
        str(root),
        "--no-plots",
        "--bootstrap-samples",
        "2",
    ]
    assert main(args) == 0
    folder = root / "healthy_vs_cancer"
    expected = pd.read_csv(folder / "outer_predictions.csv")
    # Simulate stopping after two complete folds; incomplete work is not reused.
    for name in ["splits.json", "selections.json"]:
        p = folder / name
        rows = json.loads(p.read_text())
        p.write_text(json.dumps([v for v in rows if v["outer_fold"] < 2]))
    for name in ["outer_predictions.csv", "inner_search.csv"]:
        p = folder / name
        frame = pd.read_csv(p)
        frame[frame.outer_fold < 2].to_csv(p, index=False)
    p = root / "run_manifest.json"
    manifest = json.loads(p.read_text())
    manifest["status"] = "interrupted"
    p.write_text(json.dumps(manifest))
    assert main([*args, "--resume"]) == 0
    actual = pd.read_csv(folder / "outer_predictions.csv")
    columns = ["stream", "repeat", "outer_fold", "sample_id"]
    expected = expected.sort_values(columns).reset_index(drop=True)
    actual = actual.sort_values(columns).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        actual, expected, check_exact=False, atol=1e-12, rtol=1e-12
    )
    assert json.loads((root / "run_manifest.json").read_text())["resume_history"]
    from revision.verification import verify_results

    assert verify_results(root, verify_inputs=True)["checked_outer_folds"] == 3
    # Existing files alone cannot make a corrupted patient partition valid.
    p = folder / "splits.json"
    splits = json.loads(p.read_text())
    splits[0]["test_patients"].append(splits[0]["train_patients"][0])
    p.write_text(json.dumps(splits))
    with pytest.raises(ValueError, match="outer patient overlap"):
        verify_results(root)
    # Two rejected resume attempts must never bless a new input inventory.
    p = root / "run_manifest.json"
    manifest = json.loads(p.read_text())
    manifest["status"] = "interrupted"
    name = next(iter(manifest["input_hashes"]))
    manifest["input_hashes"][name] = "different-original-input"
    original_inventory = dict(manifest["input_hashes"])
    p.write_text(json.dumps(manifest))
    for _ in range(2):
        assert main([*args, "--resume"]) == 2
        assert json.loads(p.read_text())["input_hashes"] == original_inventory


def test_fold_population_counts_preserve_classes_sites_and_quality(dataset, tmp_path):
    from revision.analysis import export_split_populations

    ix, y, classes = dataset.task("MC")
    groups = dataset.table.iloc[ix].patient_id.to_numpy()
    tr, te = group_splits(y, groups, 3, 0)[0]
    inner = group_splits(y[tr], groups[tr], 2, 1)
    split = dict(
        evaluation="nested_cv",
        repeat=0,
        outer_fold=0,
        train_patients=sorted(set(groups[tr])),
        test_patients=sorted(set(groups[te])),
        inner=[
            dict(
                train_patients=sorted(set(groups[tr][a])),
                validation_patients=sorted(set(groups[tr][b])),
            )
            for a, b in inner
        ],
    )
    (tmp_path / "splits.json").write_text(json.dumps([split]))
    export_split_populations(
        tmp_path, dataset, "MC", {"cutoffs": [10, 60], "default_cutoff": 60}
    )
    frame = pd.read_csv(tmp_path / "split_population_counts.csv")
    test = frame[
        (frame.partition == "outer_test")
        & (frame.location == "All locations")
        & (frame.cutoff == 60)
    ]
    assert set(test.class_name) == set(classes)
    assert test.n_sites.sum() == len(te)
    assert test.n_retained_sites.sum() == sum(
        dataset.table.iloc[ix[te]].missing_percent < 60
    )
    # Detailed reports preserve patients as clusters and raw pathology strata.
    from revision.demographics import export_cv_demographics

    export_cv_demographics(
        tmp_path, dataset, "MC", {"cutoffs": [10, 60], "default_cutoff": 60}
    )
    detail = pd.read_csv(tmp_path / "cv_population_summary.csv")
    detail = detail[
        (detail.partition == "outer_test") & (detail.population == "unfiltered")
    ]
    assert detail.n_samples.iloc[0] == len(te)
    assert detail.n_patients.iloc[0] == len(set(groups[te]))
    raw = pd.read_csv(tmp_path / "cv_pathology_breakdown.csv")
    raw = raw[(raw.partition == "outer_test") & (raw.population == "unfiltered")]
    assert raw.n_samples.sum() == len(te)
    availability = pd.read_csv(tmp_path / "cv_demographics_availability.csv")
    assert not availability[
        availability.variable == "smoking"
    ].available_in_source.any()


def test_per_class_cluster_intervals_keep_explicit_class_identity():
    from revision.metrics import bootstrap_metrics

    y = np.tile(np.arange(3), 6)
    p = np.eye(3)[y] * 0.8 + 0.2 / 3
    p /= p.sum(1, keepdims=True)
    summary, perclass = bootstrap_metrics(
        y, p, p.argmax(1), np.repeat(np.arange(6), 3), n_boot=10, return_perclass=True
    )
    assert set(perclass.class_index) == {0, 1, 2}
    assert (perclass[perclass.metric == "sensitivity"].ci_low == 1).all()
    assert (perclass[perclass.metric == "auc"].ci_low == 1).all()


def test_parallel_forest_training_preserves_trees_and_deterministic_predictions():
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(140)
    x = rng.normal(size=(70, 150))
    y = np.arange(70) % 3
    weights = rng.uniform(0.5, 2, 70)
    test = rng.normal(size=(13, 150))
    params = dict(
        n_estimators=25,
        max_features=0.7,
        max_depth=8,
        random_state=17,
        class_weight="balanced",
    )
    serial = RandomForestClassifier(n_jobs=1, **params).fit(x, y, sample_weight=weights)
    parallel = RandomForestClassifier(n_jobs=4, **params).fit(
        x, y, sample_weight=weights
    )
    for a, b in zip(serial.estimators_, parallel.estimators_):
        np.testing.assert_array_equal(a.tree_.feature, b.tree_.feature)
        np.testing.assert_array_equal(a.tree_.threshold, b.tree_.threshold)
    parallel.n_jobs = 1
    np.testing.assert_array_equal(
        parallel.predict_proba(test), serial.predict_proba(test)
    )
