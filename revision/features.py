"""Train-fitted transforms with explicit saved frequency and IIVV coordinates."""

from dataclasses import asdict, dataclass
from functools import lru_cache
import contextlib
import io
import warnings
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from scipy.special import expit, softmax

MASKS = {
    "All": "all",
    "Long a+": "vv_long_across",
    "Long a+ ext.": "vv_long_across_extended",
    "Med. a+ ext.": "vv_medium_across_extended",
    "Skip1 close": "vv_distance_close_skip_1",
    "Adj. close": "adjacent_close",
    "Med. adj.": "vv_distance_medium_adj",
    "Adj. far": "adjacent_far",
    "Skip1 medium": "skip1_medium",
    "Skip1 far": "skip1_far",
    "Opp. close": "opposite_close",
    "Opp. medium": "opposite_medium",
    "Opp. far": "opposite_far",
}


def finite_mean(x, axis=None):
    good = np.isfinite(x)
    count = good.sum(axis=axis)
    return np.divide(
        np.where(good, x, 0).sum(axis=axis),
        count,
        out=np.full(np.shape(count), np.nan, dtype=np.result_type(x, float)),
        where=count > 0,
    )


def patient_weights(groups):
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    w = 1 / counts[inverse].astype(float)
    return w * len(w) / w.sum()


class TrainScaler:
    """NaN-aware frequency/global scaling and fixed-column mean/median/zero filling.

    Axes are (sites, frequencies, patterns, components); statistics never see test data.
    Columns entirely missing in training get a fixed 0 fill, not a test-derived value.
    """

    def __init__(self, mode="freq", imputation="mean"):
        self.mode, self.imputation = mode, imputation

    def fit(self, x):
        axes = (
            (0, 2)
            if self.mode == "freq"
            else (0, 1, 2) if self.mode == "global" else ()
        )
        if axes:
            self.mean_ = finite_mean(x, axes)
            self.mean_ = (
                self.mean_[None, :, None, :]
                if self.mode == "freq"
                else self.mean_[None, None, None, :]
            )
            variance = finite_mean((x - self.mean_) ** 2, axes)
            std = np.sqrt(variance)
            self.scale_ = (
                std[None, :, None, :]
                if self.mode == "freq"
                else std[None, None, None, :]
            )
            self.mean_ = np.nan_to_num(self.mean_, nan=0)
            self.scale_ = np.where(
                np.isfinite(self.scale_) & (self.scale_ > 1e-12), self.scale_, 1
            )
        else:
            self.mean_, self.scale_ = 0.0, 1.0
        scaled = (x - self.mean_) / self.scale_
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.fill_ = (
                np.nanmedian(np.where(np.isfinite(scaled), scaled, np.nan), axis=0)
                if self.imputation == "median"
                else finite_mean(scaled, 0)
            )
        if self.imputation == "zero":
            self.fill_ = np.zeros(x.shape[1:])
        self.fill_ = np.nan_to_num(self.fill_, nan=0)
        self.all_missing_columns_ = (~np.isfinite(scaled)).all(axis=0)
        return self

    def transform(self, x):
        scaled = (x - self.mean_) / self.scale_
        return np.where(np.isfinite(scaled), scaled, self.fill_[None])


@lru_cache(maxsize=32)
def _geometry(name, shape, raw):
    patterns = np.frombuffer(raw, dtype=np.int64).reshape(shape)
    if name == "All":
        return np.ones(len(patterns), dtype=bool)
    # The reviewed geometry implementation was moved unchanged into this package.
    from .geometry import Masking

    mask = Masking(patterns, plot=False, prints=False)
    with contextlib.redirect_stdout(io.StringIO()):
        mask.IIVV_mask(MASKS[name], plot=False)
    result = np.asarray(mask.mask, dtype=bool)
    if result.shape != (len(patterns),) or not result.any():
        raise ValueError(f"Geometry {name} selects no matching coordinates")
    return result


@dataclass(frozen=True)
class Candidate:
    stage: str = "tuning"
    model: str = "svm"
    n_freq: int = 0  # 0 means every physical frequency
    mask: str = "All"
    selector: str = "pca"
    representation: str = "magnitude"
    cutoff: float = 60.0
    normalization: str = "freq"
    imputation: str = "mean"
    class_weight: str | None = None
    patient_weighted: bool = True
    params_json: str = "{}"

    def dictionary(self):
        return asdict(self)


def components(z, representation):
    if representation == "magnitude":
        return np.abs(z)[..., None]
    if representation == "magnitude_phase":
        return np.stack([np.abs(z), np.angle(z)], axis=-1)
    if representation == "missingness":
        return (~np.isfinite(z))[..., None].astype(float)
    raise ValueError(f"Unsupported representation {representation}")


def rank_frequencies(x, y, method, seed, max_pca_rows=100000):
    """Supervised baselines use one mean-over-patterns value per site/frequency.

    PCA uses rows of training-site/IIVV spectra, matching the legacy unsupervised
    heuristic. Row subsampling is deterministic and confined to training data.
    """
    nfreq = x.shape[1]
    if method == "all":
        return np.arange(nfreq), np.ones(nfreq)
    if method == "pca":
        matrix = x[..., 0].transpose(0, 2, 1).reshape(-1, nfreq)
        if len(matrix) > max_pca_rows:
            matrix = matrix[
                np.random.default_rng(seed).choice(
                    len(matrix), max_pca_rows, replace=False
                )
            ]
        pca = PCA(n_components=min(10, *matrix.shape), svd_solver="full").fit(matrix)
        scores = (pca.components_**2).sum(axis=0)
    elif method == "mutual_info":
        summary = x[..., 0].mean(axis=2)
        scores = mutual_info_classif(summary, y, random_state=seed)
    else:
        raise ValueError(f"Unknown frequency selector {method}")
    scores = np.nan_to_num(
        scores, nan=-np.inf, posinf=np.finfo(float).max, neginf=-np.inf
    )
    return np.argsort(-scores, kind="stable"), scores


class FeatureBuilder:
    def __init__(self, candidate, frequencies, patterns, seed=0, max_pca_rows=100000):
        self.candidate, self.frequencies, self.patterns = (
            candidate,
            np.asarray(frequencies),
            patterns,
        )
        self.seed, self.max_pca_rows = seed, max_pca_rows

    def fit(self, z, y, groups):
        c = self.candidate
        if c.mask.startswith("Zq:"):
            # Quality-eligible training patients alone define the raw 100-Hz aggregate.
            ref = int(np.argmin(np.abs(self.frequencies - 100)))
            if not np.isclose(self.frequencies[ref], 100):
                raise ValueError(
                    "Impedance-threshold masks require an actual 100-Hz coordinate"
                )
            w = patient_weights(groups)[:, None]
            vals = np.abs(z[:, ref, :])
            valid = np.isfinite(vals)
            mean = np.divide(
                np.where(valid, vals * w, 0).sum(0),
                (valid * w).sum(0),
                out=np.full(z.shape[2], np.nan),
                where=(valid * w).sum(0) > 0,
            )
            self.z_threshold_ = float(np.nanquantile(mean, float(c.mask.split(":")[1])))
            self.pattern_mask_ = np.isfinite(mean) & (mean <= self.z_threshold_)
        elif c.mask == "All":
            self.pattern_mask_ = np.ones(z.shape[2], dtype=bool)
        else:
            if self.patterns is None:
                raise ValueError("Geometry masks require verified IIVV coordinates")
            a = np.ascontiguousarray(self.patterns, dtype=np.int64)
            self.pattern_mask_ = _geometry(c.mask, a.shape, a.tobytes()).copy()
        if not self.pattern_mask_.any():
            raise ValueError("No patterns selected from training data")
        x = self._raw(z)
        self.scaler_ = TrainScaler(c.normalization, c.imputation).fit(x)
        scaled = self.scaler_.transform(x)
        self.ranking_, self.scores_ = rank_frequencies(
            scaled, y, c.selector, self.seed, self.max_pca_rows
        )
        n = c.n_freq or scaled.shape[1]
        self.selected_ = self.ranking_[:n]
        return self

    def _raw(self, z):
        z = z[:, :, self.pattern_mask_]
        return components(z, self.candidate.representation)

    def transform(self, z):
        x = self.scaler_.transform(self._raw(z))[:, self.selected_]
        return np.ascontiguousarray(x.reshape(len(x), -1))

    def metadata(self):
        used = self.selected_
        return dict(
            frequency_indices_0based=used.tolist(),
            frequencies_hz=self.frequencies[used].tolist(),
            frequency_ranking_0based=self.ranking_.tolist(),
            ranking_scores=self.scores_.tolist(),
            pattern_indices_0based=np.flatnonzero(self.pattern_mask_).tolist(),
            z_threshold_ohm=getattr(self, "z_threshold_", None),
            all_missing_training_columns=int(self.scaler_.all_missing_columns_.sum()),
        )


def make_model(candidate, seed, rf_jobs=4, lr_max_iter=5000):
    import json

    c = candidate
    params = json.loads(c.params_json)
    common = dict(class_weight=c.class_weight, random_state=seed)
    if c.model == "lr":
        from .linear_span import TrainingSpanLogisticRegression

        params.setdefault("max_iter", lr_max_iter)
        return TrainingSpanLogisticRegression(**common, **params)
    if c.model == "rf":
        return (
            RandomForestClassifier(n_estimators=100, n_jobs=rf_jobs, **common, **params)
            if "n_estimators" not in params
            else RandomForestClassifier(n_jobs=rf_jobs, **common, **params)
        )
    if c.model == "svm":
        # Patient-grouped OOF calibration is performed by the outer-training loop.
        from .kernel_svm import VectorizedSVC

        return VectorizedSVC(probability=False, **common, **params)
    raise ValueError(c.model)


def aligned_probabilities(model, x, nclasses):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(x)
    else:
        d = model.decision_function(x)
        p = (
            np.column_stack([expit(-d), expit(d)])
            if d.ndim == 1
            else softmax(d, axis=1)
        )
    result = np.zeros((len(x), nclasses))
    result[:, model.classes_.astype(int)] = p
    return result
