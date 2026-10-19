"""An L2 logistic fit in the training-row span, preserving the model objective."""
import copy
import numpy as np
from sklearn.linear_model import LogisticRegression


class TrainingSpanLogisticRegression:
    """Avoid optimizing hundreds of thousands of redundant primal coefficients.

    Every L2-penalized optimum lies in the span of training feature rows.
    A training Gram eigendecomposition provides an orthonormal basis of that
    span; its Euclidean penalty equals the original coefficient penalty. Only
    numerical null directions are discarded. The intercept remains unpenalized.
    This is internal solver algebra, not a new selected acquisition subset.
    """
    def __init__(self, **parameters):
        self.parameters=parameters

    def fit(self,x,y,sample_weight=None):
        x=np.asarray(x,dtype=float)
        self.n_features_in_=x.shape[1]
        self.used_training_span_=False
        if x.shape[1]>max(4096,4*len(x)):
            gram=x@x.T
            eigenvalues,eigenvectors=np.linalg.eigh((gram+gram.T)/2)
            tolerance=np.finfo(float).eps*max(1,len(x))*max(float(eigenvalues.max()),1.)*4
            keep=eigenvalues>tolerance
            if keep.any():
                self.used_training_span_=True
                self.gram_basis_=eigenvectors[:,keep]/np.sqrt(eigenvalues[keep])
                transformed=gram@self.gram_basis_
            else:
                transformed=np.zeros((len(x),1))
                self.used_training_span_=True
                self.gram_basis_=np.zeros((len(x),1))
        else:
            transformed=x
        self.model_=LogisticRegression(**self.parameters).fit(transformed,y,sample_weight=sample_weight)
        self.classes_=self.model_.classes_
        self.optimized_dimensions_=transformed.shape[1]
        self.prediction_model_=copy.copy(self.model_)
        if self.used_training_span_:
            # Re-express the fit in original coordinates for ordinary fast
            # inference; do not retain all training spectra in the model.
            self.prediction_model_.coef_=(self.model_.coef_@self.gram_basis_.T)@x
            self.prediction_model_.n_features_in_=self.n_features_in_
        return self

    def predict_proba(self,x):
        return self.prediction_model_.predict_proba(x)
