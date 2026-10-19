"""Vectorized standard SVC kernels; identical training-only kernel definitions."""
import numpy as np
from sklearn.svm import SVC


class VectorizedSVC:
    """Use BLAS for Gram matrices instead of repeated high-dimensional C dot loops.

    The kernels, C, sample/class weights and libsvm optimization are unchanged.
    Kernel parameters are derived exclusively from training features. This is a
    numerical implementation choice, not a kernel search or feature reduction.
    """
    def __init__(self, **parameters):
        self.parameters=parameters

    def _kernel(self,x):
        product=np.asarray(x)@self.training_features_.T
        if self.kernel_=='linear':return product
        if self.kernel_=='rbf':
            distance=(np.sum(np.asarray(x)**2,axis=1)[:,None]+self.training_norm_[None,:]-2*product)
            return np.exp(-self.gamma_*np.maximum(distance,0))
        if self.kernel_=='poly':return (self.gamma_*product+self.coef0_)**self.degree_
        if self.kernel_=='sigmoid':return np.tanh(self.gamma_*product+self.coef0_)
        raise ValueError(f'Unsupported vectorized kernel: {self.kernel_}')

    def fit(self,x,y,sample_weight=None):
        parameters=dict(self.parameters);self.kernel_=parameters.pop('kernel','rbf')
        gamma=parameters.pop('gamma','scale');self.degree_=parameters.pop('degree',3);self.coef0_=parameters.pop('coef0',0.)
        self.training_features_=np.asarray(x,float)
        self.training_norm_=np.sum(self.training_features_**2,axis=1)
        variance=float(self.training_features_.var())
        self.gamma_=((1./(x.shape[1]*variance) if variance else 1.) if gamma=='scale' else
                     1./x.shape[1] if gamma=='auto' else float(gamma))
        self.model_=SVC(kernel='precomputed',**parameters)
        self.model_.fit(self._kernel(x),y,sample_weight=sample_weight)
        self.classes_=self.model_.classes_
        self.n_features_in_=x.shape[1]
        return self

    def decision_function(self,x):
        return self.model_.decision_function(self._kernel(x))
