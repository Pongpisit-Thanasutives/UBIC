from sklearn import __version__ as SKLEARN_VERSION
print("Sklearn's version:", SKLEARN_VERSION)
from sklearn.linear_model import LinearRegression as SkLinearRegression
from scipy import stats
import numpy as np

class SKOLS(SkLinearRegression):
    def __init__(self, fit_intercept=False, normalize=False, copy_X=True, n_jobs=1, positive=False):                                                                                                                             
        if int(SKLEARN_VERSION.split('.')[1]) > 1:
            super(SKOLS, self).__init__(fit_intercept=fit_intercept, 
                                        copy_X=copy_X, 
                                        n_jobs=n_jobs, 
                                        positive=positive)
            self.normalize = normalize
            if not fit_intercept:
                self.normalize = False
        else:
            super(SKOLS, self).__init__(fit_intercept=fit_intercept, 
                                        normalize=normalize,
                                        copy_X=copy_X, 
                                        n_jobs=n_jobs, 
                                        positive=positive)
    def fit(self, X, y, n_jobs=1):
        if int(SKLEARN_VERSION.split('.')[1]) > 1:
            if self.normalize: 
                X_mean = X.mean(axis=0)
                X_scale = X-X_mean
                l2norm = np.linalg.norm(X_scale, ord=2, axis=0)
                X_scale = X_scale/l2norm
            else: 
                X_scale = X
                
            self = super(SKOLS, self).fit(X_scale, y, n_jobs)
            
            if self.normalize: 
                self.coef_ /= l2norm
                self.intercept_ += (-X_mean*self.coef_).sum()

        else:
            self = super(SKOLS, self).fit(X, y, n_jobs)
            
        return self

class PLinearRegression(SkLinearRegression):
    """
    LinearRegression class after sklearn's, but calculate t-statistics
    and p-values for model parameters (betas).
    """
    def __init__(self, fit_intercept=False, normalize=False, copy_X=True, n_jobs=1, positive=False):
        if int(SKLEARN_VERSION.split('.')[1]) < 2:
            super(PLinearRegression, self).__init__(fit_intercept=fit_intercept,
                                                    normalize=normalize,
                                                    copy_X=copy_X,
                                                    n_jobs=n_jobs,
                                                    positive=positive)
        else:
            super(PLinearRegression, self).__init__(fit_intercept=fit_intercept,
                                                    copy_X=copy_X,
                                                    n_jobs=n_jobs,
                                                    positive=positive)
            if normalize:
                print(f"Since sklearn's version {SKLEARN_VERSION} is being used, the normalize arg has no effect.")

        self.se = None
        self.t = None
        self.p = None
        self.feature_importances_ = None

    def fit(self, X, y, n_jobs=1):
        self = super(PLinearRegression, self).fit(X, y, n_jobs)

        sse = np.sum((self.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
        self.se = np.array([np.sqrt(np.diagonal(sse * np.linalg.inv(np.dot(X.T, X))))])

        self.t = self.coef_ / self.se
        self.p = 2 * (1 - stats.t.cdf(np.abs(self.t), y.shape[0] - X.shape[1]))

        self.se = self.se.ravel()
        self.t = self.t.ravel()
        self.p = self.p.ravel()

        self.feature_importances_ = 1-self.p
        self.feature_importances_ = self.feature_importances_/self.feature_importances_.sum()

        return self

### For nPIML ###
#class MLinearRegression(SkLinearRegression):
#    """
#    LinearRegression class after sklearn's LinearRegression
#    """
#    def __init__(self, fit_intercept=False, normalize=False, copy_X=True, n_jobs=1, positive=False):
#        if int(SKLEARN_VERSION.split('.')[1]) < 2:
#            super(MLinearRegression, self).__init__(fit_intercept=fit_intercept,
#                                                    normalize=normalize,
#                                                    copy_X=copy_X,
#                                                    n_jobs=n_jobs,
#                                                    positive=positive)
#        else:
#            super(MLinearRegression, self).__init__(fit_intercept=fit_intercept,
#                                                    copy_X=copy_X,
#                                                    n_jobs=n_jobs,
#                                                    positive=positive)
#            if normalize:
#                print(f"Since sklearn's version {SKLEARN_VERSION} is being used, the normalize arg has no effect.")
#            self.normalize = normalize
#
#        self.feature_importances_ = None
#
#    def fit(self, X, y, n_jobs=1):
#        self = super(MLinearRegression, self).fit(X, y, n_jobs)
#        n_cols = X.shape[1]
#        ref_mse = ((y-self.predict(X))**2).mean()
#        scores = []
#        if n_cols > 1:
#            for j in range(n_cols):
#                X_tmp = X[:, list(range(j))+list(range(j+1, n_cols))]
#                sub_model = super(MLinearRegression, self).fit(X_tmp, y, n_jobs)
#                sub_model_mse = ((y-sub_model.predict(X_tmp))**2).mean()
#                scores.append(sub_model_mse-ref_mse)
#        else: 
#            for j in range(n_cols):
#                scores.append(1.0)
#        self.feature_importances_ = np.array(scores)
#        return self

class MLinearRegression(SKOLS):
    """
    LinearRegression class after sklearn's LinearRegression
    """
    def __init__(self, fit_intercept=False, normalize=False, copy_X=True, n_jobs=1, positive=False):
        super(MLinearRegression, self).__init__(fit_intercept=fit_intercept,
                                                normalize=normalize,
                                                copy_X=copy_X,
                                                n_jobs=n_jobs,
                                                positive=positive)
        self.feature_importances_ = None

    def fit(self, X, y, n_jobs=1):
        self = super(MLinearRegression, self).fit(X, y, n_jobs)
        n_cols = X.shape[1]
        ref_mse = ((y-self.predict(X))**2).mean()
        scores = []
        if n_cols > 1:
            for j in range(n_cols):
                X_tmp = X[:, list(range(j))+list(range(j+1, n_cols))]
                sub_model = super(MLinearRegression, self).fit(X_tmp, y, n_jobs)
                sub_model_mse = ((y-sub_model.predict(X_tmp))**2).mean()
                scores.append(sub_model_mse-ref_mse)
        else: 
            for j in range(n_cols):
                scores.append(1.0)
        self.feature_importances_ = np.array(scores)
        return self
