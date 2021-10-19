import numpy as np
import scipy.stats
from scipy import stats
from copy import copy
import warnings

from si.data import Dataset


class VarianceThreshold:
    """Eliminate features in which variance is lower than threshold"""

    def __init__(self, threshold=0):
        if threshold < 0:
            warnings.warn("The threshold must be a non negative number")
            threshold = 0
        self.threshold = threshold

    def fit(self, dataset):
        X = dataset.X
        self._var = np.var(X, axis=0)

    def transform(self, dataset, inline=False):
        X, X_names = copy(dataset.X), copy(dataset.xnames)

        # Needs to grab both columns and data
        selection_list = self._var > self.threshold
        x = X[:, selection_list]
        x_names = [X_names[index] for index in range(dataset.getNumFeatures()) if selection_list[index]]

        if inline:
            dataset.X = x
            dataset.xnames = x_names
            return dataset
        else:
            return Dataset(x, copy(dataset.Y), x_names, copy(dataset.yname))

    def fit_transform(self, dataset, inline=False):
        self.fit(dataset)
        return self.transform(dataset, inline=inline)


class KBest:
    """
    função de test:
        - exemplo q quadrado, Fscore / ANOVA -> ignora Pvalue e só usa os F
    nº de features ( K )

    Implementar F regression e F classification
    """

    def __init__(self, score_func, k):
        available_funcs = ("f_classif", "f_regression")
        if score_func not in available_funcs:
            raise Exception(f"Score function not available, please choose between: {', '.join(available_funcs)}.")
        elif score_func == "f_classif":
            self._func = f_classif
        else:
            self._func = f_regression

        if k <= 0:
            raise Exception(
                "The K value provided is smaller or equal than 0, this means no features could be selected.")
        else:
            self.k = k

    def fit(self, dataset):
        self.F, self.p = self._func(dataset)

    def transform(self, dataset, inline=False):
        X, X_names = copy(dataset.X), copy(dataset.xnames)

        if self.k > X.shape[1]:
            raise warnings.warn(
                "The k value provided is equal or greater than the number of features available. All features will be selected")

        # Needs to grab both columns and data
        selection_list = np.argsort(self.F)[-10:]

        x = X[:, selection_list]
        x_names = [X_names[index] for index in selection_list]

        if inline:
            dataset.X = x
            dataset.xnames = x_names
            return dataset
        else:
            return Dataset(x, copy(dataset.Y), x_names, copy(dataset.yname))

    def fit_transform(self, dataset, inline=False):
        self.fit(dataset)
        return self.transform(dataset, inline=inline)


def f_classif(dataset):
    from scipy.stats import f_oneway
    X, y = dataset.getXy()
    args = [X[y == k, :] for k in np.unique(y)]
    F, p = f_oneway(*args)
    return F, p


def f_regression(dataset):
    # READ: http://facweb.cs.depaul.edu/sjost/csc423/documents/f-test-reg.htm
    from scipy.stats import f

    X, y = dataset.getXy()

    corr_coef = np.corrcoef(X, y, rowvar=False)[:-1,-1:].reshape(X.shape[1])
    # corr_coef = np.array([pearsonr(X[:, index], y)[0] for index in range(X.shape[1])])
    corr_coef_sq = corr_coef ** 2

    # The implementation is centered
    deg_f = y.size - 2
    F = corr_coef_sq / (1 - corr_coef_sq) * deg_f
    # corr_coef_squared (ver a forumula)
    p = f.sf(F, 1, deg_f)
    return F, p
