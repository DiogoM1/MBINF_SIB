import numpy as np

from si.unsupervised.unsupervised_model import UnsupervisedModel
from si.data.scale import StandardScaler


def evd(X):
    C = np.cov(X.T)
    # EVD
    eigen_values, eigen_vectors = np.linalg.eig(C)
    idx = eigen_values.argsort()[::-1]
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]

    # Return principal components and eigenvalues to calculate the portion of sample variance explained
    return np.dot(X, eigen_vectors), eigen_values


def svd(X):
    # SVD
    u, sigma, vh = np.linalg.svd(X, full_matrices=False)
    # Return principal components and eigenvalues to calculate the portion of sample variance explained
    return u.dot(np.diag(sigma)), (sigma ** 2) / (X.shape[0] - 1)


class PCA(UnsupervisedModel):
    # Must use scalar / centralize points
    def __init__(self, k, function=svd):
        super().__init__()
        self.k = k
        self._func = function

    def fit(self, dataset):
        self.scaler = StandardScaler()
        self.scaler.fit(dataset)
        self.is_fitted = True
        return self.scaler

    def transform(self, dataset):
        if not self.is_fitted:
            raise Exception("The model hasn't been fitted yet.")
        centered = self.scaler.transform(dataset)
        pc, self.eigen = self._func(centered.X)
        self.variance_explained()
        return pc[:, 0:self.k]

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)

    def variance_explained(self):
        self.var_exp = list(self.eigen / self.eigen.sum())
        return self.var_exp

    def cumulative_variance_explained(self):
        percent_dict = {}
        self.variance_explained()
        for i, value in enumerate(self.var_exp):
            if i == 0:
                percent_dict[i] = value
            else:
                percent_dict[i] = value + percent_dict[i - 1]
        return percent_dict

    def inverse_transform(self):
        pass