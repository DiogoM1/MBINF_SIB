from copy import copy

import numpy as np

from si.data import Dataset
from si.util.scale import StandardScaler

def EVD(X):
    C = np.cov(X.T)
    # EVD
    eigen_values, eigen_vectors = np.linalg.eig(C)
    idx = eigen_vectors.argsort()[::-1]
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]

    # Return principal components and eigenvalues to calculate the portion of sample variance explained
    return np.dot(X, eigen_vectors), eigen_values


def SVD(X):
    # SVD
    u, sigma, vh = np.linalg.svd(X, full_matrices=False)
    # Return principal components and eigenvalues to calculate the portion of sample variance explained
    return np.dot(X, vh.T), (sigma ** 2) / (X.shape[0] - 1)

class PCA:
    # Must use scalar / centralize points
    def __init__(self, k, function=SVD):
        self.k = k
        self._func = function

    def fit(self, dataset):
        self.scaler = StandardScaler()
        self.scaler.fit(dataset)
        return self.scaler

    def transform(self, dataset):
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