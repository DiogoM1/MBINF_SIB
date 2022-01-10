import numpy as np


def manhatan_distance(X, y):
    from numpy import absolute
    """
    X: (N,N)
    Y: (N,)
    """
    return absolute(X - y).sum(axis=1)


def euclidean_distance(X, y):
    from numpy import sqrt
    """
    X: (N,N)
    Y: (N,)
    """
    return sqrt(((X - y) ** 2).sum(axis=1))


def hamming_distance(X, y):
    """
    X: (N,N)
    Y: (N,)
    """
    return np.size(y) - np.equal(X, y).sum(axis=1)
