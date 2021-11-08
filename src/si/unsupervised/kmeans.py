from copy import copy

import numpy as np
from si.util.distance import euclidian_distance


def random_dist(X, k):
    """
    Init centroids by using a random distribution to generate the coordinates
    """
    xmin = X.min(axis=0)
    xmax = X.max(axis=0)
    centroids = [[np.random.uniform(xmin[i], xmax[i]) for i in range(len(xmax))] for a in
                 range(k)]
    return centroids


def random_points(X, k):
    """
    Init centroids by using a random distribution to generate the coordinates
    """
    rng = np.random.default_rng()
    centroids = rng.choice(X, k).tolist()
    return centroids


class KMeans:
    def __init__(self, k, n=1000, distance=euclidian_distance, init_centroids=random_points):
        self.k = k
        self.n = n
        self.centroids = None
        self._init_centroids = init_centroids
        self.distance = distance

    def fit(self, dataset):
        self.centroids = self._init_centroids(dataset.X, self.k)
        return self.centroids

    def transform(self, dataset):
        new_centroids = copy(self.centroids)
        # use the indxs for each cluster to make their mean from inside that same cluster
        improving = True
        k = 0
        idxs = None

        while improving and k < 1000:
            k += 1
            idxs = self.closest_centroid(dataset)
            for i, centroid in enumerate(new_centroids):
                new_centroids[i] = np.mean(dataset.X[idxs == i], axis=0).tolist()
            if new_centroids == self.centroids:
                improving = False
            else:
                self.centroids = new_centroids

        self.idxs = idxs
        return self.centroids, self.idxs

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)

    def closest_centroid(self, dataset):
        distances = [self.distance(dataset.X, centroid) for i, centroid in enumerate(self.centroids)]
        distances = np.vstack(distances)
        closest_cluster = np.argmin(distances, axis=0)
        return closest_cluster
