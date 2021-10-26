from copy import copy

import numpy as np
from si.util.distance import euclidian_distance


class KMeans:
    def __init__(self, k, n=1000, distance=None):
        self.k = k
        self.n = n
        self.centroids = None
        if distance:
            self.distance = distance
        else:
            self.distance = euclidian_distance

    def fit(self, dataset):
        X = dataset.X
        self._xmin = X.min(axis=0)
        self._xmax = X.max(axis=0)
        return self._xmin, self._xmax

    def init_centroids(self):
        self.centroids = [[np.random.uniform(self._xmin[i], self._xmax[i]) for i in range(len(self._xmax))] for a in
                          range(self.k)]
        return self.centroids

    def transform(self, dataset, inline=False):
        self.centroids = self.init_centroids()
        new_centroids = copy(self.init_centroids())
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

    def fit_transform(self, dataset, inline=False):
        self.fit(dataset)
        return self.transform(dataset, inline)

    def closest_centroid(self, dataset):
        distances = [self.distance(dataset.X, centroid) for i, centroid in enumerate(self.centroids)]
        distances = np.vstack(distances)
        closest_cluster = np.argmin(distances, axis=0)
        return closest_cluster
