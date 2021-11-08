import unittest

# noinspection DuplicatedCode
import warnings

import numpy as np


class TestKmeans(unittest.TestCase):
    """
    Test Conditions

    - Use a labeled dataset
    """

    def setUp(self):
        from si.data import Dataset
        from si.unsupervised import KMeans
        self.filename = "datasets/lr-example1.data"
        self.dataset = Dataset.from_data(self.filename, labeled=True)
        # set the threshold
        self.kmeans = KMeans(2)

    def test_fit(self):
        self.kmeans.fit(self.dataset)
        self.assertEqual(type(self.kmeans.centroids), list)
        self.assertEqual(len(self.kmeans.centroids), 2)

    def test_closest_centroid(self):
        self.kmeans.fit(self.dataset)
        new_centroids = self.kmeans.closest_centroid(self.dataset)
        self.assertEqual(new_centroids.shape, (len(self.dataset),))
        self.assertLessEqual(len(np.unique(new_centroids)), self.kmeans.k)

    def test_transform(self):
        self.kmeans.fit(self.dataset)
        self.kmeans_centroids, self.kmeans_clusters = self.kmeans.transform(self.dataset)
        self.assertEqual(len(self.kmeans_centroids), self.kmeans.k)
        self.assertEqual(self.kmeans_clusters.shape, (len(self.dataset),))

    def test_fit_transform(self):
        self.kmeans_centroids, self.kmeans_clusters = self.kmeans.fit_transform(self.dataset)
        self.assertEqual(len(self.kmeans_centroids), self.kmeans.k)
        self.assertEqual(self.kmeans_clusters.shape, (len(self.dataset),))


if __name__ == '__main__':
    unittest.main()
