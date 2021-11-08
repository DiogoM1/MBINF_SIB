import unittest

# noinspection DuplicatedCode
import warnings

import numpy as np


class TestPCA(unittest.TestCase):
    """
    Test Conditions

    - Use a labeled dataset
    """

    def setUp(self):
        from si.data import Dataset
        from si.unsupervised import PCA, SVD
        self.filename = "datasets/lr-example1.data"
        self.dataset = Dataset.from_data(self.filename, labeled=True)
        # set the threshold
        self.pca = PCA(1, SVD)

    def test_fit(self):
        from si.util.scale import StandardScaler
        self.pca.fit(self.dataset)
        self.assertEqual(type(self.pca.scaler), type(StandardScaler()))
        self.assertEqual(self.pca.scaler.mean.shape[0], self.dataset.getNumFeatures())
        self.assertEqual(self.pca.scaler.var.shape[0], self.dataset.getNumFeatures())

    def test_transform(self):
        self.pca.fit(self.dataset)
        self.pca_dataset = self.pca.transform(self.dataset)
        self.assertEqual(self.pca_dataset.shape, self.dataset.X.shape)
        self.assertEqual(self.pca.eigen.shape, (self.dataset.X.shape[1],))

    def test_fit_transform(self):
        self.pca_dataset = self.pca.fit_transform(self.dataset)
        self.assertEqual(self.pca_dataset.shape, self.dataset.X.shape)
        self.assertEqual(self.pca.eigen.shape, (self.dataset.X.shape[1],))
