import unittest
from copy import copy

import numpy as np


class TestUnlabeledDataset(unittest.TestCase):

    def setUp(self):
        from si.data import Dataset
        from si.data import scale
        self.filename = "datasets/lr-example1.data"
        self.dataset = Dataset.from_data(self.filename, labeled=True)
        self.scalar = scale.StandardScaler()

    def test_fit(self):
        self.scalar.fit(self.dataset)
        self.assertGreater(len(self.scalar.mean), 0)
        self.assertGreater(len(self.scalar.var), 0)

    def test_transform(self):
        self.scalar.fit(self.dataset)
        self.scalar_transform = self.scalar.transform(self.dataset)
        self.assertEqual(self.scalar_transform.X.shape, self.dataset.X.shape)
        self.assertFalse(np.array_equal(self.scalar_transform.X, self.dataset.X))

    def test_fit_transform_inline(self):
        self.scalar_transform = self.scalar.fit_transform(self.dataset)
        self.scalar.fit_transform(self.dataset, inline=True)
        self.assertEqual(self.scalar_transform.X.shape, self.dataset.X.shape)

    def test_fit_transform(self):
        self.scalar_transform = self.scalar.fit_transform(self.dataset)
        self.assertEqual(self.scalar_transform.X.shape, self.dataset.X.shape)
        self.assertFalse(np.array_equal(self.scalar_transform.X, self.dataset.X))

    def test_inverse_transform(self):
        self.scalar_transform = self.scalar.fit_transform(self.dataset)
        self.assertTrue(np.array_equal(self.scalar.inverse_transform(self.scalar_transform).X, self.dataset.X))

    def test_fit_transform_inline(self):
        dataset_copy = copy(self.dataset)
        self.scalar.fit_transform(self.dataset, inline=True)
        self.scalar.inverse_transform(self.dataset, inline=True)
        self.assertTrue(np.array_equal(dataset_copy.X, self.dataset.X))


if __name__ == '__main__':
    unittest.main()
