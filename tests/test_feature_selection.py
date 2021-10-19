import unittest

import numpy as np


# noinspection DuplicatedCode
class TestVarianceThreshold(unittest.TestCase):
    """
    Test Conditions

    - Use a labeled dataset
    """

    def setUp(self):
        from si.data import Dataset
        from si.util import feature_selection as fs
        self.filename = "datasets/lr-example1.data"
        self.dataset = Dataset.from_data(self.filename, labeled=True)
        # set the threshold
        self.vt = fs.VarianceThreshold(0)

    def test_fit(self):
        self.vt.fit(self.dataset)
        self.assertGreater(len(self.vt._var), 0)

    def test_transform(self):
        self.vt.fit(self.dataset)
        self.vt_transform = self.vt.transform(self.dataset)
        self.assertEqual(self.vt_transform.X.shape, self.dataset.X.shape)
        self.assertFalse(np.array_equal(self.vt_transform.X, self.dataset.X))

    def test_fit_transform(self):
        self.scalar_transform = self.scalar.fit_transform(self.dataset)
        self.assertEqual(self.scalar_transform.X.shape, self.dataset.X.shape)
        self.assertFalse(np.array_equal(self.scalar_transform.X, self.dataset.X))


class TestFFunctions(unittest.TestCase):
    """
    """

    def setUp(self):
        from si.data import Dataset
        from si.util import feature_selection as fs
        self.filename = "datasets/hearts.data"
        self.dataset = Dataset.from_data(self.filename, labeled=True)

    def test_f_classif(self):
        from si.util.feature_selection import f_classif
        F, p = f_classif(self.dataset)
        self.assertEqual(F.shape, (13,))
        self.assertEqual(p.shape, (13,))

    def test_f_regression(self):
        from si.util.feature_selection import f_regression
        F, p = f_regression(self.dataset)
        self.assertEqual(F.shape, (13,))
        self.assertEqual(p.shape, (13,))


class TestKBest(unittest.TestCase):
    """
    """

    def setUp(self):
        from si.data import Dataset
        from si.util.feature_selection import KBest
        from si.util.feature_selection import f_classif
        self.filename = "datasets/hearts.data"
        self.dataset = Dataset.from_data(self.filename, labeled=True)
        self.KBest = KBest("f_classif", 10)
        self.assertEqual(self.KBest.k, 10)
        self.assertEqual(self.KBest._func, f_classif)

    def test_fit(self):
        self.KBest.fit(self.dataset)
        self.assertEqual(len(self.KBest.F), 13)
        self.assertEqual(len(self.KBest.p), 13)

    def test_transform(self):
        self.KBest.fit(self.dataset)
        self.KBest_transform = self.KBest.transform(self.dataset)
        self.assertEqual(self.KBest_transform.X.shape, (self.dataset.X.shape[0], self.KBest.k))

    def test_fit_transform(self):
        self.KBest_transform = self.KBest.fit_transform(self.dataset)
        self.assertEqual(self.KBest_transform.X.shape, (self.dataset.X.shape[0], self.KBest.k))


if __name__ == '__main__':
    unittest.main()
