import unittest

# noinspection DuplicatedCode
import warnings


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
        self.assertWarns(Warning, fs.VarianceThreshold, -13)

    def test_fit(self):
        self.vt.fit(self.dataset)
        self.assertGreater(len(self.vt._var), 0)

    def test_transform(self):
        self.vt.fit(self.dataset)
        self.vt_transform = self.vt.transform(self.dataset)
        self.assertEqual(self.vt_transform.X.shape, self.dataset.X.shape)

    def test_transform_inline(self):
        self.vt.fit(self.dataset)
        self.vt_transform = self.vt.transform(self.dataset)
        self.assertEqual(self.vt_transform.X.shape, self.dataset.X.shape)

    def test_fit_transform(self):
        self.vt_transform = self.vt.fit_transform(self.dataset)
        self.vt.fit_transform(self.dataset, inline=True)
        self.assertEqual(self.vt_transform.X.shape, self.dataset.X.shape)

    def test_fit_transform_inline(self):
        self.vt_transform = self.vt.fit_transform(self.dataset)
        self.vt.fit_transform(self.dataset, inline=True)
        self.assertEqual(self.vt_transform.X.shape, self.dataset.X.shape)


class TestFClassif(unittest.TestCase):
    """
    """

    def setUp(self):
        from si.data import Dataset
        self.filename = "datasets/hearts.data"
        self.dataset = Dataset.from_data(self.filename, labeled=True)

    def test_f_classif(self):
        from si.util.feature_selection import f_classif
        F, p = f_classif(self.dataset)
        self.assertEqual(F.shape, (13,))
        self.assertEqual(p.shape, (13,))


class TestFRegression(unittest.TestCase):
    """
    """

    def setUp(self):
        from si.data import Dataset
        self.filename = "datasets/hearts.data"
        self.dataset = Dataset.from_data(self.filename, labeled=True)

    def test_f_regression(self):
        from si.util.feature_selection import f_regression
        F, p = f_regression(self.dataset)
        self.assertEqual(F.shape, (13,))
        self.assertEqual(p.shape, (13,))


class TestKBestClassif(unittest.TestCase):
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
        self.assertRaises(Exception, KBest, "t_classif", 10)
        self.assertRaises(Exception, KBest, "f_classif", -12)

    def test_fit(self):
        self.KBest.fit(self.dataset)
        self.assertEqual(len(self.KBest.F), 13)
        self.assertEqual(len(self.KBest.p), 13)

    def test_transform(self):
        from si.util.feature_selection import KBest
        self.KBest.fit(self.dataset)
        self.KBest_transform = self.KBest.transform(self.dataset)
        self.assertEqual(self.KBest_transform.X.shape, (self.dataset.X.shape[0], self.KBest.k))

        kb = KBest("f_classif", 23)
        kb.fit(self.dataset)
        self.assertWarns(Warning, kb.transform, self.dataset)

    def test_fit_transform(self):
        self.KBest_transform = self.KBest.fit_transform(self.dataset)
        self.assertEqual(self.KBest_transform.X.shape, (self.dataset.X.shape[0], self.KBest.k))

    def test_fit_transform_inline(self):
        self.KBest_transform = self.KBest.fit_transform(self.dataset)
        self.KBest.fit_transform(self.dataset, inline=True)
        self.assertEqual(self.KBest_transform.X.shape, self.dataset.X.shape)


class TestKBestRegression(TestKBestClassif):

    def setUp(self):
        from si.data import Dataset
        from si.util.feature_selection import KBest
        from si.util.feature_selection import f_regression
        self.filename = "datasets/hearts.data"
        self.dataset = Dataset.from_data(self.filename, labeled=True)
        self.KBest = KBest("f_regression", 10)
        self.assertEqual(self.KBest.k, 10)
        self.assertEqual(self.KBest._func, f_regression)

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
