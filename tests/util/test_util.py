import unittest


class TestLabelGen(unittest.TestCase):
    def setUp(self):
        from si.data import Dataset
        self.filename = "datasets/cpu.data"
        self.dataset = Dataset.from_data(self.filename, labeled=False)

    def test_gen(self):
        self.assertEqual(len(self.dataset.xnames), self.dataset.X.shape[1])


class TestSummaryUnlabeled(unittest.TestCase):
    def setUp(self):
        from si.data import Dataset
        self.filename = "datasets/lr-example1.data"
        self.dataset = Dataset.from_data(self.filename, labeled=False)

    def test_summary(self):
        from si.util import summary
        self.assertEqual(len(summary(self.dataset, fmt="dict")), 2)
        self.assertEqual(summary(self.dataset, fmt="df").shape, (2, 5))
        self.assertRaises(Exception, summary, self.dataset, "dtf2")


class TestSummaryLabeled(TestSummaryUnlabeled):
    def setUp(self):
        from si.data import Dataset
        self.filename = "datasets/hearts.data"
        self.dataset = Dataset.from_data(self.filename, labeled=True)

    def test_summary(self):
        from si.util import summary
        self.assertEqual(len(summary(self.dataset, fmt="dict")), 14)
        self.assertEqual(summary(self.dataset, fmt="df").shape, (14, 5))
        self.assertRaises(Exception, summary, self.dataset, "dtf2")


if __name__ == '__main__':
    unittest.main()
