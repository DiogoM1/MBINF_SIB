import unittest


class test_summary(unittest.TestCase):
    def setUp(self):
        from si.data import Dataset
        self.filename = "datasets/lr-example1.data"
        self.dataset = Dataset.from_data(self.filename, labeled=False)

    def test_summary(self):
        from si.util import summary
        self.assertEqual(len(summary(self.dataset, format="dict")), 4)
        self.assertEqual(summary(self.dataset, format="df").shape, (1, 4))
        self.assertRaises(Exception, summary, self.dataset, "dtf2")


if __name__ == '__main__':
    unittest.main()
