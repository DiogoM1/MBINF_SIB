import tempfile
import unittest
import sys
import os
from pathlib import Path

import pandas as pd

try:
    import si
except:
    DIR = os.path.dirname(os.path.realpath(__file__))
    PATH = os.path.join(DIR, '../../src/')
    sys.path.insert(0, PATH)


class TestUnlabeledDataset(unittest.TestCase):

    def setUp(self):
        from si.data import Dataset
        self.filename = "datasets/lr-example1.data"
        self.dataset = Dataset.from_data(self.filename, labeled=False)
        self.dataframe = pd.read_csv(self.filename)

    def test_empty_file(self):
        from si.data import Dataset
        self.assertRaises(Exception, Dataset.from_data, "datasets/empty.data")

    def test_init(self):
        from si.data import Dataset
        self.assertRaises(Exception, Dataset)

    def testLen(self):
        self.assertGreater(len(self.dataset), 0)

    def test_from_dataframe(self):
        from si.data import Dataset
        dataset = Dataset.from_dataframe(self.dataframe)
        self.assertEqual(len(dataset), 96)

    def test_to_dataframe(self):
        from si.data import Dataset
        dataset = Dataset.from_dataframe(self.dataframe)
        df = dataset.toDataframe()
        self.assertEqual(len(df), 96)

    def test_writeDataset(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdirname = Path(tmpdirname)
            self.dataset.writeDataset(tmpdirname / "file.txt")
            self.assertEqual(len(list(tmpdirname.iterdir())), 1)

    def test_hasLabel(self):
        self.assertFalse(self.dataset.hasLabel())

    def test_getNumFeatures(self):
        self.assertEqual(self.dataset.getNumFeatures(), 2)

    def test_getNumClasses(self):
        self.assertEqual(self.dataset.getNumClasses(), 0)


class TestLabeledDataset(TestUnlabeledDataset):

    def setUp(self):
        from si.data import Dataset
        self.filename = "datasets/hearts.data"
        self.dataset = Dataset.from_data(self.filename, labeled=True)
        self.dataframe = pd.read_csv(self.filename, header=None)

    def test_from_data(self):
        from si.data import Dataset
        self.assertGreater(len(self.dataset), 0)
        self.assertTrue(self.dataset.y.any())

    def test_from_dataframe(self):
        from si.data import Dataset
        dataset = Dataset.from_dataframe(self.dataframe, ylabel=13)
        self.assertEqual(len(dataset), 270)
        self.assertTrue(dataset.y.any())

    def test_to_dataframe(self):
        from si.data import Dataset
        dataset = self.dataset.toDataframe()
        self.assertEqual(len(dataset)+1, 271)

    def test_hasLabel(self):
        self.assertTrue(self.dataset.hasLabel())

    def test_writeDataset(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdirname = Path(tmpdirname)
            self.dataset.writeDataset(tmpdirname / "file.txt")
            self.assertEqual(len(list(tmpdirname.iterdir())), 1)

    def test_getNumFeatures(self):
        self.assertEqual(self.dataset.getNumFeatures(), 13)

    def test_getNumClasses(self):
        self.assertEqual(self.dataset.getNumClasses(), 2)
