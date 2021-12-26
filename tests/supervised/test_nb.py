import unittest

import numpy as np
import pandas as pd


# noinspection DuplicatedCode


class TestNaiveBayes(unittest.TestCase):
    """
    Test Conditions

    - Use a labeled dataset
    """

    def setUp(self):
        from si.data import Dataset
        from si.supervised import NaiveBayes
        from si.util.train import training_test_data_split, categorical_to_numeric
        self.filename = "datasets/breast-bin.data"
        data = pd.read_csv(self.filename)
        self.dataset = Dataset.from_data(self.filename, labeled=True)
        # set the threshold
        self.nb = NaiveBayes(10)

    def test_fit(self):
        self.nb.fit(self.dataset)

    def test_predict(self):
        self.nb.fit(self.dataset)
        self.nb.predict(self.dataset.X[0])

    def test_cost(self):
        self.nb.fit(self.dataset)
        self.nb_dataset = self.nb.cost(self.dataset.X, self.dataset.y)
        self.assertLessEqual(self.nb_dataset, 1)
        self.assertGreaterEqual(self.nb_dataset, 0.60)

# TODO: Implementar Naive Bayes
# TODO: Implementar uma função de holdout
