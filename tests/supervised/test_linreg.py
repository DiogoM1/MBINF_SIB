import unittest

import numpy as np
import pandas as pd
# noinspection DuplicatedCode


class TestLinearRegression(unittest.TestCase):
    """
    Test Conditions

    - Use a labeled dataset
    """

    def setUp(self):
        from si.data import Dataset
        from si.supervised import LinearRegression
        from si.util.train import training_test_data_split, categorical_to_numeric
        self.filename = "datasets/lr-example2.data"
        self.dataset = Dataset.from_data(self.filename, labeled=True)
        self.train_data, self.test_data = training_test_data_split(self.dataset)
        # set the threshold
        self.lr = LinearRegression()

    def test_fit(self):
        self.lr.fit(self.train_data)

    def test_predict(self):
        self.lr.fit(self.train_data)
        self.lr_dataset = np.ma.apply_along_axis(self.lr.predict, axis=0, arr=self.test_data.X.T)
        self.assertEqual(self.lr_dataset.shape, (self.test_data.X.shape[0],))

    def test_cost(self):
        self.lr.fit(self.dataset)
        self.lr_dataset = self.lr.cost()
        self.assertTrue(self.lr_dataset)


class TestLinearRegressionReg(unittest.TestCase):
    """
    Test Conditions

    - Use a labeled dataset
    """

    def setUp(self):
        from si.data import Dataset
        from si.supervised import LinearRegressionReg
        from si.util.train import training_test_data_split
        self.filename = "datasets/lr-example2.data"
        self.dataset = Dataset.from_data(self.filename, labeled=True)
        self.train_data, self.test_data = training_test_data_split(self.dataset)
        # set the threshold
        self.lr = LinearRegressionReg()

    def test_fit(self):
        self.lr.fit(self.train_data)

    def test_predict(self):
        self.lr.fit(self.train_data)
        self.lr_dataset = np.ma.apply_along_axis(self.lr.predict, axis=0, arr=self.test_data.X.T)
        self.assertEqual(self.lr_dataset.shape, (self.test_data.X.shape[0],))

    def test_cost(self):
        self.lr.fit(self.dataset)
        self.lr_dataset = self.lr.cost()
        self.assertTrue(self.lr_dataset)


# TODO: Implementar Naive Bayes
# TODO: Implementar uma função de holdout