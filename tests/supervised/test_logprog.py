import unittest

import numpy as np
import pandas as pd
# noinspection DuplicatedCode


class TestLogisticRegression(unittest.TestCase):
    """
    Test Conditions

    - Use a labeled dataset
    """

    def setUp(self):
        from si.data import Dataset
        from si.supervised import LogisticRegression
        from si.util.train import trainning_test_data_split, categorical_to_numeric
        self.filename = "datasets/log-ex2.data"
        self.dataset = Dataset.from_data(self.filename, labeled=True)
        self.train_data, self.test_data = trainning_test_data_split(self.dataset)
        # set the threshold
        self.log = LogisticRegression()

    def test_fit(self):
        self.log.fit(self.train_data)

    def test_predict(self):
        self.log.fit(self.train_data)
        self.log_dataset = np.ma.apply_along_axis(self.log.predict, axis=0, arr=self.test_data.X.T)
        self.assertEqual(self.log_dataset.shape, (self.test_data.X.shape[0],))

    def test_cost(self):
        self.log.fit(self.dataset)
        self.log_dataset = self.log.cost()
        self.assertTrue(self.log_dataset)


class TestLogisticRegressionReg(unittest.TestCase):
    """
    Test Conditions

    - Use a labeled dataset
    """

    def setUp(self):
        from si.data import Dataset
        from si.supervised import LogisticRegressionReg
        from si.util.train import trainning_test_data_split
        self.filename = "datasets/log-ex2.data"
        self.dataset = Dataset.from_data(self.filename, labeled=True)
        self.train_data, self.test_data = trainning_test_data_split(self.dataset)
        # set the threshold
        self.log = LogisticRegressionReg()

    def test_fit(self):
        self.log.fit(self.train_data)

    def test_predict(self):
        self.log.fit(self.train_data)
        self.log_dataset = np.ma.apply_along_axis(self.log.predict, axis=0, arr=self.test_data.X.T)
        self.assertEqual(self.log_dataset.shape, (self.test_data.X.shape[0],))

    def test_cost(self):
        self.log.fit(self.dataset)
        self.log_dataset = self.log.cost()
        self.assertTrue(self.log_dataset)


# TODO: Implementar Naive Bayes
# TODO: Implementar uma função de holdout