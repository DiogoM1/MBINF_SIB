import unittest

import numpy as np
import pandas as pd
# noinspection DuplicatedCode


class TestKNN(unittest.TestCase):
    """
    Test Conditions

    - Use a labeled dataset
    """

    def setUp(self):
        from si.data import Dataset
        from si.supervised import KNN
        from si.util.train import training_test_data_split, categorical_to_numeric
        self.filename = "datasets/iris.data"
        data = pd.read_csv(self.filename)
        self.data, self.cat = categorical_to_numeric(data, -1)
        self.dataset = Dataset.from_dataframe(self.data, ylabel="class")
        self.train_data, self.test_data = training_test_data_split(self.dataset)
        # set the threshold
        self.knn = KNN(10)

    def test_fit(self):
        pass

    def test_predict(self):
        self.knn.fit(self.train_data)
        self.knn_dataset = np.ma.apply_along_axis(self.knn.predict, axis=0, arr=self.test_data.X.T)
        self.assertEqual(self.knn_dataset.shape, (self.test_data.X.shape[0],))

    def test_cost(self):
        self.knn.fit(self.dataset)
        self.knn_dataset = self.knn.cost()
        self.assertLessEqual(self.knn_dataset, 1)
        self.assertGreaterEqual(self.knn_dataset, 0.60)

# TODO: Implementar Naive Bayes
# TODO: Implementar uma função de holdout