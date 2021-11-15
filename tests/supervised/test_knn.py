import unittest

# noinspection DuplicatedCode


class TestKNN(unittest.TestCase):
    """
    Test Conditions

    - Use a labeled dataset
    """

    def setUp(self):
        from si.data import Dataset
        from si.supervised import KNN
        self.filename = "datasets/iris.data"
        self.dataset = Dataset.from_data(self.filename, labeled=True)
        # set the threshold
        self.knn = KNN(3)

    def test_fit(self):
        pass

    def test_predict(self):
        self.knn.fit(self.dataset)
        self.knn_dataset = self.knn.predict(self.dataset)
        self.assertEqual(self.knn_dataset.shape, self.dataset.X.shape)


    def test_cost(self):
        self.knn_dataset = self.knn.fit_transform(self.dataset)
        self.assertEqual(self.knn_dataset.shape, self.dataset.X.shape)

# TODO: Implementar Naive Bayes
# TODO: Implementar uma função de holdout