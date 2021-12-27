import unittest


class TestGridSearchCV(unittest.TestCase):
    """
    Test Conditions

    - Use a labeled dataset
    """

    def setUp(self):
        from si.data import Dataset
        from si.supervised import KNN
        from si.util.cv import GridSearchCV

        self.filename = 'datasets/breast-bin.data'
        self.dataset = Dataset.from_data(self.filename, labeled=True)
        # set the threshold
        from si.util.distance import euclidian_distance, manhatan_distance

        parameters = {
            "k": [2, 3, 4],
            "distance": [euclidian_distance, manhatan_distance]
        }

        knn = KNN(2)
        self.gscv = GridSearchCV(knn, self.dataset,
                                 parameters=parameters)

    def test_run(self):
        self.gscv.run()

    def test_df(self):
        self.gscv.run()
        print(self.gscv.toDataframe())
