import numpy as np

from si.supervised.supervised_model import SupervisedModel
from si.util.distance import euclidian_distance
from si.util.metrics import accuracy
from si.util.train import vectorized_dataset


class NaiveBayes(SupervisedModel):
    def __init__(self):
        super().__init__()
        self.labels = None
        self.is_fitted = False
        self.prob_matrix = None

    def fit(self, dataset):
        if not dataset.hasLabel():
            raise Exception("Data has no labels.")
        self.labels = np.unique(dataset.y)
        prob_matrix = {}
        for label in self.labels:
            prob_matrix[label] = [np.unique(dataset.X[dataset.y == label, i], return_counts=True) for i, _ in
                                  enumerate(dataset.xnames)]
            prob_matrix[label] = [dict(zip(i[0], i[1] / i[1].sum())) for i in prob_matrix[label]]
        self.prob_matrix = prob_matrix
        self.is_fitted = True
        return self.prob_matrix

    def predict(self, x):
        if not self.is_fitted:
            raise Exception("The model hasn't been fitted yet.")
        prob = {}
        for label in self.labels:
            prob[label] = 1
            for col, val in enumerate(x):
                prob[label] *= self.prob_matrix[label][col].get(val, 0)
        choice = max(prob, key=prob.get)
        return choice

    def cost(self, X=None, y=None):
        if not self.is_fitted:
            raise Exception("The model hasn't been fitted yet.")

        y_pred = np.ma.apply_along_axis(self.predict, axis=0, arr=X.T)
        return accuracy(y, y_pred)
