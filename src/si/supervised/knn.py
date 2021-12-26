import numpy as np

from si.supervised.supervised_model import SupervisedModel
from si.util.distance import euclidian_distance
from si.util.metrics import accuracy


class KNN(SupervisedModel):
    def __init__(self, k, distance_func=euclidian_distance, classification=True):
        super().__init__()
        self.k = k
        self.classification = classification
        self._distance_func = distance_func

    def fit(self, dataset):
        if not dataset.hasLabel():
            raise Exception("Data has no labels.")
        self.dataset = dataset
        self.is_fitted = True
        return self.dataset

    def get_neighbours(self, x):
#        dist = np.ma.apply_along_axis(self._distance_func, axis=0, arr=self.dataset.X, y=x)
#        dist = self._distance_func(self.dataset.X, x)
        distances = self._distance_func(self.dataset.X, x)
        a = np.argsort(distances)[:self.k]
        return a

    def predict(self, x):
        if not self.is_fitted:
            raise Exception("The model hasn't been fitted yet.")
        neighbours = self.get_neighbours(x)
        i = neighbours.tolist()
        meta_data = self.dataset.y[i].tolist()
        if self.classification:
            prediction = max(set(meta_data), key=meta_data.count)
        else:
            prediction = sum(meta_data) / len(meta_data)
        return np.array(prediction)

    def cost(self, X=None, y=None):
        if not self.is_fitted:
            raise Exception("The model hasn't been fitted yet.")
        X = X if X is not None else self.dataset.X
        y = y if y is not None else self.dataset.y
        
        y_pred = np.ma.apply_along_axis(self.predict, axis=0, arr=self.dataset.X.T)
        return accuracy(self.dataset.y, y_pred)
