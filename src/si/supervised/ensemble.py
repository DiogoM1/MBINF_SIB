import numpy as np

from si.supervised.supervised_model import SupervisedModel


def majority(values):
    return max(set(values), key=values.count)


def average(values):
    return sum(values) / len(values)


class Ensemble(SupervisedModel):
    def __init__(self, models, f_vote, score):
        super().__init__()
        self.models = models
        self.f_vote = f_vote
        self.score = score
        self.is_fitted = False
        self.dataset = None

    def fit(self, dataset):
        self.dataset = dataset
        for model in self.models:
            model.fit(dataset)
        self.is_fitted = True

    def predict(self, x):
        votes = [model.predict(x) for model in self.models]
        consensus = self.f_vote(votes)
        return consensus

    def cost(self, X=None, y=None):
        if not self.is_fitted:
            raise Exception("The model hasn't been fitted yet.")

        X = X if X is not None else self.dataset.X
        y = y if y is not None else self.dataset.y

        pred = np.ma.apply_along_axis(self.predict, axis=0, arr=X.T)
        return self.score(y, pred)
