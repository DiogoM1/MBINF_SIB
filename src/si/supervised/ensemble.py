import numpy as np

from si.supervised.supervised_model import SupervisedModel


def majority(values):
    return max(set(values), key=values.count)


def average(values):
    return sum(values) / len(values)


class Essemble(SupervisedModel):
    def __init__(self, models, f_vote, score):
        super().__init__()
        self.models = models
        self.f_vote = f_vote
        self.score = score

    def fit(self, dataset):
        self.dataset = dataset
        for model in self.models:
            model.fit(dataset)

    def predict(self, x):
        votes = [model.preditc(x) for model in self.models]
        consensus = self.f_vote(votes)
        return consensus

    def cost(self):
        pred = np.ma.apply_along_axis(self.predict, axis=0, arr=self.dataset.X.T)
        return self.score(self.dataset.y, pred)