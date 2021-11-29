import itertools

import numpy as np

from si.util.train import trainning_test_data_split


class CrossValidationScore:

    def __init__(self, model, dataset, score=None, **kwargs):
        self.model = model
        self.dataset = dataset
        self.cv = kwargs.get("cv", 3)
        self.score = score
        self.split = kwargs.get("split", 0.8)
        self.train_scores = None
        self.test_scores = None
        self.ds = None

    def run(self):
        train_scores = []
        test_scores = []
        ds = []
        for _ in range(self.cv):
            train, test = trainning_test_data_split(self.dataset, self.split)
            ds.append((train, test))
            self.model.fit(train)
            if self.score:
                train_scores.append(self.model.cost())
                test_scores.append(self.model.cost(test.X, test.y))
            else:
                y_train = np.ma.apply_along_axis(self.model.predict(), axis=0, arr=train.X.T)
                train_scores.append(self.score(train.y, y_train))
                y_test = np.ma.apply_along_axis(self.model.predict(), axis=0, arr=test.X.T)
                train_scores.append(self.score(test.y, y_test))
        self.train_scores = train_scores
        self.test_scores = test_scores
        self.ds = ds
        return train_scores, test_scores

    def toDataframe(self):
        import pandas as pd
        assert self.train_scores and self.test_scores, "Need to run trainning before hand"
        np.array((self.train_scores, self.test_scores))


class GridSearchCV:

    def __init__(self, model, dataset, parameters, **kwargs):
        self.model = model
        self.dataset = dataset
        hasparam = [hasattr(self.model, param) for param in parameters]
        if np.all(hasparam):
            self.parameters = parameters
        else:
            index = hasparam.index(False)
            keys = list(parameters.keys())
            raise ValueError(f" Wrong parameters: {keys[index]}")
        self.kwargs = kwargs
        self.results = None

    def run(self):
        self.results = []
        attrs = list(self.parameters.keys())
        values = list(self.parameters.values())
