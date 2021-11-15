from abc import ABC, abstractmethod


class UnsupervisedModel(ABC):

    def __init__(self):
        self.is_fitted = False

    @abstractmethod
    def fit(self, dataset):
        raise NotImplementedError

    @abstractmethod
    def transform(self, dataset):
        raise NotImplementedError

    @abstractmethod
    def fit_transform(self, dataset):
        raise NotImplementedError

