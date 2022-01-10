import numpy as np

from si.supervised.supervised_model import SupervisedModel
from si.util.metrics import mse


class LinearRegression(SupervisedModel):
    def __init__(self, gd=None, epochs=1000, lr=0.001):
        super(LinearRegression, self).__init__()
        self.gd = gd
        self.theta = None
        self.epochs = epochs
        self.lr = lr
        self.history = None
        self.X, self.y = None, None

    def fit(self, dataset):
        if not dataset.hasLabel():
            raise Exception("Data has no labels.")
        X, self.y = dataset.getXy()
        self.X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.is_fitted = True
        self.train_gd(self.X, self.y) if self.gd else self.train_closed(self.X, self.y)
        return self.X, self.y

    def train_gd(self, X, y):
        m, n = X.shape
        self.history = {}
        self.theta = np.zeros(n)
        for epoch in range(self.epochs):
            grad = 1 / m * (self.theta.dot(X.T)-y).dot(X)
            self.theta -= self.lr * grad
            self.history[epoch] = [self.theta[:], self.cost()]
        return self.theta

    def train_closed(self, X, y):
        self.theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        return self.theta

    def predict(self, x):
        if not self.is_fitted:
            raise Exception("The model hasn't been fitted yet.")
        _x = np.hstack(([1], x))
        return np.dot(self.theta, _x)

    def cost(self, X=None, y=None):
        if not self.is_fitted:
            raise Exception("The model hasn't been fitted yet.")
        X = X if X is not None else self.X
        y = y if y is not None else self.y

        y_pred = X.dot(self.theta)
        return mse(y, y_pred) / 2


class LinearRegressionReg(LinearRegression):
    def __init__(self, gd=None, epochs=1000, lr=0.001, lbd=1):
        super().__init__(gd, epochs, lr)
        self.lbd = lbd

    def train_gd(self, X, y):
        m, n = X.shape
        self.history = {}
        self.theta = np.zeros(n)
        lbds = np.full(m, self.lbd)
        lbds[0] = 0
        for epoch in range(self.epochs):
            grad = 1 / m * (self.theta.dot(X.T)-y).dot(X)
            self.theta -= self.lr * (lbds+grad)
            self.history[epoch] = [self.theta[:], self.cost()]
        return self.theta

    def train_closed(self, X, y):
        m, n = X.shape
        identity = np.eye(n)
        identity[0, 0] = 0
        self.theta = np.linalg.inv(X.T.dot(X) + self.lbd*identity).dot(X.T).dot(y)
        return self.theta
