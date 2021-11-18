import numpy as np

from si.supervised.supervised_model import SupervisedModel
from si.util.distance import euclidian_distance, sigmoid
from si.util.metrics import accuracy, mse


class LogisticRegression(SupervisedModel):

    def __init__(self, epochs=1000, lr=0.001):
        super(LogisticRegression, self).__init__()
        self.theta = None
        self.epochs = epochs
        self.lr = lr

    def fit(self, dataset):
        if not dataset.hasLabel():
            raise Exception("Data has no labels.")
        X, self.y = dataset.getXy()
        self.X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.is_fitted = True
        self.train(self.X, self.y)
        return self.X, self.y

    def train(self, X, y):
        m, n = X.shape
        self.history = {}
        self.theta = np.zeros(n)
        for epoch in range(self.epochs):
            z = np.dot(X, self.theta)
            h = sigmoid(z)
            grad = np.dot(X.T, (h - y))
            self.theta -= self.lr * grad
            self.history[epoch] = [self.theta[:], self.cost()]
        return self.theta

    def predict(self, x):
        if not self.is_fitted:
            raise Exception("The model hasn't been fitted yet.")
        _x = np.hstack(([1], x))
        return np.round(sigmoid(np.dot(self.theta, _x)))

    def cost(self):
        if not self.is_fitted:
            raise Exception("The model hasn't been fitted yet.")
        y_pred = sigmoid(np.dot(self.theta, self.X.T))
        epsilon = 1e-5
        return (((-self.y).dot(np.log(y_pred + epsilon))) - ((1 - self.y).dot(np.log(1 - y_pred + epsilon)))) / self.y.size
        # change this


# Implement Regularização L1 / L2

class LogisticRegressionReg(LogisticRegression):
    def __init__(self, epochs=1000, lr=0.001, lbd=1):
        super().__init__(epochs, lr)
        self.lbd = lbd

    def train(self, X, y):
        m, n = X.shape
        self.history = {}
        self.theta = np.zeros(n)
        for epoch in range(self.epochs):
            z = np.dot(X, self.theta)
            h = sigmoid(z)
            grad = np.dot(X.T, (h - y))
            reg = self.lbd/m*self.theta
            self.theta -= self.lr * (grad - reg)
            self.history[epoch] = [self.theta[:], self.cost()]
        return self.theta

    def cost(self):
        if not self.is_fitted:
            raise Exception("The model hasn't been fitted yet.")
        y_pred = sigmoid(np.dot(self.theta, self.X.T))
        m = self.y.size
        epsilon = 1e-5
        reg = self.lbd / (2 * m) * np.sum(np.square(self.theta))
        return (((-self.y).dot(np.log(y_pred + epsilon))) - ((1 - self.y).dot(np.log(1 - y_pred + epsilon)))) / m + reg