from abc import ABC, abstractmethod
import numpy as np
from si.supervised.supervised_model import SupervisedModel
from si.util.metrics import accuracy, mse, mse_prime


class Layer(ABC):
    def __init__(self):
        self.input = None
        self.output = None
        self.weights = None
        self.bias = None

    @abstractmethod
    def foward(self, input_data):
        raise NotImplementedError

    @abstractmethod
    def backward(self, output_error, learning_rate):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.rand(input_size, output_size) -0.5
        self.bias = np.random.rand(1, output_size) # pode ser random ou nulos

    def set_weights(self, weights, bias):
        if weights.shape != self.weights.shape:
            raise ValueError(f"shapes mismathc {weights.shape} and {self.weights.shape}.")
        if bias.shape != self.bias.shape:
            raise ValueError(f"shapes mismathc {bias.shape} and {self.bias.shape}.")
        self.weights = weights
        self.bias = bias

    def foward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weight_error = np.dot(self.input.T, output_error)
        bias_error = np.sum(output_error, axis=0)
        # gradient descent is used to update the values
        self.weights -= learning_rate*weight_error
        self.bias -= learning_rate*bias_error
        return input_error


class Activation(Layer):
    def __init__(self, activation, activation_prime=None):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def foward(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_error, learning_rate):
        # learning rate is not used as there is no parameters to learn
        return np.multiply(self.activation(self.input), output_error)


class NN(SupervisedModel):
    def __init__(self, loss=None, loss_prime=None, epochs=1000, lr=0.1, verbose=False):
        super().__init__()
        if not loss:
            self.loss = mse
        else:
            self.loss = loss
        if not loss_prime:
            self.loss_prime = mse_prime
        else:
            self.loss_prime = loss_prime
        self.layers = []
        self.lr = lr
        self.epochs = epochs
        self.verbose = verbose

    def add(self, layer):
        self.layers.append(layer)

    def fit(self, dataset):
        err = 0
        X, y = dataset.getXy()
        self.dataset = dataset
        self.history = {}
        for epoch in range(self.epochs):
            output = X

            # Forward Propagation
            for layer in self.layers:
                output = layer.foward(output)

            # backward propagation
            error = self.loss_prime(y, output)
            for layer in reversed(self.layers):
                error = layer.backward(error, self.lr)

            # calculate average error on all samples
            err = self.loss(y, output)
            self.history[epoch] = err
            if self.verbose:
                print(f"epoch {epoch+1}/{self.epochs} error = {err}")

        if not self.verbose:
            print(f"error={err}")
        self.is_fitted = True

    def predict(self, input_data):
        assert self.is_fitted, "Model must be ftted before it can be predicted"
        output = input_data
        for layer in self.layers:
            output = layer.foward(output)
        return output

    def classify(self, input_data):
        return self.predict(input_data)

    def cost(self, X=None, y=None):
        if not self.is_fitted:
            raise Exception("The model hasn't been fitted yet.")

        X = X if X is not None else self.dataset.X
        y = y if y is not None else self.dataset.y

        y_pred = self.predict(X)
        return self.loss(y, y_pred)

# TODO: Get the Activation Functions
# TODO: Adicionar o Call