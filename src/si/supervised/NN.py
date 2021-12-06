from abc import ABC, abstractmethod
import numpy as np
from si.supervised.supervised_model import SupervisedModel
from si.util.metrics import accuracy


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
    def __int__(self, input_size, output_size):
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
        pass


class Activation(Layer):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation

    def foward(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward(self, input):
        return self.activation(input)


class NN(SupervisedModel):
    def __init__(self, loss=None, epoch=1000, lr=0.1, verbose=False):
        super().__init__()
        if not loss:
            self.loss = accuracy
        else:
            self.loss = loss
        self.layers = []
        self.lr = lr
        self.epochs = epoch
        self.verbose = verbose

    def add(self, layer):
        self.layers.append(layer)

    def fit(self, dataset):
        raise NotImplementedError

    def predict(self, input_data):
        assert self.is_fitted, "Model must be ftted before it can be predicted"
        output = input_data
        for layer in self.layers:
            output = layer.foward(output)
        return output

    def cost(self, X=None, Y=None):
        if not self.is_fitted:
            raise Exception("The model hasn't been fitted yet.")
        y_pred = np.ma.apply_along_axis(self.predict, axis=0, arr=self.data.X.T)
        return self.loss(self.data.y, y_pred)

# TODO: Get the Activation Functions
# TODO: Adicionar o Call
# TODO: Nas metricas adicionar a derivada da mse e cross entropy, cross entropy prime
