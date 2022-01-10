from abc import ABC, abstractmethod

import numpy as np

from si.supervised.supervised_model import SupervisedModel
from si.util.im2col import im2col, pad2D, col2im
from si.util.metrics import mse, mse_prime


class Layer(ABC):
    def __init__(self):
        self.input = None
        self.output = None
        self.weights = None
        self.bias = None

    @abstractmethod
    def forward(self, input_data):
        raise NotImplementedError

    @abstractmethod
    def backward(self, output_error, learning_rate):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size)  # pode ser random ou nulos

    def set_weights(self, weights, bias):
        if weights.shape != self.weights.shape:
            raise ValueError(f"shapes mismathc {weights.shape} and {self.weights.shape}.")
        if bias.shape != self.bias.shape:
            raise ValueError(f"shapes mismathc {bias.shape} and {self.bias.shape}.")
        self.weights = weights
        self.bias = bias

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weight_error = np.dot(self.input.T, output_error)
        bias_error = np.sum(output_error, axis=0)
        # gradient descent is used to update the values
        self.weights -= learning_rate * weight_error
        self.bias -= learning_rate * bias_error
        return input_error


class Activation(Layer):
    def __init__(self, activation, activation_prime=None):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_error, learning_rate):
        # learning rate is not used as there is no parameters to learn
        return np.multiply(self.activation(self.input), output_error)


class Conv2D(Layer):
    def __init__(self, input_shape, kernel_shape, layer_depth, stride=1, padding=0):
        super().__init__()
        self.X_shape, self.Y_shape, self.in_ch = input_shape
        self.out_ch = layer_depth
        self.stride = stride
        self.padding = padding
        # weights
        self.weights = np.random.rand(kernel_shape[0], kernel_shape[1], self.in_ch, self.out_ch) - 0.5
        # bias
        self.bias = np.zeros((self.out_ch, 1))
        self.X_col = None

    def forward(self, input_data):
        s = self.stride
        self.X_shape = input_data.shape
        _, p = pad2D(input_data, self.padding, self.weights.shape[:2], s)

        pr1, pr2, pc1, pc2 = p
        fr, fc, in_ch, out_ch = self.weights.shape
        n_ex, in_rows, in_cols, in_ch = input_data.shape

        # compute the dimensions of the convolution output
        out_rows = int((in_rows + pr1 + pr2 - fr) / s + 1)
        out_cols = int((in_cols + pc1 + pc2 - fc) / s + 1)

        # convert X and W into the appropriate 2d matrices and take their product
        self.X_col, _ = im2col(input_data, self.weights.shape, p, s)
        w_col = self.weights.transpose(3, 2, 0, 1).reshape(out_ch, -1)

        output_data = (w_col @ self.X_col + self.bias).reshape(out_ch, out_rows, out_cols, n_ex).transpose(3, 1, 2, 0)
        return output_data

    def backward(self, output_error, learning_rate):
        fr, fc, in_ch, out_ch = self.weights.shape
        p = self.padding
        db = np.sum(output_error, axis=(0, 1, 2))
        db = db.reshape(out_ch, )

        dout_reshaped = output_error.transpose(1, 2, 3, 0).reshape(out_ch, -1)
        d_w = dout_reshaped @ self.X_col.T
        d_w = d_w.reshape(self.weights.shape)
        w_reshaped = self.weights.reshape(out_ch, -1)
        d_x_col = w_reshaped.T @ dout_reshaped
        input_error = col2im(d_x_col, self.X_shape, self.weights.shape, (p, p, p, p), self.stride)

        self.weights -= learning_rate * d_w
        self.bias -= learning_rate * db

        return input_error


class AveragePooling(Layer):

    def forward(self, input_data):
        pass

    def backward(self, output_error, learning_rate):
        pass


class MaxPooling(Layer):
    def __init__(self):
        super().__init__()
        self.input_shape = None

    def forward(self, input_data):
        pass

    def backward(self, output_error, learning_rate):
        pass


class Flatten(Layer):

    def __init__(self):
        super().__init__()
        self.input_shape = None

    def forward(self, input_data):
        self.input_shape = input_data.shape
        output = input_data.reshape(self.input_shape[0], -1)
        return output

    def backward(self, output_error, learning_rate):
        return output_error.reshape(self.input_shape)


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
        self.dataset = None
        self.history = None

    def add(self, layer):
        self.layers.append(layer)

    def fit(self, dataset):
        error = 0
        X, y = dataset.getXy()
        self.dataset = dataset
        self.history = {}
        for epoch in range(self.epochs):
            output = X

            # Forward Propagation
            for layer in self.layers:
                output = layer.forward(output)

            # backward propagation
            error = self.loss_prime(y, output)
            for layer in reversed(self.layers):
                error = layer.backward(error, self.lr)

            # calculate average error on all samples
            err = self.loss(y, output)
            self.history[epoch] = err
            if self.verbose:
                print(f"epoch {epoch + 1}/{self.epochs} error = {err}")
            else:
                print('\r', f"epoch {epoch + 1}/{self.epochs} error = {err}")

        self.is_fitted = True

    def predict(self, input_data):
        assert self.is_fitted, "Model must be ftted before it can be predicted"
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
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
