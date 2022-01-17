import gzip
import os
import pickle

from matplotlib import pyplot

from si.data import Dataset
from si.supervised.NN import Pooling2D, MaxPooling2D, NN, Dense, Activation, Flatten, Conv2D
import unittest
import numpy as np

from si.util import to_categorical
from si.util.activation import Sigmoid, Tanh


def plot_img(img,shape=(28,28)):
    pic = (img*255).reshape(shape)
    pic = pic.astype('int')
    pyplot.imshow(pic, cmap=pyplot.get_cmap('gray'))
    pyplot.show()


def load_mnist(sample_size=None):
    filename = 'datasets/mnist.pkl.gz'
    f = gzip.open(filename, 'rb')
    data = pickle.load(f, encoding='bytes')
    (x_train, y_train), (x_test, y_test) = data
    if sample_size:
        return Dataset(x_train[:sample_size], y_train[:sample_size]), Dataset(x_test, y_test)
    else:
        return Dataset(x_train, y_train), Dataset(x_test, y_test)


def preprocess(data):
    # reshape and normalize input data
    data.X = data.X.reshape(data.X.shape[0], 28, 28, 1)
    data.X = data.X.astype('float32')
    data.X /= 255
    data.y = to_categorical(data.y)
    return data


class TestPoolingLayers(unittest.TestCase):

    def setUp(self):
        self.matrix = np.random.rand(3, 4, 4, 3)
        self.train, self.test = load_mnist(500)
        self.train, self.test = preprocess(self.train), preprocess(self.test)

    def testPooling2d(self):
        self.pooling = MaxPooling2D()
        foward = self.pooling.forward(self.matrix)
        backward = self.pooling.backward(np.array(
            [0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98,
             0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98,
             0.98]), 0)

    def testMaxPooling2d(self):
        self.net = NN(epochs=200, lr=0.1, verbose=False)
        self.net.add(MaxPooling2D(2, 1))
        self.net.add(Flatten())
        self.net.add(Dense(27 * 27 * 1, 10))
        self.net.add(Activation(Sigmoid()))

        from si.util.metrics import cross_entropy
        from si.util.metrics import cross_entropy_prime
        self.net.use_loss(cross_entropy, cross_entropy_prime)
        self.net.fit(self.train)


        pool1 = self.net.layers[0]
        img2 = pool1.forward(self.test.X[:1])
        plot_img(img2, shape=(27, 27))
        plot_img(self.test.X[:1], shape=(28, 28))
        a = 0


    def testMaxPooling2d2(self):
        self.net = NN(epochs=30, lr=0.1, verbose=False)

        self.net.add(Conv2D((28, 28, 1), (3, 3), 1))
        self.net.add(Activation(Tanh()))
        self.net.add(MaxPooling2D(2, 1))
        self.net.add(Flatten())
        self.net.add(Dense(25 * 25 * 1, 100))
        self.net.add(Activation(Tanh()))
        self.net.add(Dense(100, 10))
        self.net.add(Activation(Sigmoid()))

        from si.util.metrics import cross_entropy
        from si.util.metrics import cross_entropy_prime
        self.net.use_loss(cross_entropy, cross_entropy_prime)
        self.net.fit(self.train)

        out = self.net.predict(self.test.X[0:3])
        print("\n")
        print("predicted values : ")
        print(np.round(out), end="\n")
        print("true values : ")
        print(self.test.y[0:3])

        conv1 = self.net.layers[0]
        act1 = self.net.layers[1]
        pool1 = self.net.layers[2]

        img1 = conv1.forward(self.test.X[:1])
        plot_img(img1, shape=(26, 26))

        img2 = pool1.forward(act1.forward(img1))
        plot_img(img2, shape=(25, 25))
