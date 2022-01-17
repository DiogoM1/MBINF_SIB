from si.supervised.NN import Pooling2D, MaxPooling2D
import unittest
import numpy as np


class TestPoolingLayers(unittest.TestCase):

    def setUp(self):
        self.matrix = np.random.rand(3, 4, 4, 3)

    def testPooling2d(self):
        self.pooling = MaxPooling2D()
        foward = self.pooling.forward(self.matrix)
        backward = self.pooling.backward(np.array([0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98]), 0)
