import unittest

import numpy as np

from si.util.metrics import accuracy


class TestDistances(unittest.TestCase):

    def setUp(self):
        self.first_point = [5, 2, 4]
        self.second_point = [2, 3, 5]
        self.test_array = np.asarray([self.first_point, self.second_point])
        self.reference = np.asarray([5, 2, 5])
