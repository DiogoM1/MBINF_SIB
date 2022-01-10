import unittest

import numpy as np

from si.util.distance import euclidean_distance, manhatan_distance, hamming_distance


class TestDistances(unittest.TestCase):

    def setUp(self):
        self.first_point = [5, 2, 4]
        self.second_point = [2, 3, 5]
        self.test_array = np.asarray([self.first_point, self.second_point])
        self.reference = np.asarray([5, 2, 5])

    def test_manhatan_distance(self):
        dis = manhatan_distance(self.test_array, self.reference)
        real_dis = np.asarray([1, 4])
        self.assertTrue(np.array_equal(dis, real_dis))

    def test_euclidian_distance(self):
        dis = euclidean_distance(self.test_array, self.reference)
        real_dis = np.asarray([1, 3.1622776601683795])
        self.assertTrue(np.array_equal(dis, real_dis))

    def test_hamming_distance(self):
        dis = hamming_distance(self.test_array, self.reference)
        real_dis = np.asarray([1, 2])
        self.assertTrue(np.array_equal(dis, real_dis))


if __name__ == '__main__':
    unittest.main()
