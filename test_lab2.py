from unittest import TestCase
import numpy as np
from lab2 import determinant

class Test(TestCase):

    a1 = np.asarray([[1]])
    a2 = np.asarray([[1,2], [3,4]])
    a3 = np.asarray([
        [6,7,8],
        [10,11,12],
        [14,15,17]
    ])
    a4 = np.asarray([
        [2,14,-2,4],
        [5,6,7,19],
        [9,5,5,3],
        [13,17,15,16]
    ])

    det_tests = [(a1,1), (a2, -2), (a3, -4), (a4, -14104)]

    def test_determinant(self):
        for p, ans in self.det_tests:
            with self.subTest():
                print(determinant(p), ans)
                self.assertEqual(determinant(p), ans)
