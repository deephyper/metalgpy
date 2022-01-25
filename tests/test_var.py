import os
from re import A
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..")

sys.path.insert(0, PKG)

import numpy as np

import metalgpy as mpy

class TestVar(unittest.TestCase):


    def test_List(self):
        
        rng = np.random.RandomState(42)

        v = mpy.List(["0", 1, 2.0])
        assert v[0] == "0"
        assert v[1] == 1
        assert v[2] == 2.0

        idx = v.sample(rng=rng)
        assert idx == 2
        
        idx = v.sample(size=2, rng=rng)
        assert all(x1 == x2 for x1, x2 in zip(idx, [0, 2]))

    def test_Int(self):

        rng = np.random.RandomState(42)
        
        v = mpy.Int(0, 10)
        
        x = v.sample(rng=rng)
        assert x == 6
        
        x = v.sample(size=2, rng=rng)
        assert len(x) == 2
        assert all(x1 == x2 for x1, x2 in zip(x, [3, 10]))

    # def test_Combination(self):

    #     rng = np.random.RandomState(42)

    #     v = mpy.Combination(3, [i for i in range(10)])

    #     x = v.sample(rng=rng)
    #     assert all(x1 == x2 for x1, x2 in zip(x, [8, 1, 5]))

    #     x = v.sample(size=2, rng=rng)
    #     assert len(x) == 2
    #     assert all(x1 == x2 for x1, x2 in zip(x[0], [0, 1, 8]))
    #     assert all(x1 == x2 for x1, x2 in zip(x[1], [9, 2, 0]))

