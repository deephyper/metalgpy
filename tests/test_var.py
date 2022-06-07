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

        x1 = mpy.List([0, 1, 2])
        x2 = mpy.List([0, 1, 2])
        assert x1 == x2

        cat_ordinal = mpy.List(values=["low", "medium", "high"], name="cat_ordinal")
        cat_nominal = mpy.List(values=["red", "blue", "green", "yellow"], name="cat_nominal")

    def test_Int(self):

        rng = np.random.RandomState(42)

        v = mpy.Int(0, 10)

        x = v.sample(rng=rng)
        assert x == 6

        x = v.sample(size=2, rng=rng)
        assert len(x) == 2
        assert all(x1 == x2 for x1, x2 in zip(x, [3, 10]))

    def test_Float(self):

        rng = np.random.RandomState(42)

        v = mpy.Float(0, 10)

        x = v.sample(rng=rng)
        assert x == 3.745401188473625

        x = v.sample(size=2, rng=rng)
        assert len(x) == 2
        assert all(x1 == x2 for x1, x2 in zip(x, [9.50714306409916, 7.319939418114051]))
