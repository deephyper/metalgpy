import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..")

sys.path.insert(0, PKG)

import numpy as np
import metalgpy as mpy

@mpy.meta
def f(x, y):
    return x*y

class TestSampler(unittest.TestCase):

    def test_BaseSampler(self):

        rng = np.random.RandomState(42)

        program = f(mpy.Float(1, 5, name="x"), mpy.Float(1, 5, name="y"))
        s = mpy.BaseSampler(program, rng=rng)
        size = 5
        samples = s.sample(size=size)

        assert samples.shape[0] == len(program.exp().variables().keys())
        assert samples.shape[1] == size

        a = mpy.Int(10, 25, name="a")
        s_a = mpy.BaseSampler(a, rng=rng)
        samples = s_a.sample()

        assert samples.shape[0] == len(a.exp().variables().keys())
        assert samples.shape[1] == 1


