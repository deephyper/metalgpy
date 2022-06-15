import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..")

sys.path.insert(0, PKG)

import scipy
import collections
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
        samples = s.sample(size=size, flat=False)

        assert len(samples) == size
        assert list(samples[1].keys()) == list(program.exp().variables().keys())

        samples = s.sample(None, flat=False)

        assert isinstance(samples, dict)

        samples = s.sample(size, flat=True)

        assert isinstance(samples, np.ndarray)
        assert samples.shape[1] == len(program.exp().variables().keys())

        program = f(mpy.Int(1, 5, name="x"), mpy.Float(1, 5, name="y"))
        s = mpy.BaseSampler(program, rng=rng)
        samples = s.sample(flat=True)

        assert isinstance(samples, np.ndarray)
        assert samples.shape[0] == len(program.exp().variables().keys())

        program = f(mpy.List([0,1,2,3,4], ordered=False, name="x"), \
            mpy.List([0, 1, 2, 3, 4, 5], ordered=False, name="y"))
        s = mpy.BaseSampler(program, rng=rng)
        samples = s.sample(flat=False)

        assert isinstance(samples, dict) == True

        str_list = ["aaa", "bbb", "ccc", "ddd"]
        program = mpy.List(values = str_list, ordered=False, name="str_list")
        rng = np.random.RandomState(43)
        s = mpy.BaseSampler(program, rng=rng)
        samples = s.sample()

        assert isinstance(samples, np.ndarray) == True
        assert samples[0] < len(str_list)
        assert program._values[samples[0]] == str_list[samples[0]]

        size = 5
        rng = np.random.RandomState(43)
        y = mpy.Float(1, 5, name="y")
        program = f(mpy.Float(1, 5, name="x"), y)
        dist_map = {
            "x": scipy.stats.cauchy.rvs(loc=1, scale=6, size=size, random_state=rng)
        }
        s = mpy.BaseSampler(program, dist_map, rng=rng)
        samples = s.sample(size)

        assert samples.shape[0] == size
        assert np.array_equal(s._dist_map["x"], dist_map["x"])
        assert isinstance(s._dist_map["y"], collections.abc.Callable)

        size = 7
        rng = np.random.RandomState(43)
        program = f(mpy.Float(1, 5, name="x"), mpy.Int(1, 5, name="y"))
        dist_map = {
            "x": scipy.stats.cauchy.rvs(loc=1, scale=6, size=size, random_state=rng),
            "y": scipy.stats.binom.rvs(n=5, p=0.4, size=size, random_state=rng)
        }
        s = mpy.BaseSampler(program, dist_map, rng=rng)
        samples = s.sample(size)

        assert isinstance(s._dist_map["x"], np.ndarray)
        assert isinstance(s._dist_map["y"], np.ndarray)
        assert np.array_equal(s._dist_map["x"], dist_map["x"])
        assert np.array_equal(s._dist_map["y"], dist_map["y"])

        size = mpy.Int(1, 5, name="size")
        build_nd_array = mpy.meta(np.arange)
        rng = np.random.RandomState(43)
        program = build_nd_array(size)
        s = mpy.BaseSampler(program, rng=rng)
        samples = s.sample(3, flat=False)

        assert isinstance(build_nd_array(samples[0]['size']), collections.abc.Callable)
        assert isinstance(build_nd_array(samples[0]['size']).evaluate(), np.ndarray)
        assert np.array_equal(np.arange(samples[0]['size']), \
                              build_nd_array(samples[0]['size']).evaluate())

