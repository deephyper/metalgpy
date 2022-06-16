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
from metalgpy.sampler import BaseSampler


@mpy.meta
def f(x, y):
    return x * y


@mpy.meta
class Identity:
    def __repr__(self) -> str:
        return "Identity"


@mpy.meta
class Dense:
    def __init__(self, units, activation) -> None:
        self.units = units
        self.activation = activation

    def __repr__(self) -> str:
        return f"Dense({self.units}, {self.activation})"


@mpy.meta
class Net:
    def __init__(self, layers):
        self.layers = layers

    def __repr__(self) -> str:
        out = "Net("
        for l in self.layers:
            out += f"\n\t{l}"
        out += "\n)"
        return out


class TestSampler(unittest.TestCase):
    def test_BaseSampler(self):

        rng = np.random.RandomState(42)

        program = f(mpy.Float(1, 5, name="x"), mpy.Float(1, 5, name="y"))
        s = BaseSampler(program, rng=rng)
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
        s = BaseSampler(program, rng=rng)
        samples = s.sample(flat=True)

        assert isinstance(samples, np.ndarray)
        assert samples.shape[0] == len(program.exp().variables().keys())

        program = f(
            mpy.List([0, 1, 2, 3, 4], ordered=False, name="x"),
            mpy.List([0, 1, 2, 3, 4, 5], ordered=False, name="y"),
        )
        s = BaseSampler(program, rng=rng)
        samples = s.sample(flat=False)

        assert isinstance(samples, dict) == True

        str_list = ["aaa", "bbb", "ccc", "ddd"]
        program = mpy.List(values=str_list, ordered=False, name="str_list")
        s = BaseSampler(program, rng=rng)
        samples = s.sample()

        assert isinstance(samples, np.ndarray) == True
        assert samples[0] < len(str_list)
        assert program._values[samples[0]] == str_list[samples[0]]

        size = 5
        y = mpy.Float(1, 5, name="y")
        program = f(mpy.Float(1, 5, name="x"), y)
        dist_map = {"x": (scipy.stats.cauchy, {"loc": 1, "scale": 4})}
        s = BaseSampler(program, dist_map, rng=rng)
        samples = s.sample(size)

        assert samples.shape[0] == size
        assert np.array_equal(s._dist_map["x"], dist_map["x"])
        assert isinstance(s._dist_map["y"][0], collections.abc.Callable)

        size = 7
        program = f(mpy.Float(1, 5, name="x"), mpy.Int(1, 5, name="y"))
        dist_map = {
            "x": (scipy.stats.cauchy, {"loc": 1, "scale": 4}),
            "y": (scipy.stats.binom, {"n": 5, "p": 0.4}),
        }
        s = BaseSampler(program, dist_map, rng=rng)
        samples = s.sample(size)

        assert isinstance(s._dist_map["x"][1], dict)
        assert isinstance(s._dist_map["y"][1], dict)

        var_x = mpy.List(
            [
                mpy.Int(1, 3, name="vx1"),
                mpy.Float(4, 6, name="vx2"),
                mpy.Int(7, 9, name="vx3"),
            ],
            ordered=False,
            name="var_x",
        )

        var_y = mpy.List(
            [
                mpy.Float(10, 12, name="vy1"),
                mpy.Int(13, 15, name="vy2"),
                mpy.Float(16, 18, name="vy3"),
            ],
            ordered=False,
            name="var_y",
        )
        dist_map = {
            "vx2": (scipy.stats.cauchy, {"loc": 4, "scale": 2}),
            ("vy1", "vy3"): (scipy.stats.multivariate_normal, {"mean": 10, "cov": 8}),
        }
        program = f(var_x, var_y)
        s = BaseSampler(program, dist_map, rng=rng)
        samples = s.sample(3, flat=False)

        assert isinstance(samples[0], dict)
        assert samples[0].keys() == program.variables().keys()
        assert samples[0]["vx1"] > program.variables()["vx1"]._low
        assert samples[0]["vy3"] < program.variables()["vy3"]._high
        assert samples[0]["var_x"] < program.variables()["var_x"]._length()
        assert s._dist_map["vx2"][0] == dist_map["vx2"][0]
        assert s._dist_map["vy1"][0] == s._dist_map["vy3"][0]

        size = mpy.Int(1, 5, name="size")
        build_nd_array = mpy.meta(np.arange)
        program = build_nd_array(size)
        s = BaseSampler(program, rng=rng)
        samples = s.sample(3, flat=False)

        assert isinstance(build_nd_array(samples[0]["size"]), collections.abc.Callable)
        assert isinstance(build_nd_array(samples[0]["size"]).evaluate(), np.ndarray)
        assert np.array_equal(
            np.arange(samples[0]["size"]), build_nd_array(samples[0]["size"]).evaluate()
        )

        layer = mpy.List(
            [
                Identity(),
                Dense(
                    units=mpy.Int(1, 16, name="units"),
                    activation=mpy.List(
                        ["identity", "sigmoid", "relu"], name="activation"
                    ),
                ),
            ],
            name="layer",
        )

        program = Net([layer])
        s = BaseSampler(program, rng=rng)
        samples = s.sample(10, flat=False)
        frozen_model_config = samples[-1]
        act_fn = program.variables()["activation"]._values[
            frozen_model_config["activation"]
        ]
        num_units = frozen_model_config["units"]
        program.freeze(frozen_model_config)
        eval_model = program.evaluate()

        assert eval_model.layers[0].activation == act_fn
        assert eval_model.layers[0].units == num_units
