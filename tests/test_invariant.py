import os
import sys
import unittest


HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..")

sys.path.insert(0, PKG)

import metalgpy as mpy
import numpy as np


@mpy.meta
def layer(num_units):
    return f"layer_{num_units}"


@mpy.meta
def network(layers):
    return layers

@unittest.skip
class TestInvariant(unittest.TestCase):
    def setUp(self):
        # initialization for test
        mpy.VarExpression.var_id = 0

    def test_1(self):

        rng = np.random.RandomState(42)

        max_layers = 5
        num_layers = mpy.Int(1, max_layers, name="num_layers")
        layers = (
            ["input_layer"]
            + mpy.List(
                values=[
                    mpy.List([f"dense_{i}", f"conv_{i}"], name=f"layer_{i}")
                    for i in range(max_layers)
                ],
                k=num_layers,
                invariant=True,
                name="layers",
            )
            + ["output_layer"]
        )
        net = network(layers)

        variables = net.variables()
        assert list(variables.keys()) == [
            "layers",
            "num_layers",
            "layer_0",
            "layer_1",
            "layer_2",
            "layer_3",
            "layer_4",
        ]

        for _, sample_exp in mpy.sample(net, size=1, rng=rng):
            res = sample_exp.evaluate()

            assert res == [
                "input_layer",
                "dense_0",
                "dense_1",
                "dense_2",
                "conv_3",
                "output_layer",
            ]

        choice = list(mpy.sample_choice(net, size=2, rng=rng, with_none=True))
        print(choice)
