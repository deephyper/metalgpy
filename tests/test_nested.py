import os
import sys
import unittest


HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..")

sys.path.insert(0, PKG)

import metalgpy as mpy
import numpy as np


@mpy.meta
def layers(*args):
    return args


class TestNested(unittest.TestCase):
    def test_Int(self):

        rng = np.random.RandomState(42)

        # meta program
        units_layer1 = mpy.Int(32, 512)
        units_layer2 = mpy.Int(32, units_layer1)
        units_layer3 = mpy.Int(32, units_layer2)
        units_layer4 = mpy.Int(32, units_layer3)

        exp = layers(units_layer1, units_layer2, units_layer3, units_layer4)

        for _, sample_program in mpy.sample(exp, size=1, rng=rng):
            units = sample_program.evaluate()
            assert units == (220, 52, 38, 33)

    def test_Float(self):

        rng = np.random.RandomState(42)

        # meta program
        units_layer1 = mpy.Float(32, 512)
        units_layer2 = mpy.Float(32, units_layer1)
        units_layer3 = mpy.Float(32, units_layer2)
        units_layer4 = mpy.Float(32, units_layer3)

        exp = layers(units_layer1, units_layer2, units_layer3, units_layer4)

        for _, sample_program in mpy.sample(exp, size=1, rng=rng):
            units = sample_program.evaluate()
            print(units)
            assert units == (59.88013384073574, 56.14910687385782, 46.516390662067046, 42.27865815638425)
