import os
import sys
import unittest


HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..")

sys.path.insert(0, PKG)

import metalgpy as mpy
import numpy as np

@mpy.meta
def f(x):
    return x

@mpy.meta
def g(x):
    return -x

@mpy.meta
def h(x):
    return x+1


class TestHierarchical(unittest.TestCase):

    def setUp(self):
        # initialization for test 
        mpy.VarExpression.var_id = 0 

    def test_1(self):

        program = h(mpy.List([f, g])(1))
        choices = program.choices()
        assert len(choices) == 1
        assert choices["0"] == mpy.List([f,g])
        program.freeze({"0": 0})
        res = program.evaluate()
        assert res == 2

    def test_2(self):
        program = h(mpy.List([f(mpy.List([1,3,5])), g(mpy.List([2,4,6]))]))
        choices = program.choices()
        assert len(choices) == 1
        program.freeze({"2":0, "0":2})
        res = program.evaluate()
        assert res == 6

