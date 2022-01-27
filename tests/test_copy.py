import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..")

sys.path.insert(0, PKG)

import metalgpy as mpy
import numpy as np


@mpy.meta
class Foo:
    def __init__(self, a):
        self.a = a

    def __call__(self, x):
        return self.a - x


class TestCopy(unittest.TestCase):

    def setUp(self):
        # initialization for test
        mpy.VarExpression.var_id = 0 
        
    def test_shallow_clone(self):


        foo = Foo(mpy.List([1, 2, 3]))
        foo_choices = foo.choices()
        assert len(foo_choices) == 1
        assert foo_choices["0"] == mpy.List([1,2,3])
        y = foo(mpy.List([4, 5, 6]))
        y_choices = y.choices()
        assert len(y_choices) == 2
        assert y_choices["0"] == mpy.List([1,2,3])
        assert y_choices["2"] == mpy.List([4,5,6])

        foo_clone = foo.clone()
        foo_clone_choices = foo_clone.choices()
        assert len(foo_clone_choices) == 1
        assert foo_clone_choices["0"] == mpy.List([1,2,3])
        foo_clone.freeze({"0": 0})
        assert foo_clone.args[0].value == 1

        y_clone = y.clone()
        y_clone_choices = y_clone.choices()
        assert len(y_clone_choices) == 2
        assert y_clone_choices["0"] == mpy.List([1,2,3])
        assert y_clone_choices["2"] == mpy.List([4,5,6])

    def test_deep_clone(self):


        foo = Foo(mpy.List([1, 2, 3]))
        foo_choices = foo.choices()
        assert len(foo_choices) == 1
        assert foo_choices["0"] == mpy.List([1,2,3])
        y = foo(mpy.List([4, 5, 6]))
        y_choices = y.choices()
        assert len(y_choices) == 2
        assert y_choices["0"] == mpy.List([1,2,3])
        assert y_choices["2"] == mpy.List([4,5,6])

        foo_clone = foo.clone(deep=True)
        foo_clone_choices = foo_clone.choices()
        assert len(foo_clone_choices) == 1
        assert foo_clone_choices["0"] == mpy.List([1,2,3])
        foo_clone.freeze({"0": 0})
        assert foo_clone.args[0].value == 1

        y_clone = y.clone()
        y_clone_choices = y_clone.choices()
        assert len(y_clone_choices) == 2
        assert y_clone_choices["0"] == mpy.List([1,2,3])
        assert y_clone_choices["2"] == mpy.List([4,5,6])

