import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..")

sys.path.insert(0, PKG)

import metalgpy as mpy


@mpy.meta
class Foo:
    def __init__(self, a):
        self.a = a

    def __call__(self, x):
        return self.a - x


class TestFunctional(unittest.TestCase):
    def setUp(self):
        # initialization for test
        mpy.VarExpression.var_id = 0

    def test_class_call(self):

        foo = Foo(mpy.List([1, 2, 3]))
        foo_choices = foo.choices()
        assert len(foo_choices) == 1
        assert foo_choices["0"] == mpy.List([1, 2, 3])
        y = foo(mpy.List([4, 5, 6]))
        y_choices = y.choices()
        assert len(y_choices) == 2
        assert y_choices["0"] == mpy.List([1, 2, 3])
        assert y_choices["2"] == mpy.List([4, 5, 6])
        choice = {"0": 2, "2": 0}
        y.freeze(choice)
        res = y.evaluate()
        assert res == -1

    def test_attribute_access(self):

        foo = Foo(mpy.List([1, 2, 3]))
        foo_choices = foo.choices()
        assert len(foo_choices) == 1
        assert foo_choices["0"] == mpy.List([1, 2, 3])
        a = foo.a
        a.freeze({"0": 0})
        res = a.evaluate()
        assert res == 1
