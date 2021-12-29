import inspect
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


class TestFunctional(unittest.TestCase):
    def test_class_call(self):

        rng = np.random.RandomState(42)

        foo = Foo(mpy.List([1, 2, 3]))
        foo_choices = foo.choice()
        assert len(foo_choices) == 1
        assert foo_choices[0] == mpy.List([1,2,3])
        y = foo(mpy.List([4, 5, 6]))
        y_choices = y.choice()
        assert len(y_choices) == 2
        assert y_choices[0] == mpy.List([1,2,3])
        assert y_choices[1] == mpy.List([4,5,6])
        choice = [v.sample(rng=rng) for v in y.choice()]
        assert choice == [2, 0]
        y.freeze(choice)
        res = y.evaluate()
        assert res == -1

    def test_attribute_access(self):

        foo = Foo(mpy.List([1, 2, 3]))
        foo_choices = foo.choice()
        assert len(foo_choices) == 1
        assert foo_choices[0] == mpy.List([1,2,3])
        a = foo.a
        a.freeze([0])
        res = a.evaluate()
        assert res == 1
