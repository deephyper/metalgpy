import os
import sys
import unittest


HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..")

sys.path.insert(0, PKG)

import metalgpy as mpy


class TestMemoization(unittest.TestCase):
    def test_1(self):

        @mpy.meta
        def foo(x1, x2):
            return x1, x2
        
        xi = mpy.Int(0, 10, name="xi")

        # the same variable is reused we should have only 1 item in choices
        exp = foo(xi, xi)

        choices = exp.choices()
        assert len(choices) == 1
        assert "xi" in choices

        exp.freeze({"xi": 0})
        x1, x2 = exp.evaluate()
        assert x1 == 0 and x1 == x2

