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
def f(x):
    return 1 + x


@mpy.meta
def g(x):
    return x - 1


@mpy.meta
def f_generator(c):
    return lambda x: x + c


class TestFunctional(unittest.TestCase):

    def setUp(self):
        # initialization for test 
        mpy.VarExpression.var_id = 0 

    def test_function(self):

        # f is a meta, evaluation should return the original function
        assert inspect.isfunction(f.evaluate())

        # test a program applying 1+x
        program = f(0)
        assert type(program) is mpy.FunctionCallExpression
        assert program.choices() == {}
        program.freeze({})
        res = program.evaluate()
        assert res == 1

        # test a program with a List variable
        program = f(mpy.List([0, 1, 2]))
        choices = program.choices()
        assert len(choices) == 1
        assert choices["0"] == mpy.List([0, 1, 2])

        choice = {"0":2}
        program.freeze(choice)
        res = program.evaluate()
        assert res == 3

    def test_composition(self):

        program = f(g(0) + mpy.List([1, 2]))
        choices = program.choices()
        assert len(choices) == 1
        assert choices["0"] == mpy.List([1, 2])
        choice = {"0":0}
        program.freeze(choice)
        res = program.evaluate()
        assert res == 1

    def test_function_generator(self):

        program = f_generator(mpy.List([0, 1, 2]))(1) + 1
        choices = program.choices()
        assert len(choices) == 1
        assert choices["0"] == mpy.List([0, 1, 2])
        choice = {"0":2}
        program.freeze(choice)
        res = program.evaluate()
        assert res == 4

    def test_builtin_function(self):

        program = mpy.meta(sum)([1 for i in range(10)])
        res = program.evaluate()
        assert res == 10
