import os
from re import A
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..")

sys.path.insert(0, PKG)

import numpy as np

import metalgpy as mpy

class TestVar(unittest.TestCase):


    def test_List(self):

        v = mpy.List(["0", 1, 2.0])
        assert v[0] == "0"
        assert v[1] == 1
        assert v[2] == 2.0

        x1 = mpy.List([0, 1, 2])
        x2 = mpy.List([0, 1, 2])
        assert x1 == x2

        cat_ordinal = mpy.List(values=["low", "medium", "high"], ordered=True, name="cat_ordinal")
        cat_nominal = mpy.List(values=["red", "blue", "green", "yellow"], ordered=False, name="cat_nominal")
        assert cat_ordinal._ordered == True
        assert cat_nominal._ordered == False

    def test_Int(self):

        v = mpy.Int(0, 10)
        v._low = 0
        v._high = 10
        assert hasattr(v, "_dist")

    def test_Float(self):

        v = mpy.Float(0, 10)
        v._low = 0
        v._high = 10
        assert hasattr(v, "_dist")
