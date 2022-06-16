import os
from re import A
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..")

sys.path.insert(0, PKG)

import metalgpy as mpy
from metalgpy.optimizer import BayesianOptimizer

@mpy.meta
def f(x):
    return x

class TestOptimizer(unittest.TestCase):


    def test_bayesian_optimizer(self):
        bb = f(mpy.Float(0, 10))

        variables = list(bb.variables().keys())
        to_dict = lambda x: {k:v for k,v in zip(variables, x)}

        opt = BayesianOptimizer(bb, random_state=42)
        results = []
        for i in range(20):
            x = opt.ask()
            x_dict = to_dict(x)
            frozen_bb = bb.clone().freeze(x_dict)
            y = frozen_bb.evaluate()
            opt.tell(x, y, fit=True)
            results.append(y)

        assert min(results) < 0.1
