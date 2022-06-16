import abc
import collections

import numpy as np

from .._expression import VarExpression
from ._utils import check_random_state

class BaseSampler(abc.ABC):

    def __init__(self,
                 expression,
                 distributions: dict=None,
                 rng:np.random.RandomState=None):
        self.expression = expression
        self.variables = self.expression.variables()
        self.distributions = distributions
        self.rng = check_random_state(rng)

    def set_random_state(self, rng):
        self.rng = check_random_state(rng)

    @abc.abstractmethod
    def sample(self, size=None, flat=True):
        pass
