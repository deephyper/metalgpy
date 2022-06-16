from ._expression import Expression
from .sampler import BaseSampler, RandomSampler
from .optimizer import Optimizer, BayesianOptimizer


class Evaluation:
    def __init__(self, x: dict, optimizer):
        self.x = x
        self._x_vec = list(x.values())
        self.y = None
        self.optimizer = optimizer

    def report(self, y):
        if self.y is None:
            self.y = y
            self.optimizer.tell(self._x_vec, y, fit=True)


def sample(sampler, optimizer=None, rng=None):

    # check the value passed for sampler
    if isinstance(sampler, Expression):
        sampler = RandomSampler(sampler, rng=rng)
    elif isinstance(sampler, BaseSampler):
        pass
    else:
        raise ValueError(
            "sampler %r is of the wrong type, it should be an expression or a sub-class of BaseSampler"
        )

    # check the value passed for optimizer
    if optimizer is None:
        optimizer = BayesianOptimizer(sampler, random_state=rng)

    variables = list(sampler.expression.variables().keys())
    to_dict = lambda x: {k: v for k, v in zip(variables, x)}

    i = 0
    while True:

        x = optimizer.ask()
        x_dict = to_dict(x)
        yield i, Evaluation(x_dict, optimizer)
        i += 1
