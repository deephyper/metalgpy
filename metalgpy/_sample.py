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


def sample(sampler, optimizer=None, size=None, rng=None):
    """Iterate over optimization steps.

    Args:
        sampler (Expression or BaseSampler): an expression or a subclass from ``BaseSampler``.
        optimizer (Optimizer, optional): a subclass from ``Optimizer``. Defaults to ``None`` for ``BayesianOptimizer``.
        size (int, optional): the number of iterations to perform. Defaults to ``None`` for infinite.
        rng (int or np.RandomState, optional): the random state of the optimization process. Defaults to ``None``.

    Raises:
        ValueError: if the sampler is of the wrong type.

    Yields:
        int, Evaluation: a tuple corresponding to the iteration and an ``Evaluation`` object.
    """

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
        i += 1

        yield i, Evaluation(x_dict, optimizer)

        # exit
        if size is not None and i >= size:
            break
