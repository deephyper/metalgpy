import numpy as np

from ._expression import List, Expression


def sample_values_from_choices(choices: list, rng) -> dict:
    """Sample variables from a list of variable choice.

    Args:
        choices (list): List of variables to sample.
        rng (np.random.RandomState): the random state.

    Returns:
        dict: keys are variable ids, values are variables values.
    """
    s = [] # var samples
    for var_exp in choices:

        s.append(var_exp.sample(rng=rng))

        if isinstance(var_exp, List) and isinstance(var_exp[s[-1]], Expression):
            d = sample_values_from_choices(var_exp[s[-1]].choice(), rng)
            s.extend(d)
        
    return s


def sample_values(exp, size=1, rng=None):

    if rng is None:
        rng = np.random.RandomState()

    choices = exp.choice()
    for _ in range(size):

        variable_choice = sample_values_from_choices(choices, rng)
        yield variable_choice


def sample(exp, size, rng=None, deepcopy=False):

    if rng is None:
        rng = np.random.RandomState()

    for variable_choice in sample_values(exp, size, rng):
        exp_clone = exp.clone(deep=deepcopy)
        exp_clone.freeze(variable_choice)

        yield variable_choice, exp_clone

