import numpy as np

from ._expression import List, Expression


def sample_from_choices(choices: dict, rng, memo=None) -> dict:
    """Sample variables from a list of variable choice.

    Args:
        choices (list): List of variables to sample.
        rng (np.random.RandomState): the random state.

    Returns:
        dict: keys are variable ids, values are variables values.
    """
    memo = memo if memo else {}  # memoization

    for var_id, var_exp in choices.items():

        var_exp.sample(rng=rng, memo=memo)

        if isinstance(var_exp, List):
            if isinstance(memo[var_id], list):
                for i in memo[var_id]:
                    if isinstance(var_exp._getitem(i), Expression):
                        sample_from_choices(var_exp._getitem(i).choices(), rng=rng, memo=memo)
            else:
                if isinstance(var_exp._getitem(memo[var_id]), Expression):
                    sample_from_choices(var_exp._getitem(memo[var_id]).choices(), rng=rng, memo=memo)

    return memo


def sample_choice(exp, size=1, rng=None):

    if rng is None:
        rng = np.random.RandomState()

    choices = exp.choices()
    for _ in range(size):

        variable_choice = sample_from_choices(choices, rng)
        yield variable_choice


def sample(exp, size, rng=None, deepcopy=False):

    if rng is None:
        rng = np.random.RandomState()

    for variable_choice in sample_choice(exp, size, rng):
        exp_clone = exp.clone(deep=deepcopy)
        exp_clone.freeze(variable_choice)

        yield variable_choice, exp_clone

