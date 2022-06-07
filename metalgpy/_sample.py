import numpy as np

from ._expression import Expression, List, VarExpression, ObjectExpression


LIST_CONSTANT = -1
LIST_CATEGORICAL = -2
LIST_KCATEGORICAL = -3


def list_type(list_exp):

    # oneof case
    if list_exp._k is None:  # "replace" is ignored (only 1 sample is drawn)

        # equivalent to constant 0
        if list_exp._invariant:
            return LIST_CONSTANT

        # equivalent to categorical of len(values)
        else:
            return LIST_CATEGORICAL

    else:  # k >= 1

        # equivalent to constant k so "replace" is ignored
        if list_exp._invariant:
            if isinstance(list_exp._k, VarExpression):
                return list_exp._k.id
            else:
                return LIST_CONSTANT

        # k Categorical of len(values)
        else:

            return LIST_KCATEGORICAL

class BaseSampler:
    def __init__(self, exp, distributions: List[tuple]=None, rng:np.random.RandomState=None):
        self.exp = exp
        self.distributions = distributions
        self.rng = rng
        

class SamplerOld:
    def __init__(self, exp) -> None:
        self.exp = exp
        self.variables = self.exp.variables()
        self.list_variables = {
            k: list_type(v) for k, v in self.variables.items() if isinstance(v, List)
        }

    def sample(self, size, rng):
        choices = self.exp.choices()
        self._cache = sample_from_choices(choices, rng)
        return self._cache

    def filter(self, sample: dict):

        fsample = sample.copy()
        for var_id, var_type in self.list_variables.items():
            if type(var_type) is str or var_type >= 0:
                fsample.pop(var_id)

        return fsample

    def fill(self, fsample: dict):

        sample = fsample.copy()
        for var_id, var_type in self.list_variables.items():
            if type(var_type) is str or var_type >= 0:
                sample[var_id] = sample[var_type]

        return sample


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
            parent_idx = memo[var_id]
            if var_exp._k is not None and var_exp._invariant:
                parent_idx = [i for i in range(parent_idx)]

            if isinstance(parent_idx, list):
                for i in parent_idx:
                    if isinstance(var_exp._getitem(i), Expression):
                        sample_from_choices(
                            var_exp._getitem(i).choices(), rng=rng, memo=memo
                        )
            else:
                if isinstance(var_exp._getitem(parent_idx), Expression):
                    sample_from_choices(
                        var_exp._getitem(parent_idx).choices(), rng=rng, memo=memo
                    )

    return memo


def sample_choice(exp, size=1, rng=None, with_none=False):

    if rng is None:
        rng = np.random.RandomState()

    choices = exp.choices()
    variables = {k: None for k in exp.variables()}

    for _ in range(size):

        choice = sample_from_choices(choices, rng)

        if with_none:
            choice_ = variables.copy()
            choice_.update(choice)
            choice = choice_

        yield choice


def sample(exp, size, rng=None, deepcopy=False):

    if rng is None:
        rng = np.random.RandomState()

    if not(isinstance(exp, Expression)):
        exp = ObjectExpression(exp)

    for variable_choice in sample_choice(exp, size, rng):
        exp_clone = exp.clone(deep=deepcopy)
        exp_clone.freeze(variable_choice)

        yield variable_choice, exp_clone
