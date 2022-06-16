import collections
import numpy as np
import scipy
import inspect

from ._expression import Expression, List, Int, Float, VarExpression, ObjectExpression


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
    """Defines a Base Sampler
    
    Args:
        exp (Expression): a metalgpy expression.
        distributions (dict, optional): preferred distributions for drawing samples. Defaults to None.
        rng (int, optional): a random seed to generate same stream of values.Defaults to None.

    Returns:
        BaseSampler: a sampler object.
    """
    def __init__(self,
                 exp,
                 distributions: dict=None,
                 rng:np.random.RandomState=None):
        self.exp = exp
        self.distributions = distributions
        self.rng = check_random_state(rng)
        self._choices = self.exp.choices()

    def sample(self, size=None, flat=True):
        # check for sample size
        if isinstance(size, int) and size > 0:
            # get the required number of samples
            samples = self._sample(self._choices, size)

            # check for flat flag
            if flat:
                # obtain just the samples from the dict
                samples = np.array([list(sample.values()) for sample in samples])

            # return the samples
            return samples

        if size is None:
            # obtain a sample
            sample = self._sample(self._choices, size=1)[0]

            # check for flat flag to see if array/dict is expected
            if flat:
                # obtain sample from the dict
                sample = np.array(list(sample.values()))

            # return the sample
            return sample

        # raise value error
        raise ValueError("The value %r cannot be specified as size, size expects an integer")

    def _sample(self, choices, size):        
        # init a dict to store samples
        self._cache = {}

        # iterate over the possible choices of the expression
        self._dist_map = self._map_dist(self.exp.variables())

        # iterate over the possible choices of the expression
        for var_id, var_exp in self.exp.variables().items():
            # check if the expression is Float
            if isinstance(var_exp, VarExpression):
                dist, params = self._dist_map[var_id]
                self._cache[var_id] = dist.rvs(**params, size=size, random_state=self.rng)

        # arrange the values by key
        self._cache = [{var_id: self._cache[var_id][i] \
                        for var_id in self._cache.keys()} for i in range(size)]

        # return the samples
        return self._cache

    def _map_dist(self, variables):
        # map the default dist map
        self._dist_map = {}

        # check if the distributions are passed
        if self.distributions is None:
            self._dist_map = {var_id: var_exp._dist if isinstance(var_exp, VarExpression) \
                              else None for var_id, var_exp in variables.items()}

        # check if a distribution dict is passed
        if isinstance(self.distributions, dict):
            for var_id, distribution in self.distributions.items():
                # collect respective distribution by key
                if isinstance(var_id, str) and var_id in variables.keys():
                    self._dist_map[var_id] = distribution

                # if the variable is passed in an iterable
                if isinstance(var_id, collections.abc.Iterable):
                    for var in var_id:
                        # check if the variable is valid
                        if var in variables.keys():
                            self._dist_map[var] = distribution

            # map default distribution for the remaining variables
            for var_id, var_exp in variables.items():
                if isinstance(var_exp, VarExpression) and var_id not in self._dist_map:
                    self._dist_map[var_id] = var_exp._dist

        # return the distribution map
        return self._dist_map
 
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

def check_random_state(rng):
    """Checks the seed provided to create a RandomState instance
    
    Courtesy : https://scikit-learn.org/stable/modules/generated/sklearn.utils.check_random_state.html

    Args:
        rng (int ? np.random.RandomState): a random state instance or an integer seed.

    Returns:
        Random State Object (np.random.RandomState)
    """
    if rng is None:
        return np.random.mtrand._rand
    if isinstance(rng, int):
        return np.random.RandomState(rng)
    if isinstance(rng, np.random.RandomState):
        return rng
    raise ValueError("The value %r cannot be used for creating a random state instance")

