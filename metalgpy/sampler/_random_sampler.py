import collections

import numpy as np

from .._expression import VarExpression
from ._base_sampler import BaseSampler


class RandomSampler(BaseSampler):
    """Defines a Base Sampler
    
    Args:
        expression (Expression): a metalgpy expression.
        distributions (dict, optional): preferred distributions for drawing samples. Defaults to None.
        rng (int, optional): a random seed to generate same stream of values.Defaults to None.

    Returns:
        RandomSampler: a sampler object.
    """
    def sample(self, size=None, flat=True):
        """Sample configurations of parameters from the variables available in the expression.

        Args:
            size (int, optional): The number of samples to draw from the search space. If ``size >= 1`` then a list is returned. Defaults to ``None` to draw a single sample..
            flat (bool, optional): If ``True`` then each sample is a 1-dim array. If ``False`` ten each sample is a ``dict`` where keys are variable names. Defaults to ``True``.

        Raises:
            ValueError: if ``size`` has a wrong value.

        Returns:
            (array|dict): an array or a list of dict depending on the choice of input parameters.
        """
        # check for sample size
        if isinstance(size, int) and size > 0:
            # get the required number of samples
            samples = self._sample(size)

            # check for flat flag
            if flat:
                # obtain just the samples from the dict
                samples = np.array([list(sample.values()) for sample in samples])

            # return the samples
            return samples

        if size is None:
            # obtain a sample
            sample = self._sample(size=1)[0]

            # check for flat flag to see if array/dict is expected
            if flat:
                # obtain sample from the dict
                sample = np.array(list(sample.values()))

            # return the sample
            return sample

        # raise value error
        raise ValueError("The value %r cannot be specified as size, size expects 'None' or a positive integer")

    def _sample(self, size):        
        # init a dict to store samples
        self._cache = {}

        # iterate over the possible choices of the expression
        self._dist_map = self._map_dist(self.variables)

        # iterate over the possible choices of the expression
        for var_id, var_exp in self.variables.items():
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
 