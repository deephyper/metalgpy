import numpy as np


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
