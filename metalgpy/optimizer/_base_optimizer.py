import abc


class Optimizer(abc.ABC):
    @abc.abstractmethod
    def ask(self, n_points=None, strategy=None):
        """Ask new configurations to the optimizer.

        Args:
            n_points (int, optional): The number of configurations to return. If ``None`` a single configuration is returned. Defaults to None.
            strategy (str, optional): The strategy to follow when multiple points are requested at once. Defaults to None.

        Returns:
            (list): A list of configurations or a single configuration.
        """
        pass

    @abc.abstractmethod
    def tell(self, x, y, fit=True):
        pass
