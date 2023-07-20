import numpy as np
from scipy.stats import norm

# from mfpml.base.model import GPRmodel


class LCB:
    """
    Lower confidence bounding
    """

    def __init__(self, model: GPRmodel) -> None:
        """
        Initialization for LCB
        Parameters
        ----------
        model: SklearnGPRmodel
            single fidelity GPR models
        """
        self.model = model

    def __call__(
        self, x: np.ndarray, explore_factor: float = 1.96
    ) -> np.ndarray:
        """
        Calculate values of LCB acquisition function
        Parameters
        ----------
        x: np.ndarray
            locations for evaluation
        explore_factor: float
            factor to control weight between exploration and exploitation

        Returns
        -------
        lcb: np.ndarray
            lcb values on x
        """
        num_dim = self.model.num_dim
        x = np.array(x).reshape((-1, num_dim))

        y, sigma = self.model.predict(x, return_std=True)
        lcb = y - explore_factor * sigma
        return lcb


class EI:
    """
    Expected improvement acquisition function
    """

    def __init__(self, model: GPRmodel) -> None:
        """
        Initialization of EI acquisition function
        Parameters
        ----------
        model:GPRmodel
            GPR models
        """
        self.model = model

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate value of EI acquisition function
        Parameters
        ----------
        x: np.ndarray
            locations for evaluation
        Returns
        -------
        EI: np.ndarray
            lcb values on x
        """
        num_dim = self.model.num_dim
        x = np.array(x).reshape((-1, num_dim))
        f_min = self.model.y.min()
        y, sigma = self.model.predict(x, return_std=True)
        ei = (f_min - y) * norm.cdf(
            (f_min - y) / (sigma + 1e-9)
        ) + sigma * norm.pdf((f_min - y) / (sigma + 1e-9))

        return -ei


class PI:
    """
    Probability improvement acquisition function
    """

    def __init__(self, model: GPRmodel) -> None:
        """
        Initialization of PI acquisition function
        Parameters
        ----------
        model:GPRmodel
            GPR models
        """
        self.model = model

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate value of PI acquisition function
        Parameters
        ----------
        x: np.ndarray
            locations for evaluation
        Returns
        -------
        PI: np.ndarray
            lcb values on x
        """
        num_dim = self.model.num_dim
        x = np.array(x).reshape((-1, num_dim))
        f_min = self.model.y.min()
        y, sigma = self.model.predict(x, return_std=True)
        pi = norm.cdf((f_min - y) / (sigma + 1e-9))
        return -pi
