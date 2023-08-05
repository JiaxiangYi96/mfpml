import numpy as np
from scipy.stats import norm

from mfpml.models.sf_gpr import Kriging


# class of lower confidence bounding
# =========================================================================== #
class LCB:
    """Lower confidence bounding"""

    def __init__(self, model: Kriging) -> None:
        """Lower bound confidence acquisition function

        Parameters
        ----------
        model : Kriging
            Kriging model
        """
        self.model = model

    def __call__(self,
                 x: np.ndarray,
                 explore_factor: float = 1.96) -> np.ndarray:
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
        # get dimension of input
        num_dim = self.model.num_dim
        x = x.reshape((-1, num_dim))

        y_hat, sigma = self.model.predict(x, return_std=True)

        # acquisition function
        lcb = y_hat - explore_factor * sigma
        return lcb


# class of expected improvement
# =========================================================================== #


class EI:
    """
    Expected improvement acquisition function
    """

    def __init__(self, model: Kriging) -> None:
        """
        Initialization of EI acquisition function
        Parameters
        ----------
        model: Kriging
           Kriging model
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
        f_min = self.model.sample_Y.min()
        y_hat, sigma = self.model.predict(x, return_std=True)
        # expected improvement
        ei = (f_min - y_hat) * norm.cdf(
            (f_min - y_hat) / (sigma + 1e-9)
        ) + sigma * norm.pdf((f_min - y_hat) / (sigma + 1e-9))

        return -ei

# class of probability improvement
# =========================================================================== #


class PI:
    """
    Probability improvement acquisition function
    """

    def __init__(self, model: Kriging) -> None:
        """
        Initialization of PI acquisition function
        Parameters
        ----------
        model: Kriging
            Kriging model
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
        f_min = self.model.sample_Y.min()
        # get predicted mean and standard deviation
        y_hat, sigma = self.model.predict(x, return_std=True)
        # probability improvement
        pi = norm.cdf((f_min - y_hat) / (sigma + 1e-9))
        return -pi
