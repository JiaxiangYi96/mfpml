from typing import Any

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import norm

from mfpml.models.kriging import Kriging


class sfSingleObjAcf:
    """base class for sf acquisition functions for single objective
    """

    def query(self, surrogate: Any, params: dict) -> np.ndarray:
        """get the next location to evaluate

        Parameters
        ----------
        surrogate : Any
            Kriging model
        params : dict
            params of bayesian optimization

        Returns
        -------
        opt_x : np.ndarray
            location of update point
        """

        if self.optimizer is None:
            res = differential_evolution(self.eval,
                                         bounds=params['design_space'],
                                         args=(surrogate,),
                                         maxiter=500,
                                         popsize=40)
            # get the next location
            opt_x = res.x

        else:
            # use local optimizer
            _, opt_acq, opt_x = self.optimizer.run_optimizer(
                self.eval,
                num_dim=surrogate.num_dim,
                design_space=params['design_space'],
            )

        return opt_x
# class of lower confidence bounding
# =========================================================================== #


class LCB(sfSingleObjAcf):
    """Lower confidence bounding"""

    def __init__(self,
                 optimizer: Any = None,
                 kappa: list = [1.0, 1.96]) -> None:
        """initialize lcb acquisition function

        Parameters
        ----------
        optimizer : Any
            optimizer for getting update points
        kappa : list, optional
            factors for exploration and exploitation, by default [1.0, 1.96]
        """
        # optimizer for getting update points
        self.optimizer = optimizer
        # kappa for lcb (factor for exploration and exploitation)
        self.kappa = kappa

    def eval(self,
             x: np.ndarray,
             surrogate: Kriging,
             ) -> np.ndarray:
        """
        Calculate values of LCB acquisition function
        Parameters
        ----------
        x: np.ndarray
            locations for evaluation
        surrogate: Kriging
            surrogate model

        Returns
        -------
        lcb: np.ndarray
            lcb values on x
        """
        # get dimension of input
        num_dim = surrogate.num_dim
        # reshape x
        x = x.reshape((-1, num_dim))
        # get predicted mean and standard deviation
        y_hat, sigma = surrogate.predict(x, return_std=True)
        # acquisition function
        lcb = self.kappa[0]*y_hat - self.kappa[1] * sigma
        return lcb


# class of expected improvement
# =========================================================================== #


class EI(sfSingleObjAcf):
    """
    Expected improvement acquisition function
    """

    def __init__(self, optimizer: Any = None) -> None:

        self.optimizer = optimizer

    def eval(self, x: np.ndarray,
             surrogate: Kriging) -> np.ndarray:
        """core of expected improvement function

        Parameters
        ----------
        x : np.ndarray
            locations for evaluation
        surrogate : Kriging
            Kriging model at current iteration

        Returns
        -------
        -ei : np.ndarray
            negative expected improvement values
        """
        num_dim = surrogate.num_dim
        x = np.array(x).reshape((-1, num_dim))
        f_min = surrogate.sample_y.min()
        y_hat, sigma = surrogate.predict(x, return_std=True)
        # expected improvement
        ei = (f_min - y_hat) * norm.cdf(
            (f_min - y_hat) / (sigma + 1e-9)
        ) + sigma * norm.pdf((f_min - y_hat) / (sigma + 1e-9))

        return -ei

# class of probability improvement
# =========================================================================== #


class PI(sfSingleObjAcf):
    """
    Probability improvement acquisition function
    """

    def __init__(self, optimizer: Any = None) -> None:
        """initialization for PI

        Parameters
        ----------
        optimizer : Any, optional
            optimizer for get new location, by default None
        """
        self.optimizer = optimizer

    def eval(self, x: np.ndarray,
             surrogate: Kriging) -> np.ndarray:
        """core of probability improvement function

        Parameters
        ----------
        x : np.ndarray
            locations for evaluation
        surrogate : Kriging
            Kriging model at current iteration

        Returns
        -------
        -pi : np.ndarray
            negative probability improvement values
        """

        num_dim = surrogate.num_dim
        x = np.array(x).reshape((-1, num_dim))
        f_min = surrogate.sample_y.min()
        # get predicted mean and standard deviation
        y_hat, sigma = surrogate.predict(x, return_std=True)
        # probability improvement
        pi = norm.cdf((f_min - y_hat) / (sigma + 1e-9))
        return -pi
