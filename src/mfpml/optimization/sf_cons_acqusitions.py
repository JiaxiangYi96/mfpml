from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import norm

from mfpml.models.gaussian_process import GaussianProcessRegression as Kriging


class SFConsAcq(ABC):
    """base class for sf acquisition functions for single objective
    """

    def __init__(self,
                 optimizer: Any) -> None:

        # optimizer for getting update points
        self.optimizer = optimizer

    @abstractmethod
    def eval(self,
             x: np.ndarray,
             obj_surrogate: Kriging,
             cons_surrogates: List) -> np.ndarray:
        """core of acquisition function

        Parameters
        ----------
        x : np.ndarray
            locations for evaluation
        surrogate : Kriging
            Kriging model at current iteration
        

        Returns
        -------
        np.ndarray
            values of acquisition function
        """
        raise NotImplementedError("eval method is not implemented")

    def query(self,
              obj_surrogate: Kriging,
              cons_surrogate: List) -> np.ndarray:
        """get the next location to evaluate

        Parameters
        ----------
        surrogate : Kriging
            Gaussian process regression model
        Returns
        -------
        opt_x : np.ndarray
            location of update point
        """

        if self.optimizer is None:
            res = differential_evolution(self.eval,
                                         bounds=obj_surrogate.bounds,
                                         args=(obj_surrogate,cons_surrogate),
                                         maxiter=500,
                                         popsize=40)
            # get the next location
            opt_x = res.x

        else:
            _, _, opt_x = self.optimizer.run_optimizer(
                self.eval,
                num_dim=obj_surrogate.num_dim,
                design_space=obj_surrogate.bounds,
            )

        return opt_x

class CEI(SFConsAcq):
    """
    Constrained Expected improvement acquisition function
    """

    def __init__(self,
                 optimizer: Any = None) -> None:
        super(CEI, self).__init__(optimizer=optimizer)

    def eval(self,
             x: np.ndarray,
             obj_surrogate: Kriging,
             cons_surrogates: List) -> np.ndarray:
        """core of expected improvement function

        Parameters
        ----------
        x : np.ndarray
            locations for evaluation
        surrogate : Kriging
            Kriging model at current iteration

        Returns
        -------
        cei : np.ndarray
            negative expected improvement values
        """
        
        # get the expected improvement of objective function
        ei_obj = self.ei(x, obj_surrogate)
        # get the probability of feasibility
        pof = self.pof(x, cons_surrogates)
        # get the expected improvement of constraints
        cei = -ei_obj * pof

        return cei
    
    def ei(self, x, obj_surrogate: Kriging) -> np.ndarray:

        num_dim = obj_surrogate.num_dim
        x = np.array(x).reshape((-1, num_dim))
        f_min = obj_surrogate.sample_y.min()
        y_hat, sigma = obj_surrogate.predict(x, return_std=True)
        # expected improvement
        ei = (f_min - y_hat) * norm.cdf(
            (f_min - y_hat) / (sigma + 1e-9)
        ) + sigma * norm.pdf((f_min - y_hat) / (sigma + 1e-9))
        return ei

    def pof(self, x, cons_surrogates: List) -> np.ndarray:
        num_cons = len(cons_surrogates)
        u_g = np.zeros((x.shape[0], num_cons))
        mse_g = np.zeros((x.shape[0], num_cons))
        for ii in range(num_cons):
            cons_surrogate = cons_surrogates[ii]
            u_g[:, ii], mse_g[:, ii] = cons_surrogate.predict(x, return_std=True)

        # probability of feasibility
        pof = np.prod(norm.cdf(u_g / (mse_g + 1e-9)), axis=1)
        return pof

