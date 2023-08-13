from typing import Any

import numpy as np
from scipy.stats import norm

from mfpml.models.sf_gpr import Kriging


class SFLearningfunction(object):

    def query(self, surrogate: Kriging,
              search_x: np.ndarray,
              ls: float = 0.0) -> tuple[Any, np.ndarray]:

        obj = self.eval(x=search_x, surrogate=surrogate, ls=ls)

        # Find the best objective value and corresponding input point
        update_y = np.min(obj)
        update_x = search_x[np.argmin(obj), :]

        return update_x, update_y


class EFF(SFLearningfunction):

    def eval(self, x: np.ndarray,
             surrogate: Kriging,
             ls: float = 0.0) -> np.ndarray:
        """
        Calculate values of expected feasibility function
        Parameters
        ----------
        x: np.ndarray
            locations for evaluation
        surrogate: Kriging
            surrogate model

        Returns
        -------
        eff: np.ndarray
            expected feasibility function values on x
        """
        y, s = surrogate.predict(x, return_std=True)
        # Define the value of e
        e = 2 * s
        # Calculate the EF value
        Z = (ls - y) / s
        EF = (y - ls) * (2 * norm.cdf(Z) - norm.cdf((ls - e - y) / s)
                         - norm.cdf((ls + e - y) / s)) \
            - s * (2 * norm.pdf(Z) - norm.pdf((ls - e - y) / s)
                   - norm.pdf((ls + e - y) / s)) \
            + e * (norm.cdf((ls + e - y) / s) - norm.cdf((ls - e - y) / s))

        # Return minus EF because the optimizer minimizes the objective
        obj = -EF

        return obj


class U(SFLearningfunction):

    def eval(self, surrogate: Kriging,
             x: np.ndarray,
             ls: float = 0.0) -> np.ndarray:
        """
        Calculate values of expected feasibility function
        Parameters
        ----------
        x: np.ndarray
            locations for evaluation
        surrogate: Kriging
            surrogate model

        Returns
        -------
        eff: np.ndarray
            expected feasibility function values on x
        """

        # Predict the Kriging model response and its mean squared error (mse)
        y, mse = surrogate.predict(x, return_std=True)

        # Calculate U function
        U = np.abs(y-ls) / (mse + 1e-6)

        return U
