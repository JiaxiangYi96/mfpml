from typing import Any

import numpy as np
from scipy.spatial import distance_matrix
from scipy.stats import norm

from mfpml.models.kriging import Kriging


class SFLearningFunction(object):
    """Base class for learning functions

    Parameters
    ----------
    surrogate: Kriging
        surrogate model
    """

    def query(self, surrogate: Kriging,
              search_x: np.ndarray,
              ls: float = 0.0,
              **kwargs: Any) -> tuple[Any, np.ndarray]:
        """get the next query point

        Parameters
        ----------
        surrogate : Kriging
            Kriging model
        search_x : np.ndarray
            Monte Carlo samples
        ls : float, optional
            limit state of reliability analysis problem, by default 0.0

        Returns
        -------
        tuple[Any, np.ndarray]
            a tuple of next query point and its corresponding objective value
        """

        obj = self.eval(x=search_x, surrogate=surrogate, ls=ls, **kwargs)

        # Find the best objective value and corresponding input point
        update_y = np.min(obj)
        update_x = search_x[np.argmin(obj), :]

        return update_x, update_y


class EFF(SFLearningFunction):
    """Expected Feasibility Function

    Parameters
    ----------
    SFLearningFunction : class
        base class
    """

    def eval(self, x: np.ndarray,
             surrogate: Kriging,
             ls: float = 0.0,
             **kwargs: Any) -> np.ndarray:
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


class U(SFLearningFunction):
    """Learning function U

    Parameters
    ----------
    SFLearningFunction : class
        base class
    """

    def eval(self, surrogate: Kriging,
             x: np.ndarray,
             ls: float = 0.0,
             **kwargs: Any) -> np.ndarray:
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


class RLCB(SFLearningFunction):
    """Learning function RLCB

    Parameters
    ----------
    SFLearningFunction : class
        base class
    """

    def eval(self, surrogate: Kriging,
             x: np.ndarray,
             ls: float = 0.0,
             **kwargs: Any) -> np.ndarray:
        """evaluate the RLCB value

        Parameters
        ----------
        surrogate : Kriging
            surrogate model
        x : np.ndarray
            search points
        ls : float, optional
            limit state, by default 0.0

        Returns
        -------
        rlcb : np.ndarray
            rlcb values
        """

        y, mse = surrogate.predict(x, return_std=True)

        # Get the sampled points
        sample_x = kwargs["sample_x"]

        y = np.abs(y-ls)

        # calculate rlcb value
        rlcb = y - norm.pdf(y/mse) * mse

        # find the minimum distance among sample_x
        d0 = distance_matrix(sample_x, sample_x)
        d0[d0 == 0] = np.Inf
        d_threshold = np.min(d0)

        # values of sample_x and search_x
        d2 = np.min(d_threshold - distance_matrix(sample_x, x), axis=0)
        # set d2<0 to 0
        d2[d2 < 0] = 0

        rlcb = rlcb + np.reshape(d2 * 1e4, (-1, 1))

        return rlcb
