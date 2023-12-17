from typing import Any

import numpy as np
from scipy.stats import norm


class mfLearningFunction:
    """base class of multi-fidelity learning functions
    """
    @staticmethod
    def _initial_update() -> dict:
        # initialize the update dict
        update_x = {}
        update_x['hf'] = None
        update_x['lf'] = None
        return update_x


class AugmentedEFF(mfLearningFunction):
    """Augmented Expected Feasibility Function

    Parameters
    ----------
    mfLearningFunction : class
        base class of multi-fidelity learning functions
    """

    def eval(
            self,
            x: np.ndarray,
            mf_surrogate: Any,
            ls: float,
            cost_ratio: float,
            fidelity: str,
            **kwargs) -> np.ndarray:

        # get the predictive mean and std of high-fidelity
        y, s = mf_surrogate.predict(x, return_std=True)

        # calculate the correlation value
        alpha1 = self.corr(x, mf_surrogate, fidelity)

        # calculate the fidelity ratio
        if fidelity == 'hf':
            alpha3 = 1.0
        elif fidelity == 'lf':
            alpha3 = cost_ratio
        else:
            ValueError('Unknown fidelity input.')
        # Define the value of e
        e = 2 * s
        # Calculate the EF value
        Z = (ls - y) / s
        EF = (y - ls) * (2 * norm.cdf(Z) - norm.cdf((ls - e - y) / s)
                         - norm.cdf((ls + e - y) / s)) \
            - s * (2 * norm.pdf(Z) - norm.pdf((ls - e - y) / s)
                   - norm.pdf((ls + e - y) / s)) \
            + e * (norm.cdf((ls + e - y) / s) - norm.cdf((ls - e - y) / s))

        # return the augmented expected improvement
        obj = EF * alpha1 * alpha3

        return -obj

    def query(self, mf_surrogate: Any,
              search_x: np.ndarray,
              ls: float = 0.0,
              cost_ratio: float = 0.0,
              **kwargs: Any) -> tuple[np.ndarray, float]:

        update_x = self._initial_update()

        # getting points for high-fidelity
        ef_hf = self.eval(x=search_x,
                          mf_surrogate=mf_surrogate,
                          ls=ls,
                          cost_ratio=cost_ratio,
                          fidelity='hf',
                          **kwargs)

        #  best point for high-fidelity and corresponding objective value
        opt_ef_hf = np.min(ef_hf)
        x_hf = search_x[np.argmin(ef_hf), :]

        # getting points for low-fidelity
        ef_lf = self.eval(x=search_x,
                          mf_surrogate=mf_surrogate,
                          ls=ls,
                          cost_ratio=cost_ratio,
                          fidelity='lf',
                          **kwargs)
        #  best point for high-fidelity and corresponding objective value
        opt_ef_lf = np.min(ef_lf)
        x_lf = search_x[np.argmin(ef_lf), :]
        print(opt_ef_hf, opt_ef_lf, cost_ratio)
        # choose fidelity
        if opt_ef_hf <= opt_ef_lf:
            update_x['hf'] = np.atleast_2d(x_hf)
        else:
            update_x['lf'] = np.atleast_2d(x_lf)

        return update_x

    @staticmethod
    def corr(x: np.ndarray, mf_surrogate: Any, fidelity: str) -> np.ndarray:
        """Evaluate correlation between different fidelity

        Parameters
        ----------
        x : np.ndarray
            point to evaluate
        mf_surrogate : Any
            multi-fidelity surrogate instance
        fidelity : str
            str indicating fidelity level

        Returns
        -------
        np.ndarray
            correlation values
        """
        x = np.atleast_2d(x)
        if fidelity == 'hf':
            return np.ones((x.shape[0], 1))
        elif fidelity == 'lf':
            _, std = mf_surrogate.predict(x, return_std=True)
            _, std_lf = mf_surrogate.predict_lf(x, return_std=True)

            return std_lf / std
        else:
            raise ValueError('Unknown fidelity input.')
