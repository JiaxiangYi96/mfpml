
from typing import Any

import numpy as np
from scipy.stats import norm
from sklearn.utils import resample

from .sf_learning_functions import EFF, U


class EFFStoppingCriteria:

    def __init__(self) -> None:
        # define the learning function
        self.learning_function = EFF()

    def stopping_value(self, search_x: np.ndarray,
                       surrogate: Any,
                       iter: int,
                       **kwargs: Any) -> float:
        """get the stopping value

        Parameters
        ----------
        search_x : np.ndarray
            Monte Carlo samples
        surrogate : Kriging
            surrogate model
        iter : int
            iteration number
        kwargs : Any
            other parameters

        Returns
        -------
        stopping value : float
            stopping value
        """

        if iter == 0:
            stopping_value = np.abs(np.min(self.learning_function.eval(
                x=search_x,
                surrogate=surrogate)))
        else:
            # use learning function value as stopping value
            stopping_value = np.abs(kwargs["lf_value"])
        return stopping_value


class UStoppingCriteria:
    """use the negative value"""

    def __init__(self) -> None:
        """initialize the stopping criterion
        """
        self.learning_function = U()

    def stopping_value(self, search_x: np.ndarray,
                       surrogate: Any,
                       iter: int,
                       **kwargs: Any) -> Any | float:
        """get the stopping value

        Parameters
        ----------
        search_x : np.ndarray
            Monte Carlo samples
        surrogate : Kriging
            surrogate model
        iter : int
            iteration number
        kwargs : Any
            other parameters

        Returns
        -------
        stopping value : float
            stopping value
        """
        if iter == 0:
            stopping_value = np.min(self.learning_function.eval(
                x=search_x,
                surrogate=surrogate))
            stopping_value = -stopping_value
        else:
            # use learning function value as stopping value
            stopping_value = float(-kwargs["lf_value"])

        return stopping_value


class BootstrapStoppingCriteria:

    def __init__(self, CI_factor: float = 0.05) -> None:
        """ initialize the stopping criterion

        Parameters
        ----------
        CI_factor : float, optional
            confidence interval factor, by default 0.05
        """
        self.CI_factor = CI_factor

    def stopping_value(self,
                       search_x: np.ndarray,
                       surrogate: Any,
                       iter: int,
                       **kwargs: Any) -> Any | float:
        """get the stopping value

        Parameters
        ----------
        search_x : np.ndarray
            Monte Carlo samples
        surrogate : Kriging
            surrogate model
        iter : int
            iteration number
        kwargs : Any
            other parameters

        Returns
        -------
        stopping value : float
            stopping value
        """
        # get prediction
        y, mse = surrogate.predict(search_x, return_std=True)

        # samples in the safe domain, but has risk of been failure
        safe_indices = np.logical_and(y >= 0, y - 1.96 * mse < 0).flatten()
        predict_safe = search_x[safe_indices]
        # get safe_y and corresponding mse
        safe_y = y[safe_indices]
        safe_mse = mse[safe_indices]

        # samples in the failure domain, but has risk of been safe
        fail_indices = np.logical_and(y < 0, y + 1.96 * mse > 0).flatten()
        predict_fail = search_x[fail_indices]
        # get fail_y and corresponding mse
        fail_y = y[fail_indices]
        fail_mse = mse[fail_indices]

        # Analysis of samples in the safe domain
        X_temp = -np.abs(safe_y / safe_mse)
        p_fail = norm.cdf(X_temp)
        mu_fail = predict_safe.shape[0] * np.mean(p_fail)

        # Estimate the bootstrap confidence interval for the safe domain
        if predict_safe.shape[0] > 1:
            boot_sample = np.array([np.mean(resample(p_fail))
                                   for _ in range(1000)])
            boot_sample = np.sort(boot_sample)
            k1 = int(1000 * self.CI_factor / 2)
            k2 = int(1000 * (1 - self.CI_factor / 2))
            S_con_interval = np.array(
                [predict_safe.shape[0] * boot_sample[k1],
                 predict_safe.shape[0] * boot_sample[k2]])
            CI_safe_max = np.max(S_con_interval)
        else:
            CI_safe_max = mu_fail

        # Analysis of samples in the failure domain
        X_temp = -np.abs(fail_y / fail_mse)
        p_safe = norm.cdf(X_temp)
        mu_safe = predict_fail.shape[0] * np.mean(p_safe)

        # Estimate the bootstrap confidence interval for the failure domain
        if predict_fail.shape[0] > 1:
            boot_fail = np.array([np.mean(resample(p_safe))
                                 for _ in range(1000)])
            boot_fail = np.sort(boot_fail)
            k1 = int(1000 * self.CI_factor / 2)
            k2 = int(1000 * (1 - self.CI_factor / 2))
            F_con_interval = np.array(
                [predict_fail.shape[0] * boot_fail[k1],
                 predict_fail.shape[0] * boot_fail[k2]])
            CI_fail_max = np.max(F_con_interval)
        else:
            CI_fail_max = mu_safe

        # find number of samples of y < 0
        N_f = y[y < 0].shape[0]

        error_1 = np.abs((N_f / (N_f - CI_safe_max)) - 1)
        error_2 = np.abs((N_f / (N_f + CI_fail_max)) - 1)

        error = max(error_1, error_2)

        return error
