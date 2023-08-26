from typing import Any

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import norm

# base class for multi-fidelity acquisition functions
# =========================================================================== #


class mfSingleObjAcf:
    """
    Base class for mf acquisition functions for single objective
    optimization.
    """
    @staticmethod
    def _initial_update() -> dict:
        # initialize the update dict
        update_x = {}
        update_x['hf'] = None
        update_x['lf'] = None
        return update_x

# augmented expected improvement
# =========================================================================== #


class augmentedEI(mfSingleObjAcf):
    """Augmented Expected Improvement acquisition function"""

    def __init__(
            self,
            optimizer: Any = None) -> None:
        """Initialize the multi-fidelity acquisition

        Parameters
        ----------
        optimizer : Any
            optimizer instance for getting update location
        """

        super().__init__()
        self.optimizer = optimizer

    def eval(
            self,
            x: np.ndarray,
            mf_surrogate: Any,
            cost_ratio: float,
            fidelity: str) -> np.ndarray:
        """Evaluates selected acquisition function at certain fidelity

        Parameters
        ----------
        x : np.ndarray
            point to evaluate
        mf_surrogate : any
            multi-fidelity surrogate instance
        cost_ratio : float
            ratio of high-fidelity cost to low-fidelity cost
        fidelity : str
            str indicating fidelity level

        Returns
        -------
        np.ndarray
            acqusition function values
        """
        # get the predictive mean and std of high-fidelity
        pre, std = mf_surrogate.predict(x, return_std=True)
        # get the predictive mean of low-fidelity
        fmin = mf_surrogate.predict(
            self._effective_best(mf_surrogate=mf_surrogate))
        # calculate the correlation value
        alpha1 = self.corr(x, mf_surrogate, fidelity)
        # calculate the fidelity ratio
        if fidelity == 'hf':
            alpha3 = 1.0
        elif fidelity == 'lf':
            alpha3 = cost_ratio
        else:
            ValueError('Unknown fidelity input.')
        # calculate the augmented expected improvement
        z = (fmin - pre) / std
        aei = (fmin - pre) * norm.cdf(z) + std * norm.pdf(z)
        aei[std < np.finfo(float).eps] = 0.
        # return the augmented expected improvement
        aei = aei * alpha1 * alpha3
        return (-aei).ravel()

    def query(self, mf_surrogate: Any, params: dict) -> dict:
        """Query the acqusition function

        Parameters
        ----------
        mf_surrogate : Any
            multi-fidelity surrogate instance
        params : dict
            parameters of Bayesian Optimization

        Returns
        -------
        dict
            contains two values where 'hf' is the update points
            for high-fidelity and 'lf' for low-fidelity
        """
        update_x = self._initial_update()
        if self.optimizer is None:
            # identify the best point for high-fidelity
            res_hf = differential_evolution(self.eval,
                                            bounds=params['design_space'],
                                            args=(mf_surrogate,
                                                  params['cr'], 'hf'),
                                            maxiter=500,
                                            popsize=40)
            # identify the best point for low-fidelity
            res_lf = differential_evolution(self.eval,
                                            bounds=params['design_space'],
                                            args=(mf_surrogate,
                                                  params['cr'], 'lf'),
                                            maxiter=500,
                                            popsize=40)
            # update the point for high-fidelity
            opt_hf = res_hf.fun
            x_hf = res_hf.x
            # update the point for low-fidelity
            opt_lf = res_lf.fun
            x_lf = res_lf.x
        else:
            pass
        # choose fidelity
        if opt_hf <= opt_lf:
            update_x['hf'] = np.atleast_2d(x_hf)
        else:
            update_x['lf'] = np.atleast_2d(x_lf)
        return update_x

    @staticmethod
    def _effective_best(mf_surrogate: Any, c: float = 1.) -> np.ndarray:
        """Return the effective best solution

        Parameters
        ----------
        mf_surrogate : Any
            multi-fidelity surrogate instance
        c : float, optional
            degree of risk aversion, by default 1

        Returns
        -------
        np.ndarray
            effective best solution
        """
        x = np.concatenate((mf_surrogate._get_sample_hf,
                           mf_surrogate._get_sample_lf), axis=0)
        pre, std = mf_surrogate.predict(x, return_std=True)
        u = pre + c * std
        return x[np.argmin(u.squeeze()), :]

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
            pre, std = mf_surrogate.predict(x, return_std=True)
            pre_lf, std_lf = mf_surrogate.predict_lf(x, return_std=True)
            return std_lf / (np.abs(pre - pre_lf).reshape(-1, 1) + std)

# probability of improvement
# =========================================================================== #


class vfei(mfSingleObjAcf):
    def __init__(
            self,
            optimizer: Any = None) -> None:
        """Initialize the multi-fidelity acqusition

        Parameters
        ----------
        optimizer : Any
            optimizer instance
        """
        self.optimizer = optimizer

    def eval(
            self,
            x: np.ndarray,
            fmin: float,
            mf_surrogate: Any,
            fidelity: str) -> np.ndarray:
        """
        Evaluates selected acqusition function at certain fidelity

        Parameters:
        -----------------
        x: np.ndarray
            point to evaluate
        fmin: float
            best observed function evaluation
        mf_surrogate: any
            multi-fidelity surrogate instance
        fidelity: str
            str indicating fidelity level

        Returns
        -----------------
        np.ndarray
            Acqusition function value w.r.t corresponding fidelity level.
        """
        # get prediction for inputs
        pre, std = mf_surrogate.predict(x, return_std=True)
        if fidelity == 'hf':
            # get predicted standard deviation for high-fidelity
            s = std
        elif fidelity == 'lf':
            # get predicted standard deviation for low-fidelity
            _, std_lf = mf_surrogate.predict_lf(x, return_std=True)
            s = mf_surrogate.mu * std_lf
        else:
            ValueError('Unknown fidelity input.')
        # expected improvement
        z = (fmin - pre) / std
        vfei = (fmin - pre) * norm.cdf(z) + std * norm.pdf(z)
        vfei[s < np.finfo(float).eps] = 0.
        return (- vfei).ravel()

    def query(self, mf_surrogate: Any, params: dict) -> dict:
        """Query the acqusition function

        Parameters
        ----------
        mf_surrogate : Any
            multi-fidelity surrogate instance
        params : dict
            parameters of Bayesian Optimization

        Returns
        -------
        dict
            contains two values where 'hf' is the update points
            for high-fidelity and 'lf' for low-fidelity
        """
        update_x = self._initial_update()
        if self.optimizer is None:
            # identify the best point for high-fidelity
            res_hf = differential_evolution(self.eval,
                                            bounds=params['design_space'],
                                            args=(params['fmin'],
                                                  mf_surrogate, 'hf'),
                                            maxiter=500,
                                            popsize=40)
            # identify the best point for low-fidelity
            res_lf = differential_evolution(self.eval,
                                            bounds=params['design_space'],
                                            args=(params['fmin'],
                                                  mf_surrogate, 'lf'),
                                            maxiter=500,
                                            popsize=40)
            # update the point for high-fidelity
            opt_hf = res_hf.fun
            x_hf = res_hf.x
            # update the point for low-fidelity
            opt_lf = res_lf.fun
            x_lf = res_lf.x
        else:
            pass
        # choose fidelity
        if opt_hf <= opt_lf:
            update_x['hf'] = np.atleast_2d(x_hf)
        else:
            update_x['lf'] = np.atleast_2d(x_lf)
        return update_x

# multi fidelity lower confidence bound
# =========================================================================== #


class vflcb(mfSingleObjAcf):
    """Variable-fidelity Lower Confidence Bound acqusition function"""

    def __init__(
            self,
            optimizer: Any = None,
            kappa: list = [1., 1.96]) -> None:
        """Initialize the vflcb acqusition

        Parameters
        ----------
        optimizer : any, optional
            optimizer instance, by default 'L-BFGS-B'
        kappa : list, optional
            balance factors for exploitation and exploration respectively
            , by default [1., 1.96]
        """
        super().__init__()
        self.optimizer = optimizer
        self.kappa = kappa

    def eval(
            self,
            x: np.ndarray,
            mf_surrogate: Any,
            cost_ratio: float,
            fidelity: str) -> np.ndarray:
        """Evaluate vflcb function values at certain fidelity

        Parameters
        ----------
        x : np.ndarray
            point to evaluate
        mf_surrogate : Any
            multi-fidelity surrogate model instance
        cost_ratio : float
            ratio of high-fidelity cost to low-fidelity cost
        fidelity : str
            str indicating fidelity level

        Returns
        -------
        np.ndarray
            acqusition function values
        """
        cr = cost_ratio
        # get prediction for inputs
        mean_hf, std_hf = mf_surrogate.predict(x, return_std=True)
        # get predicted standard deviation for low-fidelity
        _, std_lf = mf_surrogate.predict_lf(x, return_std=True)
        if fidelity == 'hf':
            std = std_hf
        elif fidelity == 'lf':
            std = std_lf * cr
        else:
            ValueError('Unknown fidelity input.')
        # vflcb
        vflcb = self.kappa[0] * mean_hf - self.kappa[1] * std
        return vflcb.ravel()

    def query(self, mf_surrogate: Any, params: dict) -> dict:
        """Query the acqusition function

        Parameters
        ----------
        mf_surrogate : Any
            multi-fidelity surrogate instance
        params : dict
            parameters of Bayesian Optimization

        Returns
        -------
        dict
            contains two values where 'hf' is the update points
            for high-fidelity and 'lf' for low-fidelity
        """
        update_x = self._initial_update()
        if self.optimizer is None:
            # identify the best point for high-fidelity
            res_hf = differential_evolution(self.eval,
                                            bounds=params['design_space'],
                                            args=(mf_surrogate,
                                                  params['cr'], 'hf'),
                                            maxiter=500,
                                            popsize=40)
            # identify the best point for low-fidelity
            res_lf = differential_evolution(self.eval,
                                            bounds=params['design_space'],
                                            args=(mf_surrogate,
                                                  params['cr'], 'lf'),
                                            maxiter=500,
                                            popsize=40)
            # obtain optimal value for high fidelity and corresponding point
            opt_hf = res_hf.fun
            x_hf = res_hf.x
            # obtain optimal value for low fidelity and corresponding point
            opt_lf = res_lf.fun
            x_lf = res_lf.x
        else:
            pass
        if opt_hf <= opt_lf:
            # if high fidelity is better, update high fidelity
            update_x['hf'] = np.atleast_2d(x_hf)
        else:
            # if low fidelity is better, update low fidelity
            update_x['lf'] = np.atleast_2d(x_lf)
        return update_x


class extendedPI(mfSingleObjAcf):
    """Extended Probability Improvement acqusition function

    Reference
    ---------
    [1] Ruan, X., Jiang, P., Zhou, Q., Hu, J., & Shu, L. (2020).
    Variable-fidelity probability of improvement method for
    efficient global optimization of expensive black-box problems.
    Structural and Multidisciplinary Optimization, 62(6), 3021-3052.

    """

    def __init__(
            self,
            optimizer: Any = None) -> None:
        """Initialize the multi-fidelity acqusition

        Parameters
        ----------
        optimizer : any
            optimizer instance
        """
        super().__init__()
        self.optimizer = optimizer

    def eval(self,
             x: np.ndarray,
             mf_surrogate: Any,
             fmin: np.ndarray,
             cost_ratio: dict,
             fidelity: str) -> np.ndarray:
        """Evaluates selected acqusition function at certain fidelity

        Parameters
        ----------
        x : np.ndarray
            point to evaluate
        mf_surrogate : Any
            multi-fidelity surrogate instance
        fmin : np.ndarray
            best prediction of multi-fidelity model
        cost_ratio : float
            ratio of high-fidelity cost to low-fidelity cost
        fidelity : str
            str indicating fidelity level

        Returns
        -------
        np.ndarray
            acqusition function values
        """
        # predict mean and standard deviation for inputs
        pre, std = mf_surrogate.predict(x, return_std=True)
        # expected improvement
        z = (fmin - pre) / std
        # probability of improvement
        pi = norm.cdf(z)
        pi[std < np.finfo(float).eps] = 0
        corr = self.corr(x, mf_surrogate, fidelity)
        if fidelity == 'hf':
            cr = 1
            eta = np.prod(mf_surrogate._eval_corr(
                x, mf_surrogate._get_sample_hf), axis=1)
        elif fidelity == 'lf':
            cr = cost_ratio
            eta = np.prod(mf_surrogate._eval_corr(
                x, mf_surrogate._get_sample_lf, fidelity='lf'), axis=1)
        return -pi*corr*cr*eta

    def query(self, mf_surrogate: Any, params: dict) -> dict:
        """Query the acqusition function

        Parameters
        ----------
        mf_surrogate : any
            multi-fidelity surrogate instance
        params : dict
            parameters of Bayesian Optimization

        Returns
        -------
        dict
            contains two values where 'hf' is the update points
            for high-fidelity and 'lf' for low-fidelity
        """
        # get the best point from the multi-fidelity model
        premin = differential_evolution(mf_surrogate.predict,
                                        bounds=params['design_space'],
                                        maxiter=250, popsize=40)
        update_x = self._initial_update()
        if self.optimizer is None:
            # get best point for high-fidelity
            res_hf = differential_evolution(self.eval,
                                            bounds=params['design_space'],
                                            args=(mf_surrogate, premin.fun,
                                                  params['cr'], 'hf'),
                                            maxiter=500,
                                            popsize=40)
            # get best point for low-fidelity
            res_lf = differential_evolution(self.eval,
                                            bounds=params['design_space'],
                                            args=(mf_surrogate, premin.fun,
                                                  params['cr'], 'lf'),
                                            maxiter=500,
                                            popsize=40)
            # obtain optimal value for high fidelity and corresponding point
            opt_hf = res_hf.fun
            x_hf = res_hf.x
            # obtain optimal value for low fidelity and corresponding point
            opt_lf = res_lf.fun
            x_lf = res_lf.x
        else:
            pass
        # choose fidelity with better optimal value
        if opt_hf <= opt_lf:
            # choose high high fidelity
            update_x['hf'] = np.atleast_2d(x_hf)
        else:
            # choose low fidelity
            update_x['lf'] = np.atleast_2d(x_lf)
        return update_x

    @staticmethod
    def corr(x: np.ndarray, mf_surrogate: Any, fidelity: str) -> np.ndarray:
        """Evaluate correlation between different fidelity

        Parameters
        ----------
        x : np.ndarray
            point to evaluate
        mf_surrogate : any
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
            pre, std = mf_surrogate.predict(x, return_std=True)
            pre_lf, std_lf = mf_surrogate.predict_lf(x, return_std=True)
            return std_lf / (np.abs(pre - pre_lf).reshape(-1, 1) + std)
