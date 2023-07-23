from typing import Any

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import norm


class mfAcqusitionFunction:
    """
    Base class for multi-fidelity acqusition functions.

    """
    pass


class mfSingleObjAcf(mfAcqusitionFunction):
    """
    Base class for mf acqusition functions for Single Objective Opti.
    """
    @staticmethod
    def _initial_update():
        update_x = {}
        update_x['hf'] = None
        update_x['lf'] = None
        return update_x

    pass


class augmentedEI(mfSingleObjAcf):
    """Augmented Expected Improvement acqusition function"""

    def __init__(
            self,
            optimizer: Any = None,
            constraint: bool = False) -> None:
        """Initialize the multi-fidelity acqusition

        Parameters
        ----------
        optimizer : Any
            optimizer instance
        constraint : bool, optional
            whether to use for constrained optimization
        """
        super().__init__()
        self.optimizer = optimizer
        self.constraint = constraint

    def eval(
            self,
            x: np.ndarray,
            mf_surrogate: Any,
            cost_ratio: float,
            fidelity: str) -> np.ndarray:
        """Evaluates selected acqusition function at certain fidelity

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
        pre, std = mf_surrogate.predict(x, return_std=True)
        fmin = mf_surrogate.predict(
            self._effective_best(mf_surrogate=mf_surrogate))
        alpha1 = self.corr(x, mf_surrogate, fidelity)
        if fidelity == 'hf':
            alpha3 = 1
        elif fidelity == 'lf':
            alpha3 = cost_ratio
        else:
            ValueError('Unknown fidelity input.')
        z = (fmin - pre) / std
        aei = (fmin - pre) * norm.cdf(z) + std * norm.pdf(z)
        aei[std < np.finfo(float).eps] = 0.
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
            res_hf = differential_evolution(self.eval,
                                            bounds=params['design_space'],
                                            args=(mf_surrogate,
                                                  params['cr'], 'hf'),
                                            maxiter=500,
                                            popsize=40)
            res_lf = differential_evolution(self.eval,
                                            bounds=params['design_space'],
                                            args=(mf_surrogate,
                                                  params['cr'], 'lf'),
                                            maxiter=500,
                                            popsize=40)
            opt_hf = res_hf.fun
            x_hf = res_hf.x
            opt_lf = res_lf.fun
            x_lf = res_lf.x
        else:
            pass
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


class vfei(mfSingleObjAcf):
    def __init__(
            self,
            optimizer: Any = None,
            constraint: bool = False) -> None:
        """Initialize the multi-fidelity acqusition

        Parameters
        ----------
        optimizer : Any
            optimizer instance
        constraint : bool, optional
            whether to use for constrained optimization
        """
        self.constraint = constraint
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
        pre, std = mf_surrogate.predict(x, return_std=True)
        if fidelity == 'hf':
            s = std
        elif fidelity == 'lf':
            _, std_lf = mf_surrogate.predict_lf(x, return_std=True)
            s = mf_surrogate.mu * std_lf
        else:
            ValueError('Unknown fidelity input.')

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
            res_hf = differential_evolution(self.eval,
                                            bounds=params['design_space'],
                                            args=(params['fmin'],
                                                  mf_surrogate, 'hf'),
                                            maxiter=500,
                                            popsize=40)
            res_lf = differential_evolution(self.eval,
                                            bounds=params['design_space'],
                                            args=(params['fmin'],
                                                  mf_surrogate, 'lf'),
                                            maxiter=500,
                                            popsize=40)
            opt_hf = res_hf.fun
            x_hf = res_hf.x
            opt_lf = res_lf.fun
            x_lf = res_lf.x
        else:
            pass
        if opt_hf <= opt_lf:
            update_x['hf'] = np.atleast_2d(x_hf)
        else:
            update_x['lf'] = np.atleast_2d(x_lf)
        return update_x


class vflcb(mfSingleObjAcf):
    """Variable-fidelity Lower Confidence Bound acqusition function"""

    def __init__(
            self,
            optimizer: Any = None,
            kappa: list = [1., 1.96],
            constraint: bool = False) -> None:
        """Initialize the vflcb acqusition

        Parameters
        ----------
        optimizer : any, optional
            optimizer instance, by default 'L-BFGS-B'
        kappa : list, optional
            balance factors for exploitation and exploration respectively
            , by default [1., 1.96]
        constraint : bool, optional
            use for constrained problem or not, by default False
        """
        super().__init__()
        self.optimizer = optimizer
        self.kappa = kappa
        self.constraint = constraint

    def eval(
            self,
            x: np.ndarray,
            mf_surrogate: any,
            cost_ratio: float,
            fidelity: str) -> np.ndarray:
        """Evaluate vflcb function values at certain fidelity

        Parameters
        ----------
        x : np.ndarray
            point to evaluate
        mf_surrogate : any
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
        mean_hf, std_hf = mf_surrogate.predict(x, return_std=True)
        _, std_lf = mf_surrogate.predict_lf(x, return_std=True)
        if fidelity == 'hf':
            std = std_hf
        elif fidelity == 'lf':
            std = std_lf * cr
        else:
            ValueError('Unknown fidelity input.')
        vflcb = self.kappa[0] * mean_hf - self.kappa[1] * std
        return vflcb.ravel()

    def query(self, mf_surrogate: any, params: dict) -> dict:
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
        update_x = self._initial_update()
        if self.optimizer is None:
            res_hf = differential_evolution(self.eval,
                                            bounds=params['design_space'],
                                            args=(mf_surrogate,
                                                  params['cr'], 'hf'),
                                            maxiter=500,
                                            popsize=40)
            res_lf = differential_evolution(self.eval,
                                            bounds=params['design_space'],
                                            args=(mf_surrogate,
                                                  params['cr'], 'lf'),
                                            maxiter=500,
                                            popsize=40)
            opt_hf = res_hf.fun
            x_hf = res_hf.x
            opt_lf = res_lf.fun
            x_lf = res_lf.x
        else:
            pass
        if opt_hf <= opt_lf:
            update_x['hf'] = np.atleast_2d(x_hf)
        else:
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
            optimizer: Any = None,
            constraint: bool = False) -> None:
        """Initialize the multi-fidelity acqusition

        Parameters
        ----------
        optimizer : any
            optimizer instance
        constraint : bool, optional
            whether to use for constrained optimization
        """
        super().__init__()
        self.optimizer = optimizer
        self.constraint = constraint

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
        pre, std = mf_surrogate.predict(x, return_std=True)
        z = (fmin - pre) / std
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

    def query(self, mf_surrogate: any, params: dict) -> dict:
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
        premin = differential_evolution(mf_surrogate.predict,
                                        bounds=params['design_space'],
                                        maxiter=250, popsize=40)
        update_x = self._initial_update()
        if self.optimizer is None:
            res_hf = differential_evolution(self.eval,
                                            bounds=params['design_space'],
                                            args=(mf_surrogate, premin.fun,
                                                  params['cr'], 'hf'),
                                            maxiter=500,
                                            popsize=40)
            res_lf = differential_evolution(self.eval,
                                            bounds=params['design_space'],
                                            args=(mf_surrogate, premin.fun,
                                                  params['cr'], 'lf'),
                                            maxiter=500,
                                            popsize=40)
            opt_hf = res_hf.fun
            x_hf = res_hf.x
            opt_lf = res_lf.fun
            x_lf = res_lf.x
        else:
            pass
        if opt_hf <= opt_lf:
            update_x['hf'] = np.atleast_2d(x_hf)
        else:
            update_x['lf'] = np.atleast_2d(x_lf)
        return update_x

    @staticmethod
    def corr(x: np.ndarray, mf_surrogate: any, fidelity: str) -> np.ndarray:
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
