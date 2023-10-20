import time
from typing import Any

import numpy as np
from scipy.linalg import cholesky, solve
from scipy.optimize import minimize

from .gpr_base import mf_model
from .kernels import RBF
from .kriging import Kriging


class HierarchicalKriging(mf_model):
    def __init__(
        self,
        design_space: np.ndarray,
        optimizer: Any = None,
        kernel_bound: list = [-4.0, 2.0],
    ) -> None:
        """Initialize hierarchical Kriging model

        Parameters
        ----------
        design_space: np.ndarray
            array of shape=((num_dim,2)) where each row describes
            the bound for the variable on specfic dimension.
        optimizer: Any, optional
            instance of the optimizer used to optimize the hyperparameters
            with the use style optimizer.run_optimizer(objective function,
            number of dimension, design space of variables), if not assigned,
            the 'L-BFGS-B' method in scipy is used.
        kernel_bound: list, optional
            log bound of the kernel for hierarchical Kriging model, by
            default [-2, 3].
        """
        self.bounds = design_space
        self.optimizer = optimizer
        self.num_dim = design_space.shape[0]
        self.kernel = RBF(theta=np.zeros(self.num_dim), bounds=kernel_bound)

        self.lf_model = Kriging(design_space=design_space, optimizer=optimizer)

    def _train_hf(self, sample_xh: np.ndarray, sample_yh: np.ndarray) -> None:
        """Train the high-fidelity model

        Parameters
        ----------
        sample_xh : np.ndarray
            array of high-fidelity samples
        sample_yh : np.ndarray
            array of high-fidelity responses
        """
        self.sample_xh = sample_xh
        self.sample_xh_scaled = self.normalize_input(self.sample_xh)
        self.sample_yh = sample_yh.reshape(-1, 1)
        # prediction of low-fidelity at high-fidelity locations
        self.F = self.predict_lf(self.sample_xh)
        # optimize the hyper parameters
        start_time = time.time()
        self._optHyp()
        end_time = time.time()
        print("Optimizing time: ", end_time - start_time)
        self.kernel.set_params(self.opt_param)
        self._update_parameters()

    def predict(
        self, test_x: np.ndarray, return_std: bool = False
    ) -> np.ndarray:
        """Predict high-fidelity responses

        Parameters
        ----------
        test_x : np.ndarray
            array of high-fidelity to be predicted
        return_std : bool, optional
            whether to return std values, by default False

        Returns
        -------
        np.ndarray
            prediction of high-fidelity
        """
        pre_lf = self.predict_lf(test_x)
        XHnew = np.atleast_2d(self.normalize_input(test_x))
        knew = self.kernel.get_kernel_matrix(XHnew, self.sample_xh_scaled)
        fmean = self.mu * pre_lf + np.dot(knew, self.gamma)
        if not return_std:
            return fmean.reshape(-1, 1)
        else:
            delta = solve(self.L.T, solve(self.L, knew.T))
            mse = self.sigma2 * (
                1
                - np.diag(knew.dot(delta))
                + np.diag(
                    np.dot(
                        (knew.dot(self.beta) - pre_lf),
                        (knew.dot(self.beta) - pre_lf).T,
                    )
                )
                / self.F.T.dot(self.beta)
            )
            return fmean.reshape(-1, 1), np.sqrt(np.maximum(mse, 0)).reshape(
                -1, 1
            )

    def _optHyp(self, grads=None):
        """Optimize the hyperparameters

        Parameters
        ----------
        grads : bool, optional
            whether to use gradients, by default None
        """
        if self.optimizer is None:
            n_trials = 5
            opt_fs = float("inf")
            for _ in range(n_trials):
                x0 = np.random.uniform(
                    self.kernel._get_low_bound,
                    self.kernel._get_high_bound,
                    self.kernel._get_num_para,
                )
                optRes = minimize(
                    self._logLikelihood,
                    x0=x0,
                    method="L-BFGS-B",
                    bounds=self.kernel._get_bounds_list,
                )
                if optRes.fun < opt_fs:
                    opt_param = optRes.x
                    opt_fs = optRes.fun
        else:
            optRes, _, _ = self.optimizer.run_optimizer(
                self._logLikelihood,
                num_dim=self.kernel._get_num_para,
                design_space=self.kernel._get_bounds,
            )
            opt_param = optRes["best_x"]
        self.opt_param = opt_param

    def _logLikelihood(self, params):
        """Compute the concentrated ln-likelihood

        Parameters
        ----------
        params : np.ndarray
            parameters of the kernel

        Returns
        -------
        np.ndarray
            log likelihood
        """
        params = np.atleast_2d(params)
        out = np.zeros(params.shape[0])
        for i in range(params.shape[0]):
            param = params[i, :]
            K = self.kernel(self.sample_xh_scaled,
                            self.sample_xh_scaled, param)
            L = cholesky(K, lower=True)
            alpha = solve(L.T, solve(L, self.sample_yh))
            beta = solve(L.T, solve(L, self.F))
            mu = (np.dot(self.F.T, alpha) / np.dot(self.F.T, beta)).squeeze()
            gamma = solve(L.T, solve(L, (self.sample_yh - mu * self.F)))
            sigma2 = (
                np.dot((self.sample_yh - mu * self.F).T, gamma).squeeze()
                / self._num_xh
            ).squeeze()
            logp = -self._num_xh * np.log(sigma2) - 2 * np.sum(
                np.log(np.diag(L))
            )
            out[i] = logp.squeeze()
        return -out

    def _update_parameters(self) -> None:
        """Update parameters of the model"""
        self.K = self.kernel.get_kernel_matrix(
            self.sample_xh_scaled, self.sample_xh_scaled)
        self.L = cholesky(self.K, lower=True)
        self.alpha = solve(self.L.T, solve(self.L, self.sample_yh))
        self.beta = solve(self.L.T, solve(self.L, self.F))
        self.mu = (
            np.dot(self.F.T, self.alpha) / np.dot(self.F.T, self.beta)
        ).item()
        self.gamma = solve(
            self.L.T, solve(self.L, (self.sample_yh - self.mu * self.F))
        )
        self.sigma2 = (
            np.dot((self.sample_yh - self.mu * self.F).T, self.gamma).squeeze()
            / self._num_xh
        ).item()
        self.logp = (
            -self._num_xh * np.log(self.sigma2)
            - np.sum(np.log(np.diag(self.L)))
        ).item()
