
from typing import Any, Tuple

import numpy as np
from numpy.linalg import cholesky, solve
from scipy.optimize import minimize

from .gpr_base import MultiFidelityGP
from .kernels import RBF
from .kriging import Kriging


class HierarchicalKriging(MultiFidelityGP):
    def __init__(
        self,
        design_space: np.ndarray,
        optimizer: Any = None,
        optimizer_restart: int = 0,
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
        self.optimizer_restart = optimizer_restart
        self.num_dim = design_space.shape[0]
        self.kernel = RBF(theta=np.zeros(self.num_dim), bounds=kernel_bound)

        self.lf_model = Kriging(design_space=design_space,
                                optimizer=optimizer,
                                optimizer_restart=optimizer_restart)

    def _train_hf(self, sample_xh: np.ndarray, sample_yh: np.ndarray) -> None:
        """Train the high-fidelity model

        Parameters
        ----------
        sample_xh : np.ndarray
            array of high-fidelity samples
        sample_yh : np.ndarray
            array of high-fidelity responses
        """
        # get samples
        self.sample_xh = sample_xh
        self.sample_yh = sample_yh
        # normalization
        self.sample_xh_scaled = self.normalize_input(self.sample_xh)
        self.sample_yh_scaled = self.normalize_hf_output(self.sample_yh)
        # prediction of low-fidelity at high-fidelity locations
        F = self.predict_lf(self.sample_xh)
        self.F = (F-self.yh_mean)/self.yh_std
        # optimize the hyper parameters
        self._optHyp()
        self.kernel.set_params(self.opt_param)
        self._update_parameters()

    def predict(
        self, x_predict: np.ndarray, return_std: bool = False
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """Predict high-fidelity responses

        Parameters
        ----------
        x_predict : np.ndarray
            array of high-fidelity to be predicted
        return_std : bool, optional
            whether to return std values, by default False

        Returns
        -------
        np.ndarray
            prediction of high-fidelity
        """
        # original prediction of lf
        pre_lf = self.predict_lf(x_predict)
        # scale it to hf
        pre_lf = (pre_lf - self.yh_mean) / self.yh_std

        # normalize the input
        XHnew = np.atleast_2d(self.normalize_input(x_predict))
        knew = self.kernel.get_kernel_matrix(XHnew, self.sample_xh_scaled)

        # get the prediction
        fmean = self.mu * pre_lf + np.dot(knew, self.gamma)
        # scale it back to original scale
        fmean = fmean * self.yh_std + self.yh_mean
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

            # scale it back to original scale
            mse = np.sqrt(np.maximum(mse, 0))*self.yh_std
            return fmean.reshape(-1, 1), mse.reshape(-1, 1)

    def _optHyp(self):
        """Optimize the hyperparameters"""
        if self.optimizer is None:
            n_trials = self.optimizer_restart + 1
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
            L = cholesky(K)
            alpha = solve(L.T, solve(L, self.sample_yh_scaled))
            beta = solve(L.T, solve(L, self.F))
            mu = (np.dot(self.F.T, alpha) / np.dot(self.F.T, beta)).squeeze()
            gamma = solve(L.T, solve(L, (self.sample_yh_scaled - mu * self.F)))
            sigma2 = (
                np.dot((self.sample_yh_scaled -
                        mu * self.F).T, gamma).squeeze()
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
        self.L = cholesky(self.K)
        self.alpha = solve(self.L.T, solve(self.L, self.sample_yh_scaled))
        self.beta = solve(self.L.T, solve(self.L, self.F))
        self.mu = (
            np.dot(self.F.T, self.alpha) / np.dot(self.F.T, self.beta)
        ).item()
        self.gamma = solve(
            self.L.T, solve(self.L, (self.sample_yh_scaled - self.mu * self.F))
        )
        self.sigma2 = (
            np.dot((self.sample_yh_scaled - self.mu * self.F).T,
                   self.gamma).squeeze()
            / self._num_xh
        ).item()
        self.logp = (
            -self._num_xh * np.log(self.sigma2)
            - np.sum(np.log(np.diag(self.L)))
        ).item()
