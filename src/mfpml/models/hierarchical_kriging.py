
from typing import Any, Tuple

import numpy as np
from numpy.linalg import cholesky, solve
from scipy.optimize import minimize

from .gaussian_process import GaussianProcessRegression as GP
from .kernels import RBF
from .mf_gaussian_process import _mfGaussianProcess


class HierarchicalKriging(_mfGaussianProcess):

    def __init__(
        self,
        design_space: np.ndarray,
        optimizer: Any = None,
        optimizer_restart: int = 5,
        kernel_bound: list = [-4.0, 2.0],
        noise_prior: float = None,
    ) -> None:

        # initialize the base class
        super().__init__(design_space)

        # initialize the parameters
        self.optimizer = optimizer
        self.optimizer_restart = optimizer_restart
        self.num_dim = design_space.shape[0]
        self.kernel = RBF(theta=np.zeros(self.num_dim), bounds=kernel_bound)
        self.noise = noise_prior
        # define the low-fidelity model
        self.lfGP = GP(design_space=design_space,
                       optimizer=optimizer,
                       optimizer_restart=optimizer_restart,
                       noise_prior=noise_prior)

    def _train_hf(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Train the high-fidelity model

        Parameters
        ----------
        X : np.ndarray
            array of high-fidelity samples
        Y : np.ndarray
            array of high-fidelity responses
        """
        # get samples
        self.sample_xh = X
        self.sample_yh = Y
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
        self, X: np.ndarray, return_std: bool = False
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """Predict high-fidelity responses

        Parameters
        ----------
        X : np.ndarray
            array of high-fidelity to be predicted
        return_std : bool, optional
            whether to return std values, by default False

        Returns
        -------
        np.ndarray
            prediction of high-fidelity
        """
        # original prediction of lf
        pre_lf = self.predict_lf(X)
        # scale it to hf
        pre_lf = (pre_lf - self.yh_mean) / self.yh_std

        # normalize the input
        XHnew = np.atleast_2d(self.normalize_input(X))
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
        if self.noise is None:
            # noise value needs to be optimized
            lower_bound_theta = self.kernel._get_low_bound
            upper_bound_theta = self.kernel._get_high_bound
            # set up the bounds for noise sigma
            lower_bound_sigma = 1e-5
            upper_bound_sigma = 10.0
            # set up the bounds for the hyper-parameters
            lower_bound = np.hstack((lower_bound_theta, lower_bound_sigma))
            upper_bound = np.hstack((upper_bound_theta, upper_bound_sigma))
            # bounds for the hyper-parameters
            hyper_bounds = np.vstack((lower_bound, upper_bound)).T
            # number of hyper-parameters
            num_hyper = self.kernel._get_num_para + 1
        else:
            lower_bound = self.kernel._get_low_bound
            upper_bound = self.kernel._get_high_bound
            # bounds for the hyper-parameters
            hyper_bounds = np.vstack((lower_bound, upper_bound)).T
            # number of hyper-parameters
            num_hyper = self.kernel._get_num_para

        if self.optimizer is None:
            n_trials = self.optimizer_restart + 1
            opt_fs = float("inf")
            for _ in range(n_trials):
                x0 = np.random.uniform(
                    lower_bound,
                    upper_bound,
                    num_hyper
                )
                optRes = minimize(
                    self._logLikelihood,
                    x0=x0,
                    method="L-BFGS-B",
                    bounds=hyper_bounds,
                )
                if optRes.fun < opt_fs:
                    opt_param = optRes.x
                    opt_fs = optRes.fun
        else:
            optRes, _, _ = self.optimizer.run_optimizer(
                self._logLikelihood,
                num_dim=num_hyper,
                design_space=hyper_bounds,
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
        num_params = params.shape[1]
        out = np.zeros(params.shape[0])
        for i in range(params.shape[0]):
            # for optimization every row is a parameter set
            if self.noise is None:
                param = params[i, 0: num_params - 1]
                noise_sigma = params[i, -1]
            else:
                param = params[i, :]
                noise_sigma = self.noise / self.yh_std

            K = self.kernel(self.sample_xh_scaled,
                            self.sample_xh_scaled, param) + noise_sigma**2 * np.eye(self._num_xh)
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
        if self.noise is None:
            self.noise = self.opt_param[-1]*self.yh_std
            self.kernel.set_params(self.opt_param[:-1])
        else:
            self.kernel.set_params(self.opt_param)
        self.K = self.kernel.get_kernel_matrix(
            self.sample_xh_scaled, self.sample_xh_scaled) + \
            (self.noise/self.yh_std)**2 * np.eye(self._num_xh)
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
