from typing import Any, Tuple

import numpy as np
from numpy.linalg import cholesky, solve
from scipy.optimize import minimize

from .gpr_base import MultiFidelityGP
from .kernels import RBF
from .kriging import Kriging


class CoKriging(MultiFidelityGP):
    def __init__(
        self,
        design_space: np.ndarray,
        optimizer: Any = None,
        optimizer_restart: int = 0,
        kernel_bound: list = [-4.0, 3.0],
        rho_bound: list = [1e-2, 1e2],
    ) -> None:
        """co-kriging model for handling multi-fidelity data

        Parameters
        ----------
        design_space : np.ndarray
            array of shape=((num_dim,2)) where each row describes
            the bound for the variable on specfic dimension
        optimizer : Any, optional
            instance of the optimizer used to optimize the hyperparameters
            with the use style optimizer.run_optimizer(objective function,
            number of dimension, design space of variables), if not assigned,
            the 'L-BFGS-B' method in scipy is used
        kernel_bound : list, optional
            log bound of the kernel for difference Gaussian process model,
            by default [-4, 3]
        rho_bound : list, optional
            bound for scale factor, by default [1e-2, 1e2]
        """
        super().__init__()
        # get information of the modeling problem
        self.bounds = design_space
        self.optimizer = optimizer
        self.optimizer_restart = optimizer_restart
        self.num_dim = design_space.shape[0]
        self.rho_bound = rho_bound
        # initialize kernel and low-fidelity model
        self.kernel = RBF(theta=np.zeros(self.num_dim), bounds=kernel_bound)
        # lf model is a Kriging model with RBF kernel
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
        self.sample_xh = sample_xh
        # normalize the input for high-fidelity
        self.sample_xh_scaled = self.normalize_input(self.sample_xh.copy())
        # get the high-fidelity responses
        self.sample_yh = sample_yh
        self.sample_yh_scaled = self.normalize_hf_output(sample_yh.copy())
        self.sample_yl_scaled = (self.sample_yl - self.yh_mean) / self.yh_std
        # prediction of low-fidelity at high-fidelity locations
        pred_ylh = self.predict_lf(self.sample_xh, return_std=False)
        # scale it to the same scale as high-fidelity
        self.pred_ylh = (pred_ylh - self.yh_mean) / self.yh_std
        # optimize the hyper parameters
        self._optHyp()
        # update rho value
        self.rho = self.opt_param[0]
        # calculate the covariance matrix
        self.kernel.set_params(self.opt_param[1:])

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
        # transfer to 2d array
        Xnew = np.atleast_2d(self.normalize_input(x_predict))
        #
        oneC = np.ones((self.C.shape[0], 1))
        # calculate the covariance matrix
        c = np.concatenate(
            (
                self.rho
                * self.lf_model.sigma2
                * self.lf_model.kernel.get_kernel_matrix(
                    self.sample_xl_scaled, Xnew),
                self.rho**2
                * self.lf_model.sigma2 *
                self.lf_model.kernel.get_kernel_matrix(
                    self.sample_xh_scaled, Xnew)
                + self.sigma2 *
                self.kernel.get_kernel_matrix(self.sample_xh_scaled, Xnew),
            ),
            axis=0,
        )
        # get the predicted mean and std
        fmean = self.mu + c.T.dot(
            solve(self.LC.T, solve(self.LC, (self.y - self.mu)))
        )
        # scale to original scale
        fmean = fmean * self.yh_std + self.yh_mean
        if not return_std:
            return fmean.reshape(-1, 1)
        else:
            s2 = (
                self.rho**2 * self.lf_model.sigma2
                + self.sigma2
                - c.T.dot(solve(self.LC.T, solve(self.LC, c)))
                + (1-oneC.T.dot(solve(self.LC.T, solve(self.LC, c)))) /
                oneC.T.dot(solve(self.LC.T, solve(self.LC, oneC)))
            )
            std = np.sqrt(np.maximum(np.diag(s2), 0))*self.yh_std
            return fmean.reshape(-1, 1), std.reshape(-1, 1)

    def _optHyp(self) -> None:
        """Optimize the hyperparameters"""
        bounds = np.concatenate(
            (np.array([self.rho_bound]), self.kernel._get_bounds), axis=0
        )
        if self.optimizer is None:
            # define the number of trials of L-BFGS-B optimizations
            n_trials = self.optimizer_restart + 1
            opt_fs = float("inf")
            for _ in range(n_trials):
                x0 = np.random.uniform(
                    bounds[:, 0], bounds[:, 1], bounds.shape[0]
                )
                optRes = minimize(
                    self._logLikelihood,
                    x0=x0,
                    method="L-BFGS-B",
                    bounds=bounds,
                )
                if optRes.fun < opt_fs:
                    opt_param = optRes.x
                    opt_fs = optRes.fun
        else:
            optRes, _, _ = self.optimizer.run_optimizer(
                self._logLikelihood,
                num_dim=bounds.shape[0],
                design_space=bounds,
            )
            opt_param = optRes["best_x"]
        # update the hyper-parameters
        self.opt_param = opt_param.squeeze()

    def _logLikelihood(self, params: np.ndarray) -> np.ndarray:
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
            rho = params[i, 0]
            theta = params[i, 1:]
            # correlation matrix R
            K = self.kernel(self.sample_xh_scaled,
                            self.sample_xh_scaled,
                            theta)
            L = cholesky(K)
            # responses for difference
            diff_y = self.sample_yh_scaled - rho * self.pred_ylh
            # R^(-1)(Y - rho * YL)
            alpha = solve(L.T, solve(L, diff_y))
            one = np.ones((self._num_xh, 1))
            # R^(-1)1
            beta = solve(L.T, solve(L, one))
            # 1R^(-1)Y / 1R^(-1)vector(1)
            mu = (np.dot(one.T, alpha) / np.dot(one.T, beta)).squeeze()
            gamma = solve(L.T, solve(L, (diff_y - mu)))
            sigma2 = np.dot((diff_y - mu).T, gamma) / self._num_xh
            # negative log likelihood
            logp = -0.5 * self._num_xh * \
                np.log(sigma2) - np.sum(np.log(np.diag(L)))
            out[i] = logp.ravel()
        return -out

    def _update_parameters(self) -> None:
        """Update parameters of the model"""
        # correlation matrix R
        self.K = self.kernel.get_kernel_matrix(self.sample_xh_scaled,
                                               self.sample_xh_scaled)
        L = cholesky(self.K)
        # R^(-1)Y
        self.diff_y = self.sample_yh_scaled - self.rho * self.pred_ylh
        alpha = solve(L.T, solve(L, self.diff_y))
        one = np.ones((self._num_xh, 1))
        # R^(-1)1
        beta = solve(L.T, solve(L, one))
        # 1R^(-1)Y / 1R^(-1)vector(1)
        self.mu_d = (np.dot(one.T, alpha) / np.dot(one.T, beta)).item()
        gamma = solve(L.T, solve(L, (self.diff_y - self.mu_d)))
        self.sigma2 = (np.dot((self.diff_y - self.mu_d).T,
                       gamma) / self._num_xh).item()
        self.logp = (-0.5 * self._num_xh * np.log(self.sigma2) -
                     np.sum(np.log(np.diag(L)))).item()
        # cov matrix for Co-Kriging
        self.C = np.concatenate(
            (
                np.concatenate(
                    (
                        self.lf_model.sigma2 * self.lf_model.K,
                        self.rho * self.lf_model.sigma2
                        * self.lf_model.kernel.get_kernel_matrix(
                            self.sample_xl_scaled, self.sample_xh_scaled),
                    ),
                    axis=1,
                ),
                np.concatenate(
                    (
                        self.rho * self.lf_model.sigma2
                        * self.lf_model.kernel.get_kernel_matrix(
                            self.sample_xh_scaled,
                            self.sample_xl_scaled),
                        self.rho**2 * self.lf_model.sigma2
                        * self.lf_model.kernel.get_kernel_matrix(
                            self.sample_xh_scaled,
                            self.sample_xh_scaled)
                        + self.sigma2 * self.K,
                    ),
                    axis=1,
                ),
            ),
            axis=0,
        )
        # all y values
        self.y = np.concatenate(
            (self.sample_yl_scaled, self.sample_yh_scaled), axis=0)
        self.LC = cholesky(
            self.C + np.eye(self.C.shape[1]) * 10**-10)
        oneC = np.ones((self.C.shape[0], 1))
        self.mu = oneC.T.dot(
            solve(self.LC.T, solve(self.LC, self.y))
        ) / oneC.T.dot(solve(self.LC.T, solve(self.LC, oneC))).item()
