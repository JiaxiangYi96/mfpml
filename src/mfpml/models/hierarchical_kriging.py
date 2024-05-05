
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
        kernel_bound: list = [-2.0, 3.0],
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
        F = self.predict_lf(X)
        self.F = (F-self.yh_mean)/self.yh_std
        # optimize the hyper parameters
        self._optHyp()
        # update the parameters
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
        # # original prediction of lf
        # pre_lf = self.predict_lf(X)
        # # scale it to hf
        # pre_lf = (pre_lf - self.yh_mean) / self.yh_std

        # # normalize the input
        # XHnew = np.atleast_2d(self.normalize_input(X))
        # knew = self.kernel.get_kernel_matrix(XHnew, self.sample_xh_scaled)

        # # get the prediction
        # fmean = self.beta * pre_lf + np.dot(knew, self.gamma)
        # # scale it back to original scale
        # fmean = fmean * self.yh_std + self.yh_mean
        # if not return_std:
        #     return fmean.reshape(-1, 1)
        # else:
        #     delta = solve(self.L.T, solve(self.L, knew.T))
        #     mse = self.sigma2 * (
        #         1
        #         - np.diag(knew.dot(delta))
        #         + np.diag(
        #             np.dot(
        #                 (knew.dot(self.beta) - pre_lf),
        #                 (knew.dot(self.beta) - pre_lf).T,
        #             )
        #         )
        #         / self.F.T.dot(self.beta)
        #     )

        #     # scale it back to original scale
        #     self.epis_std = np.sqrt(np.maximum(mse, 0))*self.yh_std

        #     # total std
        #     total_std = np.sqrt((self.epis_std + self.noise**2))
        # normalize the input
        sample_new = self.normalize_input(X)
        sample_new = np.atleast_2d(sample_new)
        # get the kernel matrix for predicted samples(scaled samples)
        knew = self.kernel.get_kernel_matrix(self.sample_xh_scaled, sample_new)
        # calculate the predicted mean
        f = self.predict_lf(X, return_std=False)
        # scale the prediction to high-fidelity
        f = (f - self.yh_mean) / self.yh_std
        # get the mean
        fmean = np.dot(f, self.beta) + np.dot(knew.T, self.gamma)
        fmean = (fmean * self.yh_std + self.yh_mean).reshape(-1, 1)
        # calculate the standard deviation
        if not return_std:
            return fmean.reshape(-1, 1)
        else:
            delta = solve(self.L.T, solve(self.L, knew))
            R = f.T - np.dot(self.F.T, delta)
            # epistemic uncertainty calculation
            mse = self.sigma2 * \
                (1 - np.diag(np.dot(knew.T, delta)) +
                    np.diag(R.T.dot(solve(self.ld.T, solve(self.ld, R))))
                 )
            std = np.sqrt(np.maximum(mse, 0)).reshape(-1, 1)
            # epistemic uncertainty scale back
            self.epistemic = std*self.yh_std

            # total uncertainty
            total_unc = np.sqrt(self.epistemic**2 + self.noise**2)
            return fmean, total_unc

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
        nll = np.zeros(params.shape[0])
        for i in range(params.shape[0]):
            # for optimization every row is a parameter set
            if self.noise is None:
                param = params[i, 0: num_params - 1]
                noise_sigma = params[i, -1]
            else:
                param = params[i, :]
                noise_sigma = self.noise / self.yh_std

            # Step 1: estimate beta, which is the coefficient of basis function
            # f, basis function
            # f = self.predict_lf(self.sample_xh)
            # alpha = K^(-1) * Y
            # calculate the covariance matrix
            K = self.kernel(self.sample_xh_scaled,
                            self.sample_xh_scaled,
                            param) + noise_sigma**2 * np.eye(self._num_xh)
            L = cholesky(K)
            alpha = solve(L.T, solve(L, self.sample_yh_scaled))
            # K^(-1)f
            KF = solve(L.T, solve(L, self.F))
            # cholesky decomposition for (F^T *K^(-1)* F)
            ld = cholesky(np.dot(self.F.T, KF))
            # beta = (F^T *K^(-1)* F)^(-1) * F^T *R^(-1) * Y
            beta = solve(ld.T, solve(ld, np.dot(self.F.T, alpha)))

            # step 2: estimate sigma2
            # gamma = 1/n * (Y - F * beta)^T * K^(-1) * (Y - F * beta)
            gamma = solve(L.T, solve(
                L, (self.sample_yh_scaled - np.dot(self.F, beta))))
            sigma2 = np.dot((self.sample_yh_scaled - np.dot(self.F, beta)).T,
                            gamma) / self._num_xh

            # step 3: calculate the log likelihood
            logp = -0.5 * self._num_xh * sigma2 - np.sum(np.log(np.diag(L)))

            nll[i] = -logp.ravel()

        return nll

    def _update_parameters(self) -> None:
        """Update parameters of the model"""
        # update parameters with optimized hyper-parameters
        if self.noise is None:
            self.noise = self.opt_param[-1]*self.yh_std
            self.kernel.set_params(self.opt_param[:-1])
        else:
            self.kernel.set_params(self.opt_param)
        # get the kernel matrix
        self.K = self.kernel.get_kernel_matrix(
            self.sample_xh_scaled, self.sample_xh_scaled) + \
            (self.noise/self.yh_std)**2 * np.eye(self._num_xh)

        self.L = cholesky(self.K)

        # step 1: get the optimal beta
        # alpha = K^(-1) * Y
        self.alpha = solve(self.L.T, solve(self.L, self.sample_yh_scaled))
        # K^(-1)f
        self.KF = solve(self.L.T, solve(self.L, self.F))
        self.ld = cholesky(np.dot(self.F.T, self.KF))
        # beta = (F^T *K^(-1)* F)^(-1) * F^T *R^(-1) * Y
        self.beta = solve(self.ld.T, solve(
            self.ld, np.dot(self.F.T, self.alpha)))

        # step 2: get the optimal sigma2
        self.gamma = solve(self.L.T, solve(
            self.L, (self.sample_yh_scaled - np.dot(self.F, self.beta))))
        self.sigma2 = np.dot((self.sample_yh_scaled - np.dot(self.F, self.beta)).T,
                             self.gamma) / self._num_xh

        # step 3: get the optimal log likelihood
        self.logp = (-0.5 * self._num_xh * self.sigma2 -
                     np.sum(np.log(np.diag(self.L)))).item()
