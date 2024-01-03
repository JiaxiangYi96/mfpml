from typing import Any

import numpy as np
from numpy.linalg import cholesky, solve
from scipy.optimize import minimize

from .basis_functions import Ordinary
from .gpr_base import SingleFidelityGP
from .kernels import RBF


class Kriging(SingleFidelityGP):
    """Deterministic Gaussian Process Regression (GPR) model."""

    def __init__(self,
                 design_space: np.ndarray,
                 kernel: Any = None,
                 regr: Any = Ordinary(),
                 optimizer: Any = None,
                 optimizer_restart: int = 0) -> None:

        # initialize parameters
        self.num_dim = design_space.shape[0]
        # bounds of design space
        self.bounds = design_space

        # getting kernel
        if kernel is None:
            # if kernel is None, use RBF kernel as default
            self.kernel = RBF(theta=np.zeros(self.num_dim))
        else:
            self.kernel = kernel

        # get optimizer for optimizing parameters
        self.optimizer = optimizer
        self.optimizer_restart = optimizer_restart
        # get basis function
        self.regr = regr

    def _hyper_paras_optimization(self) -> None:

        if self.optimizer is None:
            # use the L-BFGS-B method in scipy
            n_trials = self.optimizer_restart + 1
            optimum_value = float("inf")
            for _ in range(n_trials):
                # initial point
                x0 = np.random.uniform(
                    self.kernel._get_low_bound,
                    self.kernel._get_high_bound,
                    self.kernel._get_num_para,
                )
                # get the optimum value
                optimum_info = minimize(
                    self._logLikelihood,
                    x0=x0,
                    method="l-bfgs-b",
                    bounds=self.kernel._get_bounds_list,
                    options={"maxfun": 200},

                )
                # greedy search for the optimum value
                if optimum_info.fun < optimum_value:
                    opt_param = optimum_info.x
                    optimum_value = optimum_info.fun
        else:
            # use the optimizer in the repo, such as DE and PSO
            optimum_info, _, _ = self.optimizer.run_optimizer(
                self._logLikelihood,
                num_dim=self.kernel._get_num_para,
                design_space=self.kernel._get_bounds,
            )
            opt_param = optimum_info["best_x"]
        # update the kernel with optimized parameters
        self.opt_param = opt_param

    def _logLikelihood(self, params: np.ndarray) -> np.ndarray:
        """Compute the concentrated ln-likelihood

        Parameters
        ----------
        params : np.ndarray
            parameters of the kernel

        Returns
        -------
        nll : np.ndarray
            negative log likelihood
        """
        params = np.atleast_2d(params)
        nll = np.zeros(params.shape[0])

        for i in range(params.shape[0]):
            # for optimization every row is a parameter set
            param = params[i, :]
            # calculate the covariance matrix
            K = self.kernel(self.sample_scaled_x,
                            self.sample_scaled_x,
                            param)
            L = cholesky(K)
            # Step 1: estimate beta, which is the coefficient of basis function
            # f, basis function
            f = self.regr(self.sample_scaled_x)
            # alpha = K^(-1) * Y
            alpha = solve(L.T, solve(L, self.sample_y_scaled))
            # K^(-1)f
            KF = solve(L.T, solve(L, f))
            # KF = cholesky_solve(K, f)
            ld = cholesky(np.dot(f.T, KF))
            # beta = (F^T *K^(-1)* F)^(-1) * F^T *R^(-1) * Y
            beta = solve(ld.T, solve(ld, np.dot(f.T, alpha)))

            # step 2: estimate sigma2
            # gamma = 1/n * (Y - F * beta)^T * K^(-1) * (Y - F * beta)
            gamma = solve(L.T, solve(
                L, (self.sample_y_scaled - np.dot(f, beta))))
            sigma2 = np.dot((self.sample_y_scaled - np.dot(f, beta)).T,
                            gamma) / self.num_samples

            # step 3: calculate the log likelihood
            logp = -0.5 * self.num_samples * \
                np.log(sigma2) - np.sum(np.log(np.diag(L)))
            nll[i] = -logp.ravel()

        return nll

    def _update_kernel_matrix(self) -> None:
        """Update the kernel matrix with optimized parameters."""
        # assign the best hyper-parameter to the kernel
        self.kernel.set_params(self.opt_param)
        # update parameters with optimized hyper-parameters
        self.K = self.kernel.get_kernel_matrix(self.sample_scaled_x,
                                               self.sample_scaled_x)
        self.L = cholesky(self.K)

        # step 1: get the optimal beta
        # f, basis function
        self.f = self.regr(self.sample_scaled_x)
        # alpha = K^(-1) * Y
        self.alpha = solve(self.L.T, solve(self.L, self.sample_y_scaled))
        # K^(-1)f
        KF = solve(self.L.T, solve(self.L, self.f))
        self.ld = cholesky(np.dot(self.f.T, KF))
        # beta = (F^T *K^(-1)* F)^(-1) * F^T *R^(-1) * Y
        self.beta = solve(self.ld.T, solve(
            self.ld, np.dot(self.f.T, self.alpha)))

        # step 2: get the optimal sigma2
        self.gamma = solve(self.L.T, solve(
            self.L, (self.sample_y_scaled - np.dot(self.f, self.beta))))
        self.sigma2 = np.dot((self.sample_y_scaled -
                              np.dot(self.f, self.beta)).T, self.gamma) \
            / self.num_samples

        # step 3: get the optimal log likelihood
        self.logp = (-0.5 * self.num_samples * np.log(self.sigma2) -
                     np.sum(np.log(np.diag(self.L)))).item()

    def predict(self,
                x_predict: np.ndarray,
                return_std: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Predict responses through the Kriging model

        Parameters
        ----------
        x_predict : np.ndarray
            new sample need to predict
        return_std : bool, optional
            whether return the standard deviation
            , by default False

        Returns
        -------
        np.ndarray
            return the prediction with shape (#Xinput, 1)
        """
        # normalize the input
        sample_new = self.normalize_input(x_predict, self.bounds)
        sample_new = np.atleast_2d(sample_new)
        # get the kernel matrix for predicted samples(scaled samples)
        knew = self.kernel.get_kernel_matrix(self.sample_scaled_x, sample_new)
        # calculate the predicted mean
        f = self.regr(sample_new)
        fmean = np.dot(f, self.beta) + np.dot(knew.T, self.gamma)
        fmean = fmean * self.y_std + self.y_mean
        # calculate the standard deviation
        if not return_std:
            return fmean
        else:
            delta = solve(self.L.T, solve(self.L, knew))
            R = f.T - np.dot(self.f.T, delta)
            # first use vectorization to calculate the epistemic uncertainty
            try:
                mse = self.sigma2 * \
                    (1 - np.diag(np.dot(knew.T, delta)) +
                     np.diag(R.T.dot(solve(self.ld.T, solve(self.ld, R))))
                     )
            except Exception:
                print("The matrix is too big, use for loop to calculate")
                # if the matrix is too big, use for loop to calculate
                mse = np.zeros((knew.shape[1], 1))
                batch = 100
                iter = knew.shape[1] / batch
                for i in range(int(iter)):
                    try:
                        knew_i = knew[:, i * batch: (i+1)*batch]
                        delta_i = solve(self.L.T, solve(self.L, knew_i))
                        R_i = R[:, i * batch: (i+1)*batch]
                        mse[i * batch: (i+1)*batch] = self.sigma2 * \
                            (1 - np.diag(np.dot(knew_i.T, delta_i)) +
                                np.diag(R_i.T.dot(solve(self.ld.T,
                                                        solve(self.ld, R_i))))
                             ).reshape(-1, 1)

                    except Exception:
                        # remain part of the matrix
                        knew_i = knew[:, i * batch:]
                        delta_i = solve(self.L.T, solve(self.L, knew_i))
                        R_i = R[:, i * batch:]
                        mse[i * batch:] = self.sigma2 * \
                            (1 - np.diag(np.dot(knew_i.T, delta_i)) +
                                np.diag(R_i.T.dot(solve(self.ld.T,
                                                        solve(self.ld, R_i))))
                             ).reshape(-1, 1)

            # epistemic uncertainty
            std = self.y_std*np.sqrt(np.maximum(mse, 0)).reshape(-1, 1)

            return fmean, std
