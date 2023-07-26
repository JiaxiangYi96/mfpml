from typing import Any

import numpy as np
from scipy.linalg import cholesky, solve
from scipy.optimize import minimize

from .kernels import RBF


class gpr:
    """
    gpr model with noise term
    """

    def __init__(
        self,
        design_space: np.ndarray,
        optimizer: Any = None,
        kernel_bound: list = [-4.0, 3.0],
        noise: float = 1.0,
    ) -> None:
        """Initialize the Kriging model

        Parameters
        ----------
        design_space : np.ndarray
            array of shape=((num_dim,2)) where each row describes
            the bound for the variable on specific dimension
        optimizer : any, optional
            instance of the optimizer used to optimize the hyper parameters
            with the use style optimizer.run_optimizer(objective function,
            number of dimension, design space of variables), if not assigned,
            the 'L-BFGS-B' method in scipy is used
        kernel_bound : list, optional
            log bound for the Kriging kernel, by default [-4, 3]
        noise : float, optional
            mean value for the prior, by default 0
        """
        # dimension of the modeling problem
        self.num_dim = design_space.shape[0]
        # bounds of the design space
        self.bounds = design_space
        # optimizer
        self.optimizer = optimizer
        # noise
        self.noise = noise
        # kernel
        self.kernel = RBF(theta=np.zeros(self.num_dim), bounds=kernel_bound)

    def train(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Train the Kriging model

        Parameters
        ----------
        X : np.ndarray
            sample array of sample
        Y : np.ndarray
            responses of the sample
        """
        self.sample_X = X
        self.X = self.normalize_input(X, self.bounds)
        self.sample_Y = Y.reshape(-1, 1)
        # get number samples
        self.num_samples = self.X.shape[0]
        # optimizer hyper-parameters
        self._hyper_paras_optimization()

        # update kernel matrix info with optimized hyper-parameters
        self._update_kernel_matrix()

    def predict(self, Xinput: np.ndarray, return_std: bool = False):
        """Predict responses through the Kriging model

        Parameters
        ----------
        Xinput : np.ndarray
            new sample need to predict
        return_std : bool, optional
            whether return the standard deviation
            , by default False

        Returns
        -------
        np.ndarray
            return the prediction with shape (#Xinput, 1)
        """
        Xnew = self.normalize_input(Xinput, self.bounds)
        Xnew = np.atleast_2d(Xnew)
        knew = self.kernel.get_kernel_matrix(self.X, Xnew)
        fmean = self.mu + np.dot(knew.T, self.gamma)
        if not return_std:
            return fmean.reshape(-1, 1)
        else:
            one = np.ones((self.num_samples, 1))
            delta = solve(self.L.T, solve(self.L, knew))
            # epistemic uncertainty
            mse = self.sigma2 * (1 - np.diag(np.dot(knew.T, delta)) +
                                 np.diag((1 - knew.T.dot(delta)) ** 2
                                         / one.T.dot(self.beta)))
            # aleatoric uncertainty
            data_noise = self.noise**2
            # mse = self.sigma2 * (1 - np.diag(np.dot(knew.T, delta)))
            return fmean.reshape(-1, 1), np.sqrt(np.maximum(mse+data_noise, 0)).reshape(-1, 1)

    def _hyper_paras_optimization(self, grads: bool = None):
        """Optimize the hyper_parameters

        Parameters
        ----------
        grads : bool, optional
            whether to use gradients, by default None
        """
        # set up the optimization problems
        lower_bound_theta = self.kernel._get_low_bound
        upper_bound_theta = self.kernel._get_high_bound
        # set up the bounds for noise sigma
        lower_bound_sigma = 1e-5
        upper_bound_sigma = 10
        # set up the bounds for the hyper-parameters
        lower_bound = np.hstack((lower_bound_theta, lower_bound_sigma))
        upper_bound = np.hstack((upper_bound_theta, upper_bound_sigma))
        # bounds for the hyper-parameters
        hyper_bounds = np.vstack((lower_bound, upper_bound)).T
        # number of hyper-parameters
        num_hyper = self.kernel._get_num_para + 1

        if self.optimizer is None:
            n_trials = 10
            opt_fs = float("inf")
            for _ in range(n_trials):
                x0 = np.random.uniform(
                    low=lower_bound,
                    high=upper_bound,
                    size=num_hyper,
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
            _, _, opt_param = self.optimizer.run_optimizer(
                self._logLikelihood,
                num_dim=num_hyper,
                design_space=hyper_bounds,
            )
            self.optimizer.plot_optimization_history()
            # opt_param = optRes["best_x"]
        # best hyper-parameters
        self.opt_param = opt_param

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
        num_params = params.shape[1]
        out = np.zeros(params.shape[0])

        for i in range(params.shape[0]):
            # for optimization every row is a parameter set
            param = params[i, 0: num_params-1]
            noise_sigma = params[i, -1]

            # correlation matrix R
            # calculate the covariance matrix
            K = self.kernel(self.X, self.X, param) + \
                np.eye(self.num_samples) * noise_sigma**2
            #
            L = cholesky(K, lower=True)
            # R^(-1)Y
            alpha = solve(L.T, solve(L, self.sample_Y))

            one = np.ones((self.num_samples, 1))
            # R^(-1)1
            beta = solve(L.T, solve(L, one))
            # 1R^(-1)Y / 1R^(-1)vector(1)
            mu = (np.dot(one.T, alpha) / np.dot(one.T, beta)).squeeze()

            gamma = solve(L.T, solve(L, (self.sample_Y - mu)))

            sigma2 = np.dot((self.sample_Y - mu).T, gamma) / self.num_samples

            # logp = -0.5 * self.num_samples * \
            #     np.log(sigma2) - np.sum(np.log(np.diag(L)))
            logp = -0.5 * np.dot(self.sample_Y.T, alpha) - np.sum(
                np.log(np.diag(L))) - 0.5*self.num_samples*np.log(2*np.pi)

            out[i] = logp.ravel()

        return -out

    def _update_kernel_matrix(self) -> None:
        # assign the best hyper-parameter to the kernel
        self.kernel.set_params(self.opt_param[0:(len(self.opt_param)-1)])
        self.noise = self.opt_param[-1]
        # update parameters with optimized hyper-parameters
        self.K = self.kernel.get_kernel_matrix(
            self.X, self.X) + np.eye(self.num_samples) * self.noise**2
        self.L = cholesky(self.K, lower=True)
        self.alpha = solve(self.L.T, solve(self.L, self.sample_Y))
        one = np.ones((self.num_samples, 1))
        self.beta = solve(self.L.T, solve(self.L, one))
        self.mu = (np.dot(one.T, self.alpha) / np.dot(one.T, self.beta)).item()
        self.gamma = solve(self.L.T, solve(self.L, (self.sample_Y - self.mu)))
        self.sigma2 = (
            np.dot((self.sample_Y - self.mu).T, self.gamma) / self.num_samples
        ).item()
        # self.logp = (-0.5 * self.num_samples * np.log(self.sigma2) -
        #              np.sum(np.log(np.diag(self.L)))).item()
        self.logp = -0.5 * np.dot(self.sample_Y.T, self.alpha) - np.sum(
            np.log(np.diag(self.L))) - 0.5*self.num_samples*np.log(2*np.pi)

    def _update_optimizer(self, optimizer: Any) -> None:
        """Change the optimizer for optimizing hyper parameters

        Parameters
        ----------
        optimizer : any
            instance of optimizer
        """
        self.optimizer = optimizer

    @staticmethod
    def normalize_input(X: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        """Normalize samples to range [0, 1]

        Parameters
        ----------
        X : np.ndarray
            samples to scale
        bounds : np.ndarray
            bounds with shape=((num_dim, 2))

        Returns
        -------
        np.ndarray
            normalized samples
        """
        return (X - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])

    def getkernelparams(self):
        pass

    @property
    def _num_X(self) -> int:
        """Return the number of samples

        Returns
        -------
        int
            #samples
        """
        return self.sample_X.shape[0]
