import time
from typing import Any

import numpy as np
from numpy.linalg import cholesky, solve
from scipy.optimize import minimize

from .basis_functions import Ordinary
from .kernels import RBF


class GaussianProcessRegression:
    """
    Gaussian Process Regressor, it can be used for noisy data or noise-free data
    """

    def __init__(
        self,
        design_space: np.ndarray,
        kernel: Any = None,
        regr: Any = Ordinary(),
        optimizer: Any = None,
        noise_prior: float = 0.0,
        optimizer_restart: int = 5,
    ) -> None:
        """Initialize the Gaussian Process Regressor

        Parameters
        ----------
        design_space : np.ndarray
            design space of the problem with shape=(num_dim, 2), it is used to
            normalize the input samples
        kernel : Any, optional
            kernel function for spatial correlation of samples, by default None
        regr : Any, optional
            mean function, by default Ordinary()
        optimizer : Any, optional
            optimizer for optimizing the log likelihood function , by default None
        noise_prior : float, optional
            noise prior of the , by default 0.0 (noise-free data), if the
            noise is not None, the type II maximum likelihood should be used
            to optimize the for both the hyper-parameters and noise value
        optimizer_restart : int, optional
            restart the optimizer if needed, by default 5
        """

        # get the dimension of the problem
        self.num_dim = design_space.shape[0]
        # bounds of the design space
        self.bounds = design_space

        # get kernel
        if kernel is None:
            self.kernel = RBF(theta=np.zeros(self.num_dim))
        else:
            self.kernel = kernel

        # optimizer
        self.optimizer = optimizer
        self.optimizer_restart = optimizer_restart
        # basis function
        self.regr = regr
        # noisy data prior or not
        self.noise = noise_prior

    def train(self,
              X: np.ndarray,
              Y: np.ndarray) -> None:
        """training procedure of gpr

        Parameters
        ----------
        X : np.ndarray
            sample array of sample
        Y : np.ndarray
            responses of the sample
        """
        # record the training time for the model
        time_start = time.time()
        # get number samples
        self.num_samples = X.shape[0]
        # the original sample_x
        self.sample_x = X
        self.sample_scaled_x = self.normalize_input(X, self.bounds)
        # get the response
        self.sample_y = Y.reshape(-1, 1)
        self.sample_y_scaled = self.normalize_output(self.sample_y)

        # optimizer hyper-parameters
        self._hyper_paras_optimization()

        # update kernel matrix info with optimized hyper-parameters
        self._update_kernel_matrix()
        time_end = time.time()
        self.training_time = time_end - time_start

    def predict(self,
                X: np.ndarray,
                return_std: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Predict responses through the Kriging model

        Parameters
        ----------
        X : np.ndarray
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
        sample_new = self.normalize_input(X, self.bounds)
        sample_new = np.atleast_2d(sample_new)
        # get the kernel matrix for predicted samples(scaled samples)
        knew = self.kernel.get_kernel_matrix(self.sample_scaled_x, sample_new)
        # calculate the predicted mean
        f = self.regr(sample_new)
        fmean = np.dot(f, self.beta) + np.dot(knew.T, self.gamma)
        fmean = fmean*self.y_std + self.y_mean
        # calculate the standard deviation
        if not return_std:
            return fmean
        else:
            delta = solve(self.L.T, solve(self.L, knew))
            R = f.T - np.dot(self.f.T, delta)
            # first use vectorization to calculate the epistemic uncertainty
            mse = self.sigma2 * \
                (1 - np.diag(np.dot(knew.T, delta)) +
                    np.diag(R.T.dot(solve(self.ld.T, solve(self.ld, R))))
                 )

            data_noise = self.noise**2

            # epistemic uncertainty
            std = np.sqrt(np.maximum(mse*self.y_std**2 +
                          data_noise, 0)).reshape(-1, 1)

            return fmean, std

    def _hyper_paras_optimization(self) -> None:

        if self.noise is None:
            # the type II maximum likelihood should be used to optimize the
            # for both the hyper-parameters and noise value
            # noise value needs to be optimized
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
        else:
            lower_bound = self.kernel._get_low_bound
            upper_bound = self.kernel._get_high_bound
            # bounds for the hyper-parameters
            hyper_bounds = np.vstack((lower_bound, upper_bound)).T
            # number of hyper-parameters
            num_hyper = self.kernel._get_num_para

        if self.optimizer is None:
            # use L-BFGS-B method in scipy
            n_trials = self.optimizer_restart + 1
            optimum_value = float("inf")
            for _ in range(n_trials):
                # initial point
                x0 = np.random.uniform(
                    low=lower_bound,
                    high=upper_bound,
                    size=num_hyper,
                )
                # get the optimum value
                optimum_info = minimize(
                    self._logLikelihood,
                    x0=x0,
                    method="L-BFGS-B",
                    bounds=hyper_bounds,
                )
                # greedy search for the optimum value
                if optimum_info.fun < optimum_value:
                    opt_param = optimum_info.x
                    optimum_value = optimum_info.fun
        else:
            # use the optimizer in the repo, such as DE and PSO
            optimum_info, _, _ = self.optimizer.run_optimizer(
                self._logLikelihood,
                num_dim=num_hyper,
                design_space=hyper_bounds,
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
        np.ndarray
            log likelihood
        """
        params = np.atleast_2d(params)
        num_params = params.shape[1]
        nll = np.zeros(params.shape[0])

        for i in range(params.shape[0]):
            # for optimization every row is a parameter set
            if self.noise is None:
                param = params[i, 0: num_params-1]
                noise_sigma = params[i, -1]
            else:
                param = params[i, 0: num_params]
                noise_sigma = self.noise/self.y_std

            # calculate the covariance matrix with noise term added
            K = self.kernel(self.sample_scaled_x,
                            self.sample_scaled_x,
                            param) + np.eye(self.num_samples) * noise_sigma**2
            L = cholesky(K)
            # step 1: estimate beta, which is the coefficient of basis function
            # f, basis function
            f = self.regr(self.sample_scaled_x)
            # alpha = K^(-1) * Y
            alpha = solve(L.T, solve(L, self.sample_y_scaled))
            # K^(-1)f
            KF = solve(L.T, solve(L, f))
            # cholesky decomposition for (F^T *K^(-1)* F)
            ld = cholesky(np.dot(f.T, KF))
            beta = solve(ld.T, solve(ld, np.dot(f.T, alpha)))

            # step 2: estimate sigma2
            # gamma = 1/n * (Y - F * beta)^T * K^(-1) * (Y - F * beta)
            gamma = solve(L.T, solve(
                L, (self.sample_y_scaled - np.dot(f, beta))))
            sigma2 = np.dot((self.sample_y_scaled - np.dot(f, beta)).T,
                            gamma) / self.num_samples

            # step 3: calculate the log likelihood
            if self.noise == 0.0:
                logp = -0.5 * self.num_samples * \
                    np.log(sigma2) - np.sum(np.log(np.diag(L)))
            else:
                logp = -0.5 * self.num_samples * \
                    sigma2 - np.sum(np.log(np.diag(L)))
            nll[i] = -logp.ravel()

        return nll

    def _update_kernel_matrix(self) -> None:
        """update the kernel matrix with optimized parameters
        """
        # assign the best hyper-parameter to the kernel
        if self.noise is None:
            self.kernel.set_params(self.opt_param[0:(len(self.opt_param)-1)])
            self.noise = self.opt_param[-1]*self.y_std
        else:
            self.kernel.set_params(self.opt_param)
        # update parameters with optimized hyper-parameters
        self.K = self.kernel.get_kernel_matrix(
            self.sample_scaled_x,
            self.sample_scaled_x) + \
            np.eye(self.num_samples) * (self.noise/self.y_std)**2
        self.L = cholesky(self.K)

        #  step 1: get the optimal beta
        #  f, basis function
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
        self.logp = (-0.5 * self.num_samples * self.sigma2 -
                     np.sum(np.log(np.diag(self.L)))).item()

    def change_optimizer(self, optimizer: Any) -> None:
        """Change the optimizer of the model

        Parameters
        ----------
        optimizer : Any
            optimizer for optimizing the log likelihood function
        """
        self.optimizer = optimizer

    @staticmethod
    def normalize_input(X: np.ndarray,
                        bounds: np.ndarray) -> np.ndarray:
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

    @staticmethod
    def normalize_output(Y: np.ndarray) -> np.ndarray:
        """Normalize output to range [0, 1]

        Parameters
        ----------
        sample_y : np.ndarray
            output to scale

        Returns
        -------
        np.ndarray
            normalized output
        """
        # normalize the output
        y_mean = np.mean(Y)
        y_std = np.std(Y)

        return (Y - y_mean) / y_std

    @property
    def _num_samples(self) -> int:
        """Return the number of samples

        Returns
        -------
        num_samples : int
            num samples
        """
        return self.sample_x.shape[0]

    @property
    def y_mean(self) -> float:
        """Return the mean of the response

        Returns
        -------
        float
            mean of the response
        """
        return np.mean(self.sample_y)

    @property
    def y_std(self) -> float:
        """Return the standard deviation of the response

        Returns
        -------
        float
            standard deviation of the response
        """
        return np.std(self.sample_y)
