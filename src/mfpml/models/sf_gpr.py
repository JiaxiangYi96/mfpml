from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import cholesky, solve
from scipy.optimize import minimize

from .kernels import RBF


class GP:
    """base class for Gaussian Process models"""

    def train(self, sample_x: np.ndarray, sample_y: np.ndarray) -> None:
        """traning procedure of gpr models

        Parameters
        ----------
        sample_x : np.ndarray
            sample array of sample
        sample_y : np.ndarray
            responses of the sample
        """
        # normalize the input
        # the original sample_x
        self.sample_x = sample_x
        self.sample_scaled_x = self.normalize_input(sample_x, self.bounds)
        # get the response
        self.sample_y = sample_y.reshape(-1, 1)

        # get number samples
        self.num_samples = self.sample_x.shape[0]

        # optimizer hyper-parameters
        self._hyper_paras_optimization()

        # update kernel matrix info with optimized hyper-parameters
        self._update_kernel_matrix()

    def update_optimizer(self, optimizer: Any) -> None:
        """Change the optimizer for optimizing hyper parameters

        Parameters
        ----------
        optimizer : any
            instance of optimizer
        """
        self.optimizer = optimizer

    def update_model(self, update_x: np.ndarray, update_y: np.ndarray) -> None:

        sample_x = np.concatenate((self.sample_x, update_x))
        sample_y = np.concatenate((self.sample_y, update_y))
        # update the model
        self.train(sample_x, sample_y)

    def plot_prediction(self, fig_name: str = "gpr_pred",
                        save_fig: bool = False, **kwargs,
                        ) -> None:
        """plot model prediction

        Parameters
        ----------
        fig_name : str, optional
            figure name, by default "gpr_pred"
        save_fig : bool, optional
            save figure otr not, by default False
        """

        if self.num_dim == 1:
            x_plot = np.linspace(
                start=self.bounds[0, 0], stop=self.bounds[0, 1], num=1000
            )
            x_plot = x_plot.reshape((-1, 1))
            y_pred, y_sigma = self.predict(x_plot, return_std=True)
            # with plt.style.context(["ieee", "science"]):
            fig, ax = plt.subplots(**kwargs)
            ax.plot(self.sample_x, self.sample_y, "ro", label="samples",)
            ax.plot(x_plot, y_pred, "--", color='b', label="pred mean")
            ax.fill_between(
                x_plot.ravel(),
                (y_pred + 2 * y_sigma).ravel(),
                (y_pred - 2 * y_sigma).ravel(),
                color="g",
                alpha=0.3,
                label=r"95% confidence interval",
            )
            ax.tick_params(axis="both", which="major", labelsize=12)
            plt.legend(loc='best')
            plt.xlabel(r"$x$", fontsize=12)
            plt.ylabel(r"$y$", fontsize=12)
            plt.grid()
            if save_fig is True:
                fig.savefig(fig_name, dpi=300, bbox_inches="tight")
            plt.show()

    @staticmethod
    def normalize_input(sample_x: np.ndarray,
                        bounds: np.ndarray) -> np.ndarray:
        """Normalize samples to range [0, 1]

        Parameters
        ----------
        sample_x : np.ndarray
            samples to scale
        bounds : np.ndarray
            bounds with shape=((num_dim, 2))

        Returns
        -------
        np.ndarray
            normalized samples
        """
        return (sample_x - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])

    @property
    def _num_samples(self) -> int:
        """Return the number of samples

        Returns
        -------
        num_samples : int
            num samples
        """
        return self.sample_x.shape[0]

# =========================================================================== #


class Kriging(GP):
    """
    Kriging model for single fidelity, it is noted that Kriging model is the
    noise free version of the Gaussian Process model. To be specific, the
    Kriging model in the repo using RBF kernel with the following form:
    k(x,y) = exp(-theta*(x-y)^2)
    where theta is the hyper-parameter to be optimized.
    """

    def __init__(
        self,
        design_space: np.ndarray,
        optimizer: Any = None,
        kernel_bound: list = [-4.0, 3.0],
        mean_prior: float = 0.0,
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
        mean_prior : float, optional
            mean value for the prior, by default 0
        """
        # dimension of the modeling problem
        self.num_dim = design_space.shape[0]
        # bounds of the design space
        self.bounds = design_space
        # kernel
        self.kernel = RBF(theta=np.zeros(
            self.num_dim), bounds=kernel_bound)
        # optimizer
        self.optimizer = optimizer
        # mean prior
        self.mean_prior = mean_prior

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
        # get the kernel matrix for predicted samples(scaled sampless)
        knew = self.kernel.get_kernel_matrix(self.sample_scaled_x, sample_new)
        # calculate the predicted mean
        fmean = self.mu + np.dot(knew.T, self.gamma)
        fmean = fmean.reshape(-1, 1)
        # calculate the standard deviation
        if not return_std:
            return fmean
        else:
            one = np.ones((self.num_samples, 1))
            delta = solve(self.L.T, solve(self.L, knew))
            # first use vectorization to calculate the epistemic uncertainty
            try:
                mse = self.sigma2 * (1 - np.diag(np.dot(knew.T, delta)) +
                                     np.diag((1 - knew.T.dot(delta)) ** 2
                                             / one.T.dot(self.beta)))
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
                        mse[i * batch: (i+1) * batch] = self.sigma2 * \
                            (1 - np.diag(np.dot(knew_i.T, delta_i)) +
                                np.diag((1 - knew_i.T.dot(delta_i)) ** 2
                                        / one.T.dot(self.beta))).reshape(-1, 1)

                    except Exception:
                        # remain part of the matrix
                        knew_i = knew[:, i * batch:]
                        delta_i = solve(self.L.T, solve(self.L, knew_i))
                        mse[i * batch:] = self.sigma2 * \
                            (1 - np.diag(np.dot(knew_i.T, delta_i)) +
                                np.diag((1 - knew_i.T.dot(delta_i)) ** 2
                                        / one.T.dot(self.beta))).reshape(-1, 1)

            # epistemic uncertainty
            std = np.sqrt(np.maximum(mse, 0)).reshape(-1, 1)

            return fmean, std

    def _hyper_paras_optimization(self, grads: bool = None) -> None:
        """Optimize the hyper_parameters

        Parameters
        ----------
        grads : bool, optional
            whether to use gradients, by default None
        """

        if self.optimizer is None:
            # use the L-BFGS-B method in scipy
            n_trials = 10
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
            # use the optimizer in the repo
            optRes, _, _ = self.optimizer.run_optimizer(
                self._logLikelihood,
                num_dim=self.kernel._get_num_para,
                design_space=self.kernel._get_bounds,
            )
            opt_param = optRes["best_x"]
        # update the kernel with optimized hyper-parameters
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
        out = np.zeros(params.shape[0])

        for i in range(params.shape[0]):
            # for optimization every row is a parameter set
            param = params[i, :]

            # correlation matrix R
            # calculate the covariance matrix
            K = self.kernel(self.sample_scaled_x,
                            self.sample_scaled_x,
                            param)
            #
            L = cholesky(K, lower=True)
            # R^(-1)Y
            alpha = solve(L.T, solve(L, self.sample_y))

            one = np.ones((self.num_samples, 1))
            # R^(-1)1
            beta = solve(L.T, solve(L, one))
            # 1R^(-1)Y / 1R^(-1)vector(1)
            mu = (np.dot(one.T, alpha) / np.dot(one.T, beta)).squeeze()

            gamma = solve(L.T, solve(L, (self.sample_y - mu)))

            sigma2 = np.dot((self.sample_y - mu).T, gamma) / self.num_samples

            logp = -0.5 * self.num_samples * \
                np.log(sigma2) - np.sum(np.log(np.diag(L)))
            out[i] = logp.ravel()

        return -out

    def _update_kernel_matrix(self) -> None:
        # assign the best hyper-parameter to the kernel
        self.kernel.set_params(self.opt_param)
        # update parameters with optimized hyper-parameters
        self.K = self.kernel.get_kernel_matrix(self.sample_scaled_x,
                                               self.sample_scaled_x)
        self.L = cholesky(self.K, lower=True)
        self.alpha = solve(self.L.T, solve(self.L, self.sample_y))
        one = np.ones((self.num_samples, 1))
        self.beta = solve(self.L.T, solve(self.L, one))
        self.mu = (np.dot(one.T, self.alpha) / np.dot(one.T, self.beta)).item()
        self.gamma = solve(self.L.T, solve(self.L, (self.sample_y - self.mu)))
        self.sigma2 = (
            np.dot((self.sample_y - self.mu).T, self.gamma) / self.num_samples
        ).item()
        self.logp = (-0.5 * self.num_samples * np.log(self.sigma2) -
                     np.sum(np.log(np.diag(self.L)))).item()


class GaussianProcessRegressor(GP):
    """
    Gaussian Process Regressor with noise
    """

    def __init__(
        self,
        design_space: np.ndarray,
        optimizer: Any = None,
        kernel_bound: list = [-4.0, 3.0],
        noise_prior: float = 1.0,
    ) -> None:
        """Initialize the Gaussian Process Regressor

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
        noise_prior : float, optional
            mean value for the prior, by default 0
        """
        # dimension of the modeling problem
        self.num_dim = design_space.shape[0]
        # bounds of the design space
        self.bounds = design_space
        # optimizer
        self.optimizer = optimizer
        # noise
        self.noise = noise_prior
        # kernel
        self.kernel = RBF(theta=np.zeros(self.num_dim), bounds=kernel_bound)

    def predict(self,
                x_predict: np.ndarray,
                return_std: bool = False):
        """Predict responses through the Gaussian Process Regressor

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
        # calculate the kernel matrix for predicted samples
        knew = self.kernel.get_kernel_matrix(self.sample_scaled_x, sample_new)
        # get the predicted mean
        fmean = self.mu + np.dot(knew.T, self.gamma)
        fmean = fmean.reshape(-1, 1)
        # calculate the standard deviation
        if not return_std:
            return fmean
        else:
            one = np.ones((self.num_samples, 1))
            delta = solve(self.L.T, solve(self.L, knew))
            # epistemic uncertainty
            try:
                mse = self.sigma2 * (1 - np.diag(np.dot(knew.T, delta)) +
                                     np.diag((1 - knew.T.dot(delta)) ** 2
                                             / one.T.dot(self.beta)))
            except Exception:
                print("The matrix is too big, use for loop to calculate")
                mse = np.zeros((knew.shape[1], 1))
                batch = 100
                iter = knew.shape[1] / batch
                for i in range(int(iter)):
                    try:
                        knew_i = knew[:, i * batch: (i+1)*batch]
                        delta_i = solve(self.L.T, solve(self.L, knew_i))
                        mse[i * batch: (i+1) * batch] = self.sigma2 * \
                            (1 - np.diag(np.dot(knew_i.T, delta_i)) +
                                np.diag((1 - knew_i.T.dot(delta_i)) ** 2
                                        / one.T.dot(self.beta))).reshape(-1, 1)
                    except Exception:
                        # remain part of the matrix
                        knew_i = knew[:, i * batch:]
                        delta_i = solve(self.L.T, solve(self.L, knew_i))
                        mse[i * batch:] = self.sigma2 * \
                            (1 - np.diag(np.dot(knew_i.T, delta_i)) +
                                np.diag((1 - knew_i.T.dot(delta_i)) ** 2
                                        / one.T.dot(self.beta))).reshape(-1, 1)
            # aleatoric uncertainty
            data_noise = self.noise**2

            # total uncertainty
            std = np.sqrt(np.maximum(mse+data_noise, 0)).reshape(-1, 1)
            return fmean, std

    def _hyper_paras_optimization(self, grads: bool = None) -> None:
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
            optRes, _, _ = self.optimizer.run_optimizer(
                self._logLikelihood,
                num_dim=num_hyper,
                design_space=hyper_bounds,
            )

            opt_param = optRes["best_x"]
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
            K = self.kernel(self.sample_scaled_x,
                            self.sample_scaled_x,
                            param) + np.eye(self.num_samples) * noise_sigma**2
            #
            L = cholesky(K, lower=True)
            # R^(-1)Y
            alpha = solve(L.T, solve(L, self.sample_y))

            logp = -0.5 * np.dot(self.sample_y.T, alpha) - np.sum(
                np.log(np.diag(L))) - 0.5*self.num_samples*np.log(2*np.pi)

            out[i] = logp.ravel()

        return -out

    def _update_kernel_matrix(self) -> None:
        # assign the best hyper-parameter to the kernel
        self.kernel.set_params(self.opt_param[0:(len(self.opt_param)-1)])
        self.noise = self.opt_param[-1]
        # update parameters with optimized hyper-parameters
        self.K = self.kernel.get_kernel_matrix(
            self.sample_scaled_x,
            self.sample_scaled_x) + np.eye(self.num_samples) * self.noise**2
        self.L = cholesky(self.K, lower=True)
        self.alpha = solve(self.L.T, solve(self.L, self.sample_y))
        one = np.ones((self.num_samples, 1))
        self.beta = solve(self.L.T, solve(self.L, one))
        self.mu = (np.dot(one.T, self.alpha) / np.dot(one.T, self.beta)).item()
        self.gamma = solve(self.L.T, solve(self.L, (self.sample_y - self.mu)))
        self.sigma2 = (
            np.dot((self.sample_y - self.mu).T, self.gamma) / self.num_samples
        ).item()
        # self.logp = (-0.5 * self.num_samples * np.log(self.sigma2) -
        #              np.sum(np.log(np.diag(self.L)))).item()
        self.logp = -0.5 * np.dot(self.sample_y.T, self.alpha) - np.sum(
            np.log(np.diag(self.L))) - 0.5*self.num_samples*np.log(2*np.pi)
