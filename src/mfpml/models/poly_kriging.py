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


class Kriging_poly(GP):
    """
    Kriging model with first order polynomial trend function

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
        # get the kernel matrix for predicted samples(scaled samples)
        knew = self.kernel.get_kernel_matrix(self.sample_scaled_x, sample_new)
        # calculate the predicted mean
        one = np.hstack(
            (np.ones((sample_new.shape[0], 1)), sample_new))
        fmean = np.dot(one, self.mu) + np.dot(knew.T, self.gamma)
        fmean = fmean.reshape(-1, 1)
        # calculate the standard deviation
        if not return_std:
            return fmean
        else:
            delta = solve(self.L.T, solve(self.L, knew))
            R = one.T - np.dot(self.one.T, delta)
            # first use vectorization to calculate the epistemic uncertainty
            try:
                mse = self.sigma2 * \
                    (1 - np.diag(np.dot(knew.T, delta)) +
                     np.diag(R.T.dot(solve(self.low_denominator.T,
                                           solve(self.low_denominator, R))))
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
                                np.diag(R_i.T.dot(solve(self.low_denominator.T,
                                                        solve(self.low_denominator, R_i))))
                             ).reshape(-1, 1)

                    except Exception:
                        # remain part of the matrix
                        knew_i = knew[:, i * batch:]
                        delta_i = solve(self.L.T, solve(self.L, knew_i))
                        R_i = R[:, i * batch:]
                        mse[i * batch:] = self.sigma2 * \
                            (1 - np.diag(np.dot(knew_i.T, delta_i)) +
                                np.diag(R_i.T.dot(solve(self.low_denominator.T,
                                                        solve(self.low_denominator, R_i))))
                             ).reshape(-1, 1)

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
            # F, which is the base function of the trend function
            one = np.hstack((np.ones((self.num_samples, 1)),
                            self.sample_scaled_x))
            # R^(-1)1
            beta = solve(L.T, solve(L, one))
            # 1R^(-1)Y / 1R^(-1)vector(1)
            # denominator = np.dot(one.T, beta)
            # cholseky decomposition is more stable
            low_denominator = cholesky(np.dot(one.T, beta), lower=True)
            mu = solve(low_denominator.T, solve(
                low_denominator, np.dot(one.T, alpha)))

            gamma = solve(L.T, solve(L, (self.sample_y - np.dot(one, mu))))

            sigma2 = np.dot((self.sample_y - np.dot(one, mu)).T,
                            gamma) / self.num_samples

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
        self.one = np.hstack((np.ones((self.num_samples, 1)),
                              self.sample_scaled_x))
        self.beta = solve(self.L.T, solve(self.L, self.one))
        self.low_denominator = cholesky(
            np.dot(self.one.T, self.beta), lower=True)
        self.mu = solve(self.low_denominator.T, solve(
            self.low_denominator, np.dot(self.one.T, self.alpha)))

        self.gamma = solve(self.L.T, solve(
            self.L, (self.sample_y - np.dot(self.one, self.mu))))
        self.sigma2 = (
            np.dot((self.sample_y - np.dot(self.one, self.mu)).T,
                   self.gamma) / self.num_samples
        ).item()
        self.logp = (-0.5 * self.num_samples * np.log(self.sigma2) -
                     np.sum(np.log(np.diag(self.L)))).item()
