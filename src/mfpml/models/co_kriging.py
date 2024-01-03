from typing import Any, Tuple

import numpy as np
from numpy.linalg import cholesky, solve
from scipy.optimize import minimize

from .basis_functions import Ordinary
from .kernels import RBF


class Kriging_no_normalization:
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
        sample_new = np.atleast_2d(x_predict)
        # get the kernel matrix for predicted samples(scaled samples)
        knew = self.kernel.get_kernel_matrix(self.sample_scaled_x, sample_new)
        # calculate the predicted mean
        f = self.regr(sample_new)
        fmean = np.dot(f, self.beta) + np.dot(knew.T, self.gamma)
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
            std = np.sqrt(np.maximum(mse, 0)).reshape(-1, 1)

            return fmean, std

    def train(self, sample_x: np.ndarray, sample_y: np.ndarray) -> None:
        """training procedure of gpr models

        Parameters
        ----------
        sample_x : np.ndarray
            sample array of sample
        sample_y : np.ndarray
            responses of the sample
        """
        # get number samples
        self.num_samples = sample_x.shape[0]
        # the original sample_x
        self.sample_x = sample_x.copy()
        self.sample_scaled_x = sample_x.copy()
        # get the response
        self.sample_y = sample_y.reshape(-1, 1)
        self.sample_y_scaled = sample_y.reshape(-1, 1)

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
        """update the model with new samples

        Parameters
        ----------
        update_x : np.ndarray
            update sample array
        update_y : np.ndarray
            update responses
        """

        sample_x = np.concatenate((self.sample_x, update_x))
        sample_y = np.concatenate((self.sample_y, update_y))
        # update the model
        self.train(sample_x, sample_y)

    @property
    def _num_samples(self) -> int:
        """Return the number of samples

        Returns
        -------
        num_samples : int
            num samples
        """
        return self.sample_x.shape[0]


class CoKriging:
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
        self.lf_model = Kriging_no_normalization(
            design_space=design_space,
            optimizer=optimizer,
            optimizer_restart=optimizer_restart)

    def train(self, samples: dict, responses: dict) -> None:
        """Train the hierarchical Kriging model

        Parameters
        ----------
        samples : dict
            dict with two keys, 'hf' contains np.ndarray of
            high-fidelity sample points and 'lf' contains
            low-fidelity
        responses : dict
            dict with two keys, 'hf' contains high-fidelity
            responses and 'lf' contains low-fidelity ones
        """
        # load the samples and responses at the original scale
        self.sample_xh = samples["hf"]
        self.sample_yh = responses["hf"]
        self.sample_xl = samples["lf"]
        self.sample_yl = responses["lf"]

        # normalize the input and output
        self.sample_xh_scaled = self.normalize_input(self.sample_xh.copy())
        self.sample_xl_scaled = self.normalize_input(self.sample_xl.copy())
        self.sample_yh_scaled = self.normalize_hf_output(self.sample_yh.copy())
        self.sample_yl_scaled = (self.sample_yl - self.yh_mean) / self.yh_std

        # train the low-fidelity model
        self.lf_model.train(self.sample_xl_scaled,
                            self.sample_yl_scaled)

        # train the high-fidelity model
        # prediction of low-fidelity at high-fidelity locations
        self.pred_ylh = self.lf_model.predict(
            self.sample_xh_scaled, return_std=False)
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
        # oneC = np.ones((self.C.shape[0], 1))
        # calculate the covariance matrix
        c = np.concatenate(
            (
                self.rho
                * self.lf_model.sigma2
                * self.lf_model.kernel.get_kernel_matrix(
                    self.sample_xl_scaled, Xnew),
                self.rho**2
                * self.lf_model.sigma2*self.lf_model.kernel.get_kernel_matrix(
                    self.sample_xh_scaled, Xnew) + self.sigma2 *
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
                # + (1-oneC.T.dot(solve(self.LC.T, solve(self.LC, c)))) /
                # oneC.T.dot(solve(self.LC.T, solve(self.LC, oneC)))
            )
            std = np.sqrt(np.maximum(np.diag(s2), 0))*self.yh_std
            return fmean.reshape(-1, 1), std.reshape(-1, 1)

    def predict_lf(
        self, x_predict: np.ndarray, return_std: bool = False
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """Predict low-fidelity responses

        Parameters
        ----------
        x_predict : np.ndarray
            array of low-fidelity to be predicted
        return_std : bool, optional
            whether to return std values, by default False

        Returns
        -------
        np.ndarray
            prediction of low-fidelity
        """
        # get the scaled points
        x_predict = np.atleast_2d(self.normalize_input(x_predict))
        if not return_std:
            obj = self.lf_model.predict(x_predict)*self.yh_std + self.yh_mean
            return obj
        else:
            obj, std = self.lf_model.predict(x_predict, return_std=True)
            return obj*self.yh_std + self.yh_mean, std*self.yh_std

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

    def update_model(self, Xnew: dict, Ynew: dict) -> None:
        """Update the multi-fidelity model with new samples

        Parameters
        ----------
        Xnew : dict
            dict with two keys, where contains the new samples
            If value is None then no sample to update
        Ynew : dict
            corresponding responses w.r.t. Xnew
        """
        XHnew = Xnew["hf"]
        YHnew = Ynew["hf"]
        XLnew = Xnew["lf"]
        YLnew = Ynew["lf"]
        if XLnew is not None and YLnew is not None:
            if XHnew is not None and YHnew is not None:
                X = {}
                Y = {}
                X["hf"] = np.concatenate((self.sample_xh, XHnew))
                Y["hf"] = np.concatenate((self.sample_yh, YHnew))
                X["lf"] = np.concatenate((self.sample_xl, XLnew))
                Y["lf"] = np.concatenate((self.sample_yl, YLnew))
                self.train(X, Y)
            else:
                X = {}
                Y = {}
                X["hf"] = self.sample_xh
                Y["hf"] = self.sample_yh
                X["lf"] = np.concatenate((self.sample_xl, XLnew))
                Y["lf"] = np.concatenate((self.sample_yl, YLnew))
                self.train(X, Y)
        else:
            if XHnew is not None and YHnew is not None:
                XH = np.concatenate((self.sample_xh, XHnew))
                YH = np.concatenate((self.sample_yh, YHnew))
                self._train_hf(XH, YH)

    def _update_optimizer_hf(self, optimizer: Any) -> None:
        """Change the optimizer for high-fidelity hyper parameters

        Parameters
        ----------
        optimizer : any
            instance of optimizer
        """
        self.optimizer = optimizer

    def _update_optimizer_lf(self, optimizer: Any) -> None:
        """Change the optimizer for low-fidelity hyper parameters

        Parameters
        ----------
        optimizer : Any
            instance of optimizer
        """
        # update the optimizer for low-fidelity model
        self.lf_model.update_optimizer(optimizer)

    def normalize_input(self, inputs: np.ndarray) -> np.ndarray:
        """Normalize samples to range [0, 1]

        Parameters
        ----------
        inputs : np.ndarray
            samples to scale
        Returns
        -------
        np.ndarray
            normalized samples
        """
        return (inputs - self.bounds[:, 0]) / (
            self.bounds[:, 1] - self.bounds[:, 0]
        )

    def normalize_hf_output(self, outputs: np.ndarray) -> np.ndarray:
        """Normalize output to normal distribution

        Parameters
        ----------
        outputs : np.ndarray
            output to scale

        Returns
        -------
        np.ndarray
            normalized output
        """
        self.yh_mean = np.mean(outputs)
        self.yh_std = np.std(outputs)
        return (outputs - self.yh_mean) / self.yh_std

    @property
    def _get_lf_model(self) -> Any:
        """Get the low-fidelity model

        Returns
        -------
        Any
            low-fidelity model instance
        """

        return self.lf_model

    @ property
    def _num_xh(self) -> int:
        """Return the number of high-fidelity samples

        Returns
        -------
        int
            #high-fidelity samples
        """
        return self.sample_xh.shape[0]

    @ property
    def _num_xl(self) -> int:
        """Return the number of low-fidelity samples

        Returns
        -------
        int
            #low-fidelity samples
        """
        return self.lf_model._num_samples

    @ property
    def _get_sample_hf(self) -> np.ndarray:
        """Return samples of high-fidelity

        Returns
        -------
        np.ndarray
            high-fidelity samples
        """
        return self.sample_xh

    @ property
    def _get_sample_lf(self) -> np.ndarray:
        """Return samples of high-fidelity

        Returns
        -------
        np.ndarray
            high-fidelity samples
        """
        return self.sample_xl
