from typing import Any

import numpy as np
from scipy.optimize import minimize

from .gpr_base import MultiFidelityGP
from .kernels import RBF
from .kriging import Kriging


class ScaledKriging(MultiFidelityGP):
    def __init__(
        self,
        design_space: np.ndarray,
        lf_model: Any = None,
        disc_model: Any = None,
        optimizer: Any = None,
        optimizer_restart: int = 0,
        kernel_bound: list = [-4.0, 3.0],
        rho_optimize: bool = False,
        rho_method: str = "error",
        rho_bound: list = [1e-2, 1e1],
        rho_optimizer: Any = None,
    ) -> None:
        """Multi-fidelity Kriging model with scaled function

        Parameters
        ----------
        design_space : np.ndarray
            array of shape=((num_dim,2)) where each row describes
            the bound for the variable on specfic dimension
        lf_model: any
            instance of low-fidelity model, the model should have the method:
            train(x: np.ndarray, y: np.ndarray),
            predict(x: np.ndarray, return_std: bool)
        disc_model: any, optional
            instance of discrepancy model, the model should have the method:
            train(x: np.ndarray, y: np.ndarray),
            predict(x: np.ndarray, return_std: bool). Default Kriging
        optimizer: any, optional
            instance of the optimizer used to optimize the hyperparameters
            with the use style optimizer.run_optimizer(objective function,
            number of dimension, design space of variables), if not assigned,
            the 'L-BFGS-B' method in scipy is used
        kernel_bound: list, optional
            log bound of the kernel for discrepancy Kriging model, by
            default [-3, 2]
        rho_optimize : bool, optional
            whether to optimize the scale factor, if not the scale
            factor is 1, by default False
        rho_method : str, optional
            method to choose rho, can choose from ['error', 'bumpiness']
        rho_bound : list, optional
            bound for the factor rho if optimizing rho, by default [1e-1, 1e1]
        rho_optimizer : any, optional
            optimizer for the parameter rho, by default 'L-BFGS-B'
        """
        self.bounds = design_space
        self.lf_model = lf_model
        self.optimizer = optimizer
        self.optimizer_restart = optimizer_restart
        self.rho_optimize = rho_optimize
        self.rho = 1.0
        self.rho_bound = rho_bound
        self.rho_method = rho_method
        self.rho_optimizer = rho_optimizer
        self.mu = 1.0
        self.num_dim = design_space.shape[0]
        self.kernel = RBF(theta=np.zeros(self.num_dim), bounds=kernel_bound)
        if lf_model is None:
            self.lf_model = Kriging(
                design_space=design_space,
                optimizer=optimizer,
                optimizer_restart=optimizer_restart
            )
        else:
            self.lf_model = lf_model
        if disc_model is None:
            self.disc_model = Kriging(
                design_space=design_space,
                optimizer=optimizer,
                optimizer_restart=optimizer_restart,
            )
        else:
            self.disc_model = disc_model
        if optimizer is not None:
            self._update_optimizer_hf(optimizer)
            self._update_optimizer_lf(optimizer)

    def _train_hf(self, sample_xh: np.ndarray, sample_yh: np.ndarray) -> None:
        """Train the discrepancy model in mf models

        Parameters:
        -----------------
        sample_xh : np.ndarray
            array of high-fidelity samples
        sample_yh : np.ndarray
            array of high-fidelity responses
        """
        self.sample_xh = sample_xh
        self.sample_yh = sample_yh.reshape(-1, 1)
        if self.rho_optimize:
            self._rho_optimize()
        self.disc_model.train(self.sample_xh, self._getDisc())

    def predict(
        self, x_predict: np.ndarray, return_std: bool = False
    ) -> Any:
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
        if not return_std:
            return self.disc_model.predict(x_predict) * self.rho \
                + self.lf_model.predict(x_predict)
        else:
            pre_lf, std_lf = self.lf_model.predict(x_predict, return_std)
            pre_disc, std_disc = self.disc_model.predict(x_predict, return_std)
            mse = self.rho**2 * std_lf**2 + std_disc**2
            return self.rho * pre_lf + pre_disc, np.sqrt(mse)

    def _getDisc(self) -> np.ndarray:
        """Compute the discrepancy between low-fidelity prediction
        at high-fidelity samples and high-fidelity responses

        Returns
        -------
        np.ndarray
            discrepancy
        """
        return self.sample_yh - self.rho * self.predict_lf(self.sample_xh)

    def _update_optimizer_hf(self, optimizer: Any) -> None:
        """Change the optimizer for high-fidelity hyperparameters

        Parameters
        ----------
        optimizer : any
            instance of optimizer
        """
        self.disc_model.optimizer = optimizer

    def _rho_optimize(self) -> None:
        """Optimize the rho value"""
        if self.rho_optimizer is None:
            if self.rho_method == "error":
                x0 = np.random.uniform(self.rho_bound[0], self.rho_bound[1], 1)
                optRes = minimize(
                    self._eval_error,
                    x0=x0,
                    method="L-BFGS-B",
                    bounds=np.array([self.rho_bound]),
                )
            elif self.rho_method == "bumpiness":
                x0 = np.random.uniform(self.rho_bound[0], self.rho_bound[1], 1)
                optRes = minimize(
                    self._eval_bumpiness,
                    x0=x0,
                    method="L-BFGS-B",
                    bounds=np.array([self.rho_bound]),
                )
            else:
                pass
            self.rho = optRes.x
        else:
            if self.rho_method == "error":
                optRes, _, _ = self.rho_optimizer.run_optimizer(
                    self._eval_error,
                    num_dim=1,
                    design_space=np.array([self.rho_bound]),
                )
                self.rho_optimizer.plot_optimization_history()
            elif self.rho_method == "bumpiness":
                optRes, _, _ = self.rho_optimizer.run_optimizer(
                    self._eval_bumpiness,
                    num_dim=1,
                    design_space=np.array([self.rho_bound]),
                )
                self.rho_optimizer.plot_optimization_history()
            else:
                pass
            self.rho = optRes["best_x"]

    def _eval_error(self, rho: np.ndarray) -> np.ndarray:
        """Evaluate the summation of squared error for high-fidelity samples

        Parameters
        ----------
        rho : np.ndarray
            array of rho

        Returns
        -------
        np.ndarray
            sum of error
        """
        rho = np.tile(rho.reshape(-1, 1), (1, self._num_xh))
        error = (rho * self.predict_lf(self.sample_xh).ravel() -
                 self.sample_yh.ravel())
        sum_error2 = np.sum(error**2, axis=1)
        return sum_error2

    def _eval_bumpiness(self, rho: np.ndarray) -> np.ndarray:
        """Evaluate the bumpiness

        Parameters
        ----------
        rho : np.ndarray
            array of rho

        Returns
        -------
        np.ndarray
            measure of bumpiness
        """
        rho = rho.reshape(-1, 1)
        out = np.zeros(rho.shape[0])
        for i in range(rho.shape[0]):
            sum_error2 = self._eval_error(rho[i, :])
            self.disc_model.train(
                self.sample_xh,
                self.sample_yh - rho[i, :] * self.predict_lf(self.sample_xh),
            )
            theta = self.disc_model.kernel._get_param
            out[i] = sum_error2 * np.linalg.norm(theta)
        return out
