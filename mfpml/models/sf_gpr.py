from typing import Tuple, Any

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from mfpml.base.model import GPRmodel


class SklearnGPRmodel(GPRmodel):
    """Class for establish gaussian process models using sklearn package
    """

    def __init__(self) -> None:
        """
        Initialization of GPR models
        """

        self.kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-5, 1e5)) \
                      * RBF(length_scale=1.0, length_scale_bounds=(1e-20, 1e5))
        self.n_restarts_optimizer = 50
        self.model = None
        self.num_dim = None
        self.x = None
        self.y = None
        self.y_pred = None
        self.y_sigma = None

    def train_model(self, x: np.ndarray, y: np.ndarray, num_dim: int) -> any:
        """
        fit the models via given data

        Parameters
        ----------
        num_dim : int
            dimension of the models
        x: np.ndarray
            inputs
        y: np.ndarray
            outputs

        Returns
        -------
        models: Any
            GPR models

        """
        self.num_dim = num_dim
        self.x = x.reshape((-1, num_dim))
        self.y = y.reshape((-1, 1))
        self.model = GaussianProcessRegressor(kernel=self.kernel,
                                              n_restarts_optimizer=self.n_restarts_optimizer)

        # Fit to data using Maximum Likelihood Estimation of the parameters
        self.model.fit(X=self.x, y=self.y)

        return self.model

    def predict(self, test_x: np.ndarray, return_std: bool = True) -> tuple[Any, Any] | Any:
        """

        Parameters
        ----------
        test_x: np.ndarray
            test design schemes
        return_std: bool
            return predicted uncertainty or not

        Returns
        -------
        y_pred:np.ndarray
            predicted mean value of the tested schemes
        y_sigma: np.ndarray
            predicted uncertainty of the tested design schemes


        """

        if return_std:
            self.y_pred, self.y_sigma = self.model.predict(test_x, return_std=True)
            return self.y_pred, self.y_sigma
        else:
            self.y_pred = self.model.predict(X=test_x, return_std=False)
            return self.y_pred
