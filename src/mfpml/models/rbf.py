

import numpy as np
from numpy.linalg import cholesky, solve

from .kernels import RBF


class RBFSurrogate:

    def __init__(self,
                 design_space: np.ndarray,
                 ) -> None:
        # initialize parameters
        self.num_dim = design_space.shape[0]
        # bounds of design space
        self.bounds = design_space

        # set kernel
        self.kernel = RBF(theta=np.zeros(self.num_dim))
        self._set_kernel_params(params=np.ones(self.num_dim))

    def _set_kernel_params(self, params=None):
        self.kernel.set_params(params=params)

    def train(self, sample_x: np.ndarray, sample_y: np.ndarray) -> None:

        # get samples
        self.sample_x = sample_x
        self.sample_y = sample_y
        # regularization
        self.sample_x_scaled = self.normalize_input(sample_x=sample_x,
                                                    bounds=self.bounds)
        self.sample_y_scaled = self.normalize_output(sample_y=sample_y)
        # get kernel matrix
        self.K = self.kernel.get_kernel_matrix(self.sample_x_scaled,
                                               self.sample_x_scaled)
        # LU decomposition
        self.L = cholesky(self.K)

        # get weights
        self.W = solve(self.L.T, solve(self.L, self.sample_y_scaled))

    def predict(self, x_predict: np.ndarray):
        sample_new = self.normalize_input(x_predict, self.bounds)
        sample_new = np.atleast_2d(sample_new)

        # get the kernel matrix for predicted samples(scaled samples)
        knew = self.kernel.get_kernel_matrix(self.sample_x_scaled, sample_new)

        pred = np.dot(self.W.T, knew).reshape(-1, 1)
        # scale back
        pred = pred * self.y_std + self.y_mean

        return pred

    def normalize_output(self, sample_y: np.ndarray) -> np.ndarray:
        """Normalize samples to range [0, 1]

        Parameters
        ----------
        sample_y : np.ndarray
            samples to scale

        Returns
        -------
        np.ndarray
            normalized samples
        """
        self.y_mean = sample_y.mean()
        self.y_std = sample_y.std()

        return (sample_y - self.y_mean) / self.y_std

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


class NoiseRBFSurrogate:

    def __init__(self,
                 design_space: np.ndarray,
                 noise_std: float = 0.1) -> None:
        # initialize parameters
        self.num_dim = design_space.shape[0]
        # bounds of design space
        self.bounds = design_space

        # noise level
        self.noise_std = noise_std
        self.kernel = RBF(theta=np.zeros(self.num_dim))
        self._set_kernel_params(params=np.ones(self.num_dim))

    def _set_kernel_params(self, params=None):
        self.kernel.set_params(params=params)

    def train(self, sample_x: np.ndarray, sample_y: np.ndarray) -> None:

        # get samples
        self.sample_x = sample_x
        self.sample_y = sample_y
        # regularization
        self.sample_x_scaled = self.normalize_input(sample_x=sample_x,
                                                    bounds=self.bounds)
        self.sample_y_scaled = self.normalize_output(sample_y=sample_y)
        # get kernel matrix
        self.K = self.kernel.get_kernel_matrix(self.sample_x_scaled,
                                               self.sample_x_scaled) + \
            (self.noise_std/self.y_std)**2 * np.eye(self.sample_x.shape[0])
        # LU decomposition
        self.L = cholesky(self.K)

        # get weights
        self.W = solve(self.L.T, solve(self.L, self.sample_y_scaled))

    def predict(self, x_predict: np.ndarray):
        sample_new = self.normalize_input(x_predict, self.bounds)
        sample_new = np.atleast_2d(sample_new)

        # get the kernel matrix for predicted samples(scaled samples)
        knew = self.kernel.get_kernel_matrix(self.sample_x_scaled, sample_new)

        pred = np.dot(self.W.T, knew).reshape(-1, 1)
        # scale back
        pred = pred * self.y_std + self.y_mean
        return pred

    def normalize_output(self, sample_y) -> np.ndarray:
        """Normalize samples to normal distribution

        Parameters
        ----------
        sample_y : np.ndarray
            samples to scale

        Returns
        -------
        np.ndarray
            normalized samples
        """
        self.y_mean = sample_y.mean()
        self.y_std = sample_y.std()

        return (sample_y - self.y_mean) / self.y_std

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
