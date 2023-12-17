

import numpy as np


class Ordinary:
    """Ordinary basis function of Guassian process regression.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """ ordinary basis function

        Parameters
        ----------
        x : np.ndarray
            scaled design variables

        Returns
        -------
        np.ndarray
            a np array with shape (num_samples, 1), that contains the
            ordinary basis function values
        """

        num_samples = x.shape[0]

        return np.ones((num_samples, 1))


class Linear:
    """Linear basis function of Guassian process regression."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """ linear basis function

        Parameters
        ----------
        x : np.ndarray
            scaled design variables

        Returns
        -------
        np.ndarray
            a np array with shape (num_samples, num_dim + 1), that contains the
            linear basis function values
        """

        num_samples = x.shape[0]

        return np.hstack((np.ones((num_samples, 1)), x))


class Quadratic:
    """Quadratic basis function of Guassian process regression.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """ quadratic basis function

        Parameters
        ----------
        x : np.ndarray
            scaled design variables

        Returns
        -------
        np.ndarray
            a np array with shape (num_samples, num_col), that contains the
            quadratic basis function values
        """

        num_samples, num_dim = x.shape[0], x.shape[1]
        # number of columns of f
        num_col = int((num_dim + 1) * (num_dim + 2) / 2)

        # compute f
        f = np.zeros((num_samples, num_col))
        f[:, 0] = 1
        f[:, 1:(num_dim + 1)] = x
        j = num_dim + 1
        q = num_dim

        for k in range(num_dim):
            f[:, j:(j + q)] = np.tile(x[:, k], (q, 1)).T*x[:, k:num_dim]
            j = j + q
            q = q - 1

        return f
