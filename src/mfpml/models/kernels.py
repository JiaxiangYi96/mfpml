import numpy as np
from scipy.spatial.distance import cdist


class KernelCore:
    """
    Base class for kernel
    """

    @property
    def hyperparameters(self) -> list:
        """
        Returns a list of all hyper_parameters specifications
        """
        return self.param.tolist()

    @property
    def _get_num_para(self) -> int:
        """Return number of parameters of the kernel

        Returns
        -------
        int
            number of parameters
        """
        return self.num_para

    @property
    def _get_bounds(self) -> np.ndarray:
        """Get the parameters' bounds

        Returns
        -------
        np.ndarray
            design bounds of the parameters
        """
        return self.bounds

    @property
    def _get_bounds_list(self) -> list:
        """Get the parameters' bounds with list

        Returns
        -------
        list
            design bounds of the parameters
        """
        return self.bounds.tolist()

    @property
    def _get_low_bound(self) -> list:
        """Get the low bound of the parameters

        Returns
        -------
        list
            low bound of kernel's parameters
        """
        return self.bounds[:, 0].tolist()

    @property
    def _get_high_bound(self) -> list:
        """Get the high bound of the parameters

        Returns
        -------
        list
            high bound of kernel's parameters
        """
        return self.bounds[:, 1].tolist()

    @property
    def _get_param(self) -> np.ndarray:
        """Get the parameters of the corelation

        Returns
        -------
        np.ndarray
            parameters
        """
        return self.param

# =========================================================================== #
#                                 Kernel   RBF                                #
# =========================================================================== #


class RBF(KernelCore):
    def __init__(
        self,
        theta: np.ndarray,
        parameters: list = ["theta"],
        bounds: list = [-2, 3],
    ) -> None:
        """calculate the rbf kernel

        Parameters
        ----------
        theta : np.ndarray, shape=((1,num_dims))
            initial guess of the parameters
        parameters : list, optional
            list the parameters to fit, by default ['theta']
        bounds : list, optional
            log bounds of the parameters, by default [-4, 3]
        """
        self.param = 10**theta
        self.parameters = parameters
        self.bounds = []
        self.num_para = theta.size
        for _ in range(self.num_para):
            self.bounds.append(bounds)
        self.bounds = np.array(self.bounds)

    def get_kernel_matrix(self, X, Y) -> np.ndarray:
        """
        Return the kernel matrix k(x, y).

        Parameters
        ----------
        X : np.ndarray
            array of the first samples shape=((n1, num_dims))
        Y : np.ndarray
            array of the second samples shape=((n2, num_dims))

        Returns
        -------
        np.ndarray
            kernel matrix with shape ((n1, n2))
        """
        # deal with parameters
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        X_sq = np.sum(X**2 * self.param, axis=1, keepdims=True)
        Y_sq = np.sum(Y**2 * self.param, axis=1)
        XY = XY = np.dot(np.sqrt(self.param) * X, (np.sqrt(self.param) * Y).T)
        dist = X_sq + Y_sq - 2 * XY

        return np.exp(-dist) + np.eye(X.shape[0], Y.shape[0]) * 10 ** -10

    def __call__(self,
                 X: np.ndarray,
                 Y: np.ndarray,
                 param: np.ndarray,
                 ) -> np.ndarray:
        """
        Return the kernel k(x, y) and optionally its gradient.

        Parameters
        ----------
        X : np.ndarray
            array of the first samples shape=((n1, num_dims))
        Y : np.ndarray
            array of the second samples shape=((n2, num_dims))
        param : np.array
            parameters in the specific kernel, shape=((1, num_params))
        Returns
        -------
        np.array
            kernel values with respect to parameters param
        """

        # deal with parameters
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        # deal with parameters
        param = 10**param
        # compute the distance
        X_sq = np.sum(X**2 * param, axis=1, keepdims=True)
        Y_sq = np.sum(Y**2 * param, axis=1)
        XY = np.dot(np.sqrt(param) * X, (np.sqrt(param) * Y).T)
        dist = X_sq + Y_sq - 2 * XY

        return np.exp(-dist) + np.eye(X.shape[0], Y.shape[0]) * 10 ** -10

    def set_params(self, params) -> None:
        """
        Set the parameters of the kernel.
        """
        setattr(self, "param", 10**params)

# =========================================================================== #
#                                 Kernel   Matern                             #
# =========================================================================== #


class Matern(KernelCore):
    ...


# =========================================================================== #
def kronDelta(X, Y):
    """
    Computes Kronecker delta for rows in x and y.

    Parameters
    ----------
    x, y: np.ndarray, shape=((n, nfeatures))

    Returns
    -------
    np.ndarray
        Kronecker delta between row pairs of `X` and `Xstar`.
    """
    return cdist(X, Y) < np.finfo(np.float32).eps
