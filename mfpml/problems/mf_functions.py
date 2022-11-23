import numpy as np
from mfpml.base.functions import Functions


class Forrester(Functions):
    """
    Forrester function
    This file contains the definition of an adapted version of the simple 1D
    example function as presented in:
        Forrester Alexander I.J, Sóbester András and Keane Andy J "Multi-fidelity
        Optimization via Surrogate Modelling", Proceedings of the Royal Society A,
        vol. 463, http://doi.org/10.1098/rspa.2007.1900

    Function definitions:
    .. math::
        f_h(x) = (6x-2)^2 \sin(12x-4)
    For the low fidelity functions, they are different among different publications, in
    this code, the version adopted from :
        Jiang, P., Cheng, J., Zhou, Q., Shu, L., & Hu, J. (2019). Variable-fidelity lower
        confidence bounding approach for engineering optimization problems with expensive
        simulations. AIAA Journal, 57(12), 5416-5430.
    """

    num_dim: int = 1
    num_obj: int = 1
    num_cons: int = 0
    domain: list = [0.0]
    high_bound: list = [1.0]
    design_space: dict = {'x': [0.0, 1.0]}
    optimum: float = -6.020740
    optimum_scheme: list = [0.757248757841856]
    low_fidelity: list = ['low_a', 'low_b', 'low_c']
    cost_ratio: float = None

    def __init__(self, num_dim: int = 1) -> None:
        """
        Initialization of Forrester function
        Parameters
        ----------
        num_dim
        """

        self.fidelity = None

        if self.__class__.num_dim is None:
            self.__class__.num_dim = num_dim

    def __call__(self, x: np.ndarray, fidelity: str = 'high') -> np.ndarray:
        """

        Parameters
        ----------
        x: np.ndarray
            design variables need to be evaluated
        fidelity: str
            string indicating the

        Returns
        -------
        y: np.ndarray
            outputs of forrester function

        """
        self.fidelity = fidelity
        obj = self._evaluate(x=x)
        obj.reshape((x.shape[0], 1))

        return obj

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        """

        Parameters
        ----------
        x: np.ndarray
        design schemes that should be evaluated

        Returns
        -------
        obj: np.ndarray
        Responses of Forrester function

        """

        if self.fidelity == 'high':

            obj = (6 * x - 2) ** 2 * np.sin(12 * x - 4)
        elif self.fidelity == 'low_a':
            obj = (6 * x - 2) ** 2 * np.sin(12 * x - 4) - 5
        elif self.fidelity == 'low_b':
            obj = 0.5 * ((6 * x - 2) ** 2 * np.sin(12 * x - 4)) + 10 * (x - 0.5) - 5
        elif self.fidelity == 'low_c':
            obj = (6 * (x + 0.2) - 2) ** 2 * np.sin(12 * (x + 0.2) - 4)
        else:
            raise KeyError("Not defined fidelity!!! \n")

        return obj


class mf_Hartman3(Functions): 
    """_summary_

    Parameters
    ----------
    Functions : _type_
        _description_
    """
    num_dim: int = 3
    num_obj: int = 1
    num_cons: int = 0
    low_bound: list = [0.0, 0.0, 0.0]
    high_bound: list = [1.0, 1.0, 1.0]
    input_domain = np.array([[0.0, 1.0],
                             [0.0, 1.0],
                             [0.0, 1.0]])
    design_space: dict = {'x1': [0.0, 1.0],
                          'x2': [0.0, 1.0],
                          'x3': [0.0, 1.0]}
    optimum: float = -3.86278214782076
    optimum_scheme: list = [0.1, 0.55592003, 0.85218259]
    low_fidelity: list = ['low'] 

    def __call__(self, x: dict) -> dict: 
        out = {}
        out['hf'] = self.hf(x['hf'])
        out['lf'] = self.lf(x['lf'])
        return out

    @staticmethod 
    def hf(x: np.array) -> np.ndarray: 
        a = np.array([[3.0, 10.0, 30.0],
                      [0.1, 10.0, 35.0],
                      [3.0, 10.0, 30.0],
                      [0.1, 10.0, 35.0]])
        p = np.array([[0.3689, 0.117, 0.2673],
                      [0.4699, 0.4387, 0.747],
                      [0.1091, 0.8732, 0.5547],
                      [0.03815, 0.5743, 0.8828]])
        c = np.array([1, 1.2, 3, 3.2])
        num_samples = x.shape[0]
        obj = np.zeros((num_samples, 1))
        for i in range(num_samples):
            obj[i, :] = -np.dot(c, np.exp(-np.sum(a * (x[i, :] - p) ** 2, axis=1)))
        obj.reshape((x.shape[0], 1))
        return obj 

    @staticmethod
    def lf(x: np.array) -> np.ndarray:
        obj = 0.585 - 0.324 * x[:, 0] - 0.379 * x[:, 1] - 0.431 * x[:, 2] \
            - 0.208 * x[:, 0] * x[:, 1] + 0.326 * x[:, 0] * x[:, 2] \
            + 0.193 * x[:, 1] * x[:, 2] + 0.225 * x[:, 0] ** 2 \
            + 0.263 * x[:, 1] ** 2 + 0.274 * x[:, 2] ** 2
        return obj.reshape(-1, 1)

