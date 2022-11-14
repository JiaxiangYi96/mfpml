import numpy as np
from MfPml.Core.Functions import Functions


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
    low_bound: list = [0.0]
    high_bound: list = [1.0]
    design_space: dict = {'x': [0.0, 1.0]}
    optimum: float = -6.020740
    optimum_scheme: list = [0.757248757841856]
    low_fidelity: list = ['low_a', 'low_b', 'low_c']

    def __init__(self, cost=[1.0, 1.0]) -> None:
        """
        Parameters 
        ----------
        cost: list 
            Cost coefficient for high- and low-fidelity respectively 
        """
        self.fidelity = None
        self.cost = cost

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
        y = self._evaluate(x=x)

        return y

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        """

        Parameters
        ----------
        x: np.ndarray
        design schemes that should be evaluated

        Returns
        -------
        y: np.ndarray
        Responses of Forrester function

        """

        if self.fidelity == 'high':
            y = (6 * x) ** 2 * np.sin(12 * x - 4)
        elif self.fidelity == 'low_a':
            y = (6 * x) ** 2 * np.sin(12 * x - 4) - 5
        elif self.fidelity == 'low_b':
            y = 0.5 * ((6 * x) ** 2 * np.sin(12 * x - 4)) + 10 * (x - 0.5) - 5
        elif self.fidelity == 'low_c':
            y = (6 * (x + 0.2) - 2) ** 2 * np.sin(12 * (x + 0.2) - 4)
        else:
            print("Error!!! Please input the right!!! \n")

        return y

class
