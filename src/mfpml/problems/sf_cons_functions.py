
from typing import List

import numpy as np

from mfpml.problems.functions import Functions


class Branin_1(Functions):

    num_dim: int = 2
    num_obj: int = 1
    num_cons: int = 1
    input_domain = np.array([[0, 1], [0.0, 1]])
    optimum: float = 5.5757
    optimum_scheme: List = [0.96773, 0.20667]
    low_fidelity: List = None

    @classmethod
    def is_dim_compatible(cls, num_dim) -> int:
        assert (
            num_dim == cls.num_dim
        ), f"Can not change dimension for {cls.__name__} function"

        return num_dim

    def __init__(self, num_dim: int = 2) -> None:
        """
        Initialization
        """

        self.is_dim_compatible(num_dim=num_dim)

    @staticmethod
    def f(x: np.ndarray) -> np.ndarray:

        x1 = 15*x[:, 0]-5.0
        x2 = 15*x[:, 1]

        obj = (x2 - (5.1 / (4 * np.pi**2)) * x1**2 + (5 / np.pi) * x1 - 6)**2 \
            + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10 + 5 * x[:, 0]

        obj = np.reshape(obj, (len(obj), 1))

        return obj

    @staticmethod
    def f_cons(x: np.ndarray) -> np.ndarray:

        cons = -x[:, 0]*x[:, 1] + 0.2

        cons = np.reshape(cons, (len(cons), 1))
        return cons


class Branin_2(Functions):

    num_dim: int = 2
    num_obj: int = 1
    num_cons: int = 1
    input_domain = np.array([[0, 1], [0.0, 1]])
    optimum: float = 12.001
    optimum_scheme: List = [0.941, 0.317]
    low_fidelity: List = None

    @classmethod
    def is_dim_compatible(cls, num_dim) -> int:
        assert (
            num_dim == cls.num_dim
        ), f"Can not change dimension for {cls.__name__} function"

        return num_dim

    def __init__(self, num_dim: int = 2) -> None:
        """
        Initialization
        """

        self.is_dim_compatible(num_dim=num_dim)

    @staticmethod
    def f(x: np.ndarray) -> np.ndarray:

        x1 = 15*x[:, 0]-5.0
        x2 = 15*x[:, 1]

        obj = (x2 - (5.1 / (4 * np.pi**2)) * x1**2 + (5 / np.pi) * x1 - 6)**2 \
            + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10 + 5 * x[:, 0]

        obj = np.reshape(obj, (len(obj), 1))

        return obj

    @staticmethod
    def f_cons(x: np.ndarray) -> np.ndarray:

        x1 = 2 * x[:, 0] - 1
        x2 = 2 * x[:, 1] - 1
        cons = 6 - ((4 - 2.1 * x1**2 + x1**4 / 3) * x1**2 +
                    x1 * x2 + (-4 + 4 * x2**2) * x2**2
                    + 3 * np.sin(6 * (1 - x1)) + 3 * np.sin(6 * (1 - x2)))

        cons = np.reshape(cons, (len(cons), 1))
        return cons
