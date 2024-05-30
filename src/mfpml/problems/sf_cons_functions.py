
from typing import Any, List

import numpy as np

from mfpml.problems.functions import Functions

# class LSQ(Functions):

class Branin(Functions):

    num_dim: int = 2
    num_obj: int = 1
    num_cons: int = 0
    input_domain = np.array([[-5.0, 10.0], [0.0, 15.0]])
    optimum: float = 0.397887
    optimum_scheme: List = [[-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475]]
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
        a = 1
        b = 5.1 / (4 * np.pi**2)
        c = 5.0 / np.pi
        d = 6.0
        h = 10.0
        ff = 1 / (8 * np.pi)
        x1 = x[:, 0]
        x2 = x[:, 1]

        obj = (
            a * (x2 - b * x1**2 + c * x1 - d) ** 2
            + h * (1 - ff) * np.cos(x1)
            + h
        )
        obj = np.reshape(obj, (x.shape[0], 1))
        return obj

    @staticmethod
    def f_cons(x: np.ndarray) -> np.ndarray:
        

