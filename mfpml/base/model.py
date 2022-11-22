import numpy as np
from typing import Tuple, Any


class GPRmodel:
    """
    Interface of GPR models
    """

    def train_model(self, x: np.ndarray, y: np.ndarray, num_dim: int) -> Any:
        pass

    def predict(self, test_x: np.ndarray, return_std: bool = False) -> Tuple[Any, Any] | Any:
        pass
