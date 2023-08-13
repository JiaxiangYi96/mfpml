
from typing import Any

import numpy as np

from .sf_learning_functions import EFF, U


class EFFStoppingCriteria:

    def __init__(self) -> None:
        # define the learning function
        self.learning_function = EFF()

    def stopping_value(self, search_x: np.ndarray,
                       surrogate: Any,
                       iter: int,
                       **kwargs: Any) -> float:
        if iter == 0:
            stopping_value = np.abs(np.min(self.learning_function.eval(
                x=search_x,
                surrogate=surrogate)))
        else:
            # use learning function value as stopping value
            stopping_value = np.abs(kwargs["lf_value"])
        return stopping_value


class UStoppingCriteria:
    """use the negative value"""

    def __init__(self) -> None:
        self.learning_function = U()

    def stopping_value(self, search_x: np.ndarray,
                       surrogate: Any,
                       iter: int,
                       **kwargs: Any) -> Any | float:
        if iter == 0:
            stopping_value = np.min(self.learning_function.eval(
                x=search_x,
                surrogate=surrogate))
            stopping_value = -stopping_value
        else:
            # use learning function value as stopping value
            stopping_value = float(-kwargs["lf_value"])

        return stopping_value


class ESC:
    pass


class BSC:
    ...
