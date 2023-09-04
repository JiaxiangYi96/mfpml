from typing import Any

import numpy as np

from .sf_gpr import GP


class DeterministicGPR(GP):
    """Deterministic Gaussian Process Regression (GPR) model."""

    def __init__(self,
                 design_space: np.ndarray,
                 kernel_bound: list,
                 optimizer: Any,
                 regr:) -> None:
        ...
