from typing import Any

import numpy as np


class Ordinary:

    def __call__(self, x) -> np.ndarray:

        num_samples = x.shape[0]

        return np.ones((num_samples, 1))


class Linear:

    def __call__(self, x) -> np.ndarray:

        num_samples = x.shape[0]

        return np.hstack((np.ones((num_samples, 1)), x))


class Quadratic:

    def __call__(self, x) -> np.ndarray:

        num_dim = x.shape[1]

        # compute f
        f = np.zeros_like(x)
        f[:, 0] = 1
        f[:, 1:(num_dim + 1)] = x
        j = num_dim + 1
        q = num_dim

        for k in range(num_dim):
            f[:, j:(j + q)] = np.tile(x[:, k], (1, q)) * x[:, k:num_dim]
            j = j + q
            q = q - 1

        return f
