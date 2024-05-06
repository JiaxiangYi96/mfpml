from typing import Any

import matplotlib.pyplot as plt
import numpy as np


class SamplerVisualization:
    """Sampler visualization class
    """

    def __init__(self, sampler: Any ):

        self.sampler = sampler



    def plot_samples(self, num_samples: int, num_dim: int):
        pass 

    def _plot_sf_samples(self, samples: np.ndarray, num_dim: int):
        """plot samples

        Parameters
        ----------
        samples : np.ndarray
            samples to plot
        num_dim : int
            number of dimensions
        """

        if num_dim == 1:
            plt.scatter(samples, np.zeros_like(samples))
            plt.show()
        elif num_dim == 2:
            plt.scatter(samples[:, 0], samples[:, 1])
            plt.show()
        else:
            raise ValueError("Only support 1D or 2D visualization")
    
    def _plot_mf_samples(self, samples: np.ndarray, num_dim: int):
        """plot samples

        Parameters
        ----------
        samples : np.ndarray
            samples to plot
        num_dim : int
            number of dimensions
        """

        if num_dim == 1:
            plt.scatter(samples, np.zeros_like(samples))
            plt.show()
        elif num_dim == 2:
            plt.scatter(samples[:, 0], samples[:, 1])
            plt.show()
        else:
            raise ValueError("Only support 1D or 2D visualization")