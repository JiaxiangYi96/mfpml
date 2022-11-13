import numpy as np
from scipy.stats import qmc
from dataclasses import dataclass
from MfPml.Core.Samplers import Sampler


@dataclass
class LatinHyperCube(Sampler):
    """
    Latin Hyper cube sampling
    """
    num_samples: int

    def get_samples(self, num_samples: int = None) -> np.ndarray:
        self.num_samples = num_samples
        sampler = qmc.LatinHypercube(d=self.num_dim)
        sample = sampler.random(self.num_samples)
        for i, bounds in enumerate(self.design_space.values()):
            sample[:, i] = sample[:, i] * (bounds[1] - bounds[0]) + bounds[0]

        self.samples = sample

        return self.samples
