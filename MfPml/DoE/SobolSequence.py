import numpy as np
from dataclasses import dataclass
from MfPml.Core.Samplers import Sampler


@dataclass
class SobolSequence(Sampler):
    """
    Sobol Sequence sampling
    """
    num_samples: int

    def get_samples(self, num_samples: int = None) -> np.ndarray:
        self.num_samples = num_samples
        sample = sobol_sequence.sample(self.num_samples, self.num_dim)
        for i, bounds in enumerate(self.design_space.values()):
            sample[:, i] = sample[:, i] * (bounds[1] - bounds[0]) + bounds[0]

        self.samples = sample

        return self.samples
