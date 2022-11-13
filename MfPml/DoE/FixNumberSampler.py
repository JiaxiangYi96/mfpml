import numpy as np
from dataclasses import dataclass
from MfPml.Core.Samplers import Sampler


@dataclass
class FixNumberSampler(Sampler):
    num_samples: int

    def get_samples(self, num_samples: int = None) -> np.ndarray:
        self.num_samples = num_samples
        fixedvalue = list(self.design_space.values())
        sample = np.repeat(fixedvalue[0], num_samples)
        self.samples = sample

        return self.samples
