import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from mfpml.design_of_experiment.multifidelity_samplers import MFSobolSequence
from mfpml.models.hierarchical_kriging import HierarchicalKriging
from mfpml.problems.multifidelity_functions import Forrester_1b

# define function
func = Forrester_1b()
# define sampler
sampler = MFSobolSequence(design_space=func._design_space, seed=1)
sample_x = sampler.get_samples(num_hf_samples=4, num_lf_samples=12)
sample_y = func(sample_x)

# generate test samples
test_x = np.linspace(0, 1, 1000).reshape(-1, 1)
test_hy = func.hf(test_x)
test_ly = func.lf(test_x)


def test_predict():
    # Create a CoKriging instance
    coK = HierarchicalKriging(design_space=func._input_domain)
    coK.train(sample_x, sample_y)

    # Make predictions
    cok_pre, cok_mse = coK.predict(test_x, return_std=True)

    # Assert that the predictions are correct
    assert cok_pre.shape == (1000, 1)
    assert cok_mse.shape == (1000, 1)
