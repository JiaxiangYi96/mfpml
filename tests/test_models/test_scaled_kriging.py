import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from mfpml.design_of_experiment.multifidelity_samplers import MFSobolSequence
from mfpml.models.scale_kriging import ScaledKriging
from mfpml.optimization.evolutionary_algorithms import PSO
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

pso_opt = PSO(num_gen=200, num_pop=20)


def test_predict():
    # Create a CoKriging instance
    coK = ScaledKriging(design_space=func._input_domain)
    coK.train(sample_x, sample_y)

    # Make predictions
    cok_pre, cok_mse = coK.predict(test_x, return_std=True)

    # Assert that the predictions are correct
    assert cok_pre.shape == (1000, 1)
    assert cok_mse.shape == (1000, 1)


def test_predict_with_rho():
    ScKrho = ScaledKriging(
        design_space=func._input_domain,
        rho_optimizer=True,
        rho_method="error",
        rho_bound=[0.0, 10.0],
        optimizer=pso_opt,
    )

    ScKrho.train(sample_x, sample_y)
    ScKrho_pred, ScKrho_std = ScKrho.predict(test_x, return_std=True)
    lf_pred, lf_std = ScKrho.lf_model.predict(test_x, return_std=True)
    # Assert that the predictions are correct
    assert ScKrho_pred.shape == (1000, 1)
    assert ScKrho_std.shape == (1000, 1)
    assert lf_pred.shape == (1000, 1)
    assert lf_std.shape == (1000, 1)


def test_predict_with_rho_bump():
    ScKrho = ScaledKriging(
        design_space=func._input_domain,
        rho_optimizer=True,
        rho_method="bumpiness",
        rho_bound=[0.0, 10.0],
        optimizer=pso_opt,
    )

    ScKrho.train(sample_x, sample_y)
    ScKrho_pred, ScKrho_std = ScKrho.predict(test_x, return_std=True)
    lf_pred, lf_std = ScKrho.lf_model.predict(test_x, return_std=True)
    # Assert that the predictions are correct
    assert ScKrho_pred.shape == (1000, 1)
    assert ScKrho_std.shape == (1000, 1)
    assert lf_pred.shape == (1000, 1)
    assert lf_std.shape == (1000, 1)
