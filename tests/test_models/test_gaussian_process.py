# third party packages
import matplotlib.pyplot as plt
import numpy as np

from mfpml.design_of_experiment.sf_samplers import LatinHyperCube
from mfpml.models.basis_functions import Linear, Ordinary, Quadratic
from mfpml.models.gaussian_process import GaussianProcess
from mfpml.models.kernels import RBF
# local funstions
from mfpml.models.kriging import Kriging
from mfpml.optimization.evolutionary_algorithms import DE
from mfpml.problems.sf_functions import Forrester

# define function
func = Forrester()

# initialize sampler

sample_x = np.array([0.0, 0.4, 0.6, 1.0]).reshape((-1, 1))
test_x = np.linspace(0, 1, 10000001, endpoint=True).reshape(-1, 1)

# get the function value
sample_y = func.f(sample_x)
test_y = func.f(test_x)


def test_kriging():
    # Create a Kriging instance
    krig = Kriging(design_space=func._input_domain)
    krig.train(sample_x, sample_y)

    # Make predictions
    krig_pre, krig_mse = krig.predict(test_x, return_std=True)

    # Assert that the predictions are correct
    assert krig_pre.shape == (10000001, 1)
    assert krig_mse.shape == (10000001, 1)


def test_gpr():
    # sampling by sampling method
    sampler = LatinHyperCube(design_space=func._design_space, seed=1)
    sample_x = sampler.get_samples(num_samples=30)
    test_x = np.linspace(0, 1, 101, endpoint=True).reshape(-1, 1)

    # get samples by adding noise to the true function
    sample_y = func.f(sample_x) + np.random.normal(0, 0.2,
                                                   sample_x.shape[0]).reshape((-1, 1))

    # initialize optimizer
    optimizer = DE(num_gen=1000, num_pop=50, crossover_rate=0.5,
                   strategy="DE/best/1/bin")
    # initialize the regressor
    gp_model = GaussianProcess(
        design_space=func._input_domain,
        optimizer=optimizer)
    # train the model
    gp_model.train(sample_x, sample_y)
    # get the prediction
    sf_pre, sf_std = gp_model.predict(test_x, return_std=True)

    # assert the prediction is correct
    assert sf_pre.shape == (101, 1)
    assert sf_std.shape == (101, 1)
    # assert gp_model.noise is float
    assert isinstance(gp_model.noise, float)


def test_gpt_none_optimizer():
    # sampling by sampling method
    sampler = LatinHyperCube(design_space=func._design_space, seed=1)
    sample_x = sampler.get_samples(num_samples=30)
    test_x = np.linspace(0, 1, 1000000, endpoint=True).reshape(-1, 1)

    # get samples by adding noise to the true function
    sample_y = func.f(sample_x) + np.random.normal(0, 0.2,
                                                   sample_x.shape[0]).reshape((-1, 1))

    # initialize the regressor
    gp_model = GaussianProcess(
        design_space=func._input_domain,
        optimizer=None,
        optimizer_restart=5)
    # train the model
    gp_model.train(sample_x, sample_y)
    # get the prediction
    sf_pre, sf_std = gp_model.predict(test_x, return_std=True)

    # assert the prediction is correct
    assert sf_pre.shape == (1000000, 1)
    assert sf_std.shape == (1000000, 1)
    # assert gp_model.noise is float
    assert isinstance(gp_model.noise, float)
