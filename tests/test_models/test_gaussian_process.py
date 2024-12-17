
import numpy as np

from mfpml.design_of_experiment.sf_samplers import LatinHyperCube

from mfpml.models.gaussian_process import GaussianProcessRegression

from mfpml.optimization.evolutionary_algorithms import DE
from mfpml.problems.sf_functions import Forrester

# define function
func = Forrester()

# initialize sampler

sample_x = np.array([0.0, 0.4, 0.6, 1.0]).reshape((-1, 1))
test_x = np.linspace(0, 1, 10001, endpoint=True).reshape(-1, 1)

# get the function value
sample_y = func.f(sample_x)
test_y = func.f(test_x)


def test_kriging():
    # Create a Kriging instance
    krig = GaussianProcessRegression(design_space=func.input_domain)
    krig.train(sample_x, sample_y)

    # Make predictions
    krig_pre, krig_mse = krig.predict(test_x, return_std=True)

    # Assert that the predictions are correct
    assert krig_pre.shape == (10001, 1)
    assert krig_mse.shape == (10001, 1)
