import numpy as np
import pytest

from mfpml.design_of_experiment.sf_samplers import LatinHyperCube
from mfpml.problems.sf_functions import Branin

pytestmark = pytest.mark.smoke


# Continuous space tests


# @pytest.mark.smoke
def test_brainin_function_design_space() -> None:
    function = Branin()
    design_space = function.input_domain

    results = np.array([[-5.0, 10.0], [0.0, 15.0]])

    assert results.shape == design_space.shape


def test_brainin_function_values() -> None:
    function = Branin()
    design_space = function.input_domain
    # test sampling part
    sampler = LatinHyperCube(design_space=design_space)
    sample_x = sampler.get_samples(num_samples=3, seed=12)
    sample_y = function.f(sample_x)
    results = np.array([[21.59234798], [10.03613571], [154.75696146]])

    assert results == pytest.approx(sample_y)
