import numpy as np
import pytest

from mfpml.design_of_experiment.singlefideliy_samplers import LatinHyperCube
from mfpml.problems.singlefidelity_functions import Branin

pytestmark = pytest.mark.smoke


# Continuous space tests


# @pytest.mark.smoke
def test_brainin_function_design_space() -> None:
    function = Branin()
    design_space = function.design_space

    results = {"x1": [-5.0, 10.0], "x2": [0.0, 15.0]}

    assert results == design_space


def test_brainin_function_values() -> None:
    function = Branin()
    design_space = function.design_space
    # test sampling part
    sampler = LatinHyperCube(design_space=design_space, seed=12)
    sample_x = sampler.get_samples(num_samples=3)
    sample_y = function.f(sample_x)
    results = np.array([[21.59234798], [10.03613571], [154.75696146]])

    assert results == pytest.approx(sample_y)
