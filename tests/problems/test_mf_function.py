import numpy as np
import pytest

from mfpml.design_of_experiment.mf_samplers import MFSobolSequence
from mfpml.problems.mf_functions import Forrester_1b

pytestmark = pytest.mark.smoke


def test_forrester_1b() -> None:
    function = Forrester_1b()
    design_space = function.input_domain
    # test sampling part
    sampler = MFSobolSequence(design_space=design_space,
                              num_fidelity=2, nested=True)
    samples = sampler.get_samples(num_samples=[4, 11])
    sample_y = function(samples)

    expected_results = [np.array([[0.76349455],
                                  [14.27057189],
                                  [-0.11428505],
                                  [0.14214545]]),
                        np.array([[-4.02316281],
                                  [6.83662705],
                                  [-7.38944566],
                                  [-5.87517176],
                                  [-1.30415088],
                                  [-4.96440515],
                                  [-9.24888108],
                                  [-8.49730964],
                                  [-5.53718063],
                                  [-4.67968572],
                                  [-5.02206314]])]
    # Assertions for all fidelity levels
    for i, expected in enumerate(expected_results):
        assert expected == pytest.approx(sample_y[i], rel=1e-5)
