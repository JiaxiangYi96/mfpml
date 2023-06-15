import numpy as np
import pytest

from mfpml.design_of_experiment.mf_samplers import SobolSequence
from mfpml.problems.multifidelity_functions import Forrester_1b

pytestmark = pytest.mark.smoke


def test_forrester_1b() -> None:
    function = Forrester_1b()
    design_space = function.design_space
    # test sampling part
    sampler = SobolSequence(design_space=design_space, seed=12, nested=True)
    samples = sampler.get_samples(num_lf_samples=10, num_hf_samples=4)
    sample_y = {}
    sample_y["hf"] = function.hf(samples["hf"])
    sample_y["lf"] = function.lf(samples["lf"])
    results = {
        "hf": np.array(
            [
                [-4.49853915e00],
                [-5.12923283e00],
                [1.98789697e-01],
                [-5.65367131e-04],
            ]
        ),
        "lf": np.array(
            [
                [-4.1700473],
                [-5.43372262],
                [-5.75155027],
                [-6.77643713],
                [-4.03765842],
                [6.42069805],
                [-8.77875965],
                [-8.05337556],
                [3.42526783],
                [-3.99849013],
            ]
        ),
    }

    assert results["lf"] == pytest.approx(sample_y["lf"])
    assert results["hf"] == pytest.approx(sample_y["hf"])


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
