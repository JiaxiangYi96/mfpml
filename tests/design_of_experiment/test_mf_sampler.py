import numpy as np
import pytest

from mfpml.design_of_experiment.multifidelity_samplers import (LatinHyperCube,
                                                               SobolSequence)
from mfpml.design_of_experiment.space import DesignSpace

pytestmark = pytest.mark.smoke


def test_sobolsequence_nested() -> None:
    space = DesignSpace(
        names=["x1", "x2"], low_bound=[0.0, 0.0], high_bound=[1.0, 1.0]
    )
    design_space = space.design_space
    mf_sampler = SobolSequence(design_space=design_space, seed=10, nested=True)
    samples = mf_sampler.get_samples(num_lf_samples=4, num_hf_samples=2)
    # test results
    results_hf = np.array([[0.17129549, 0.73985035], [0.64001456, 0.39570647]])
    results_lf = np.array(
        [
            [0.17129549, 0.73985035],
            [0.64001456, 0.39570647],
            [0.50126341, 0.54964036],
            [0.0327029, 0.33054989],
        ]
    )
    assert results_hf == pytest.approx(samples["hf"])
    assert results_lf == pytest.approx(samples["lf"])


def test_sobolsequence_unnested() -> None:
    space = DesignSpace(
        names=["x1", "x2"], low_bound=[0.0, 0.0], high_bound=[1.0, 1.0]
    )
    design_space = space.design_space
    mf_sampler = SobolSequence(
        design_space=design_space, seed=10, nested=False
    )
    samples = mf_sampler.get_samples(num_lf_samples=4, num_hf_samples=2)
    # test results
    results_hf = np.array([[0.84127845, 0.03307158], [0.04359769, 0.88905829]])
    results_lf = np.array(
        [
            [0.17129549, 0.73985035],
            [0.64001456, 0.39570647],
            [0.50126341, 0.54964036],
            [0.0327029, 0.33054989],
        ]
    )
    assert results_hf == pytest.approx(samples["hf"])
    assert results_lf == pytest.approx(samples["lf"])


def test_lhs_unnested() -> None:
    space = DesignSpace(
        names=["x1", "x2"], low_bound=[0.0, 0.0], high_bound=[1.0, 1.0]
    )
    design_space = space.design_space
    mf_sampler = LatinHyperCube(
        design_space=design_space, seed=12, nested=False
    )
    samples = mf_sampler.get_samples(num_lf_samples=4, num_hf_samples=2)
    # test results
    results_hf = np.array([[0.06760121, 0.57234874], [0.5944883, 0.36927682]])
    results_lf = np.array(
        [
            [0.43729389, 0.26331176],
            [0.7026699, 0.95517715],
            [0.91252769, 0.69236469],
            [0.08238856, 0.22123015],
        ]
    )
    assert results_hf == pytest.approx(samples["hf"])
    assert results_lf == pytest.approx(samples["lf"])
