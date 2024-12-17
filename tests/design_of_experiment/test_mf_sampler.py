import numpy as np
import pytest
from mfpml.design_of_experiment.mf_samplers import (
    MFLatinHyperCube, MFSobolSequence
)

pytestmark = pytest.mark.smoke

@pytest.fixture
def design_space():
    """Fixture for design space."""
    return np.array([[5., 10], [1, 2]])


@pytest.fixture
def sample_sizes():
    """Fixture for sample sizes."""
    return [2, 5]


def validate_samples(mf_sampler,
                     sample_sizes,
                     seed,
                     expected_results):
    """
    Helper function to validate sampler outputs.
    """
    data = mf_sampler.get_samples(num_samples=sample_sizes, seed=seed)
    for i, result in enumerate(expected_results):
        assert result == pytest.approx(data[i])


def test_mf_lhs_nested(design_space,
                       sample_sizes):
    """Test nested multi-fidelity Latin Hypercube sampling."""
    mf_sampler = MFLatinHyperCube(design_space=design_space,
                                  num_fidelity=2,
                                  nested=True)
    # fix seed 
    np.random.seed(123456)
    expected_results = [np.array([[7.21168694, 1.40160066],
                                  [8.666279, 1.14971204]]),
                        np.array([[7.21168694, 1.40160066],
                                  [6.0414317, 1.2244589],
                                  [9.33874143, 1.77814134],
                                  [5.04239678, 1.91309308],
                                  [8.666279, 1.14971204]])]

    validate_samples(mf_sampler,
                     sample_sizes, seed=123456,
                     expected_results=expected_results)


def test_mf_lhs_unnested(design_space, sample_sizes):
    """Test unnested multi-fidelity Latin Hypercube sampling."""
    mf_sampler = MFLatinHyperCube(
        design_space=design_space, num_fidelity=2, nested=False)
    expected_results = [
        np.array([[8.40871563, 1.30759417], [7.38138645, 1.52237363]]),
        np.array([[7.21168694, 1.40160066], [6.0414317, 1.2244589],
                  [9.33874143, 1.77814134], [5.04239678, 1.91309308],
                  [8.666279, 1.14971204]])
    ]
    validate_samples(mf_sampler, sample_sizes, seed=123456,
                     expected_results=expected_results)


def test_sobol_unnested(design_space, sample_sizes):
    """Test unnested Sobol Sequence sampling."""
    mf_sampler = MFSobolSequence(
        design_space=design_space, num_fidelity=2, nested=False)
    expected_results = [
        np.array([[8.2381979, 1.28450888], [6.83417066, 1.68135563]]),
        np.array([[8.2381979, 1.28450888], [6.83417066, 1.68135563],
                  [7.40140805, 1.47841079], [7.55499782, 1.61233338],
                  [9.61076839, 1.08658168]])
    ]
    validate_samples(mf_sampler, sample_sizes, seed=123456,
                     expected_results=expected_results)


def test_sobol_nested(design_space, sample_sizes):
    """Test nested Sobol Sequence sampling."""
    mf_sampler = MFSobolSequence(
        design_space=design_space, num_fidelity=2, nested=True)
    expected_results = [
        np.array([[8.2381979, 1.28450888], [6.83417066, 1.68135563]]),
        np.array([[8.2381979, 1.28450888], [6.83417066, 1.68135563],
                  [7.40140805, 1.47841079], [7.55499782, 1.61233338],
                  [9.61076839, 1.08658168]])
    ]
    validate_samples(mf_sampler, sample_sizes, seed=123456,
                     expected_results=expected_results)
