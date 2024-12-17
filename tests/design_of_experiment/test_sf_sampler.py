import numpy as np
import pytest
from mfpml.design_of_experiment.sf_samplers import (
    FixNumberSampler,
    LatinHyperCube,
    RandomSampler,
    SobolSequence,
)

pytestmark = pytest.mark.smoke

@pytest.fixture
def design_space():
    return np.array([[1, 5], [-1, 1]])

def test_random_sampler(design_space):
    sampler = RandomSampler(design_space=design_space)
    samples = sampler.get_samples(num_samples=10, seed=123456)
    expected = np.array(
        [[1.50787933, 0.93343568],
         [2.04190402, 0.79447305],
         [2.50699886, -0.32755651],
         [2.80550588, 0.68051017],
         [1.49240858, 0.0860524],
         [2.4920489, -0.10400635],
         [1.51776272, 0.71975741],
         [4.28155345, -0.29589292],
         [1.91554922, 0.5535675],
         [3.37913436, -0.72489289]]
    )
    assert expected == pytest.approx(samples)

def test_latin_hypercube(design_space):
    sampler = LatinHyperCube(design_space=design_space)
    samples = sampler.get_samples(num_samples=10, seed=123456)
    expected = np.array(
        [[2.3453945, 0.32303767],
         [2.98102183, -0.39105055],
         [4.63757963, -0.09139391],
         [1.28778661, 0.47277024],
         [3.40771572, 0.83751364],
         [2.0779207, 0.08639781],
         [1.79172513, 0.6853799],
         [3.23512346, -0.86905749],
         [4.30781921, -0.45634598],
         [4.19106622, -0.65025851]]
    )
    assert expected == pytest.approx(samples)

def test_sobol_sequence(design_space):
    sampler = SobolSequence(design_space=design_space)
    samples = sampler.get_samples(num_samples=2, seed=123456)
    expected = np.array([[3.59055832, -0.43098223],
                         [2.46733653, 0.36271125]])
    assert expected == pytest.approx(samples)

def test_fix_number_sampler():
    sampler = FixNumberSampler(np.array([[1], [3]]))
    samples = sampler.get_samples(num_samples=2, seed=123456)
    expected = np.array([[1, 3], [1, 3]])
    assert expected == pytest.approx(samples)
