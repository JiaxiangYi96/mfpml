from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

from mfpml.design_of_experiment.sampler import Sampler
from mfpml.design_of_experiment.singlefideliy_samplers import (
    FixNumberSampler, LatinHyperCube, RandomSampler, SobolSequence)
from mfpml.design_of_experiment.space import DesignSpace

pytestmark = pytest.mark.smoke


def test_sobolsequence() -> None:
    space = DesignSpace(
        names=["x1", "x2"], low_bound=[0.0, 0.0], high_bound=[1.0, 1.0]
    )
    design_space = space.design_space
    sampler = SobolSequence(design_space=design_space, seed=12)
    samples = sampler.get_samples(num_samples=10)
    results = np.array(
        [
            [0.85791405, 0.06944251],
            [0.38605036, 0.98125293],
            [0.32699364, 0.14201562],
            [0.91701757, 0.80735379],
            [0.60346028, 0.43486295],
            [0.13684348, 0.52207483],
            [0.22297508, 0.05505399],
            [0.50147379, 0.89425058],
            [0.94087794, 0.26977609],
            [0.28773614, 0.68709683],
        ]
    )
    assert results == pytest.approx(samples)


def test_lhs() -> None:
    space = DesignSpace(
        names=["x1", "x2"], low_bound=[0.0, 0.0], high_bound=[1.0, 1.0]
    )
    design_space = space.design_space
    sampler = LatinHyperCube(design_space=design_space, seed=12)
    samples = sampler.get_samples(num_samples=10)
    results = np.array(
        [
            [0.07491755, 0.30532471],
            [0.98106796, 0.28207086],
            [0.86501108, 0.77694588],
            [0.33295543, 0.48849206],
            [0.71036906, 0.91418695],
            [0.19971730, 0.84585338],
            [0.28931487, 0.17420450],
            [0.55831040, 0.05463839],
            [0.45318534, 0.50724833],
            [0.67412289, 0.68121098],
        ]
    )
    assert results == pytest.approx(samples)


def test_randome_sampler() -> None:
    space = DesignSpace(
        names=["x1", "x2"], low_bound=[0.0, 0.0], high_bound=[1.0, 1.0]
    )
    design_space = space.design_space
    sampler = RandomSampler(design_space=design_space, seed=12)
    samples = sampler.get_samples(num_samples=2)
    results = np.array([[0.15416284, 0.7400497], [0.26331502, 0.53373939]])
    assert results == pytest.approx(samples)


def test_fix_sampler() -> None:
    design_space = OrderedDict({"x1": 0.1})
    sampler = FixNumberSampler(design_space=design_space, seed=12)
    samples = sampler.get_samples(num_samples=2)
    results = np.array([[0.1], [0.1]])
    assert results == pytest.approx(samples)


def test_create_pandas_frame_not_implemented():
    # Test _create_pandas_frame raises NotImplementedError
    design_space = {'x1': [0, 1], 'x2': [0, 1]}
    sampler = Sampler(design_space)

    with pytest.raises(NotImplementedError):
        sampler._create_pandas_frame()


def test_plot_samples_not_implemented():
    # Test plot_samples raises NotImplementedError
    design_space = {'x1': [0, 1], 'x2': [0, 1]}
    sampler = Sampler(design_space)

    with pytest.raises(NotImplementedError):
        sampler.plot_samples()


def test_plotting_2d():
    # Test plotting
    design_space = OrderedDict({"x1": [0, 1], "x2": [0, 1]})
    sampler = LatinHyperCube(design_space=design_space, seed=12)
    sampler.get_samples(num_samples=10)
    sampler.plot_samples()
    assert True


def test_plotting_1d():
    # Test plotting
    design_space = OrderedDict({"x1": [0, 1]})
    sampler = LatinHyperCube(design_space=design_space, seed=12)
    sampler.get_samples(num_samples=10)
    sampler.plot_samples()
    assert True


def test_plotting_3d():
    # Test plotting
    design_space = OrderedDict({"x1": [0, 1], "x2": [0, 1], "x3": [0, 1]})
    sampler = LatinHyperCube(design_space=design_space, seed=12)
    sampler.get_samples(num_samples=10)

    with pytest.raises(Exception):
        sampler.plot_samples()
