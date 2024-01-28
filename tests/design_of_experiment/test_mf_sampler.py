import numpy as np
import pytest

from mfpml.design_of_experiment.multifidelity_samplers import (
    MFLatinHyperCube, MFSobolSequence)

pytestmark = pytest.mark.smoke


def test_mf_lhs_nested() -> None:
    design_space = np.array([[5., 10], [1, 2]])
    mf_sampler = MFLatinHyperCube(design_space=design_space,
                                  num_fidelity=2,
                                  nested=True)

    data = mf_sampler.get_samples(num_samples=[2, 5], seed=123456)
    # test results
    results = [np.array([[8.666279, 1.14971204],
                         [9.33874143, 1.77814134]]),
               np.array([[7.21168694, 1.40160066],
                         [6.0414317, 1.2244589],
                         [9.33874143,
                          1.77814134],
                         [5.04239678,
                          1.91309308],
                         [8.666279, 1.14971204]])]
    assert results[0] == pytest.approx(data[0])
    assert results[1] == pytest.approx(data[1])


def test_mf_lhs_unnested() -> None:
    design_space = np.array([[5., 10], [1, 2]])
    mf_sampler = MFLatinHyperCube(design_space=design_space,
                                  num_fidelity=2,
                                  nested=False)

    data = mf_sampler.get_samples(num_samples=[2, 5], seed=123456)
    # test results
    results = [np.array([[8.40871563, 1.30759417],
                        [7.38138645, 1.52237363]]),
               np.array([[7.21168694, 1.40160066],
                        [6.0414317, 1.2244589],
                         [9.33874143, 1.77814134],
                         [5.04239678, 1.91309308],
                         [8.666279, 1.14971204]])]
    assert results[0] == pytest.approx(data[0])
    assert results[1] == pytest.approx(data[1])


def test_sobol_unnested() -> None:
    design_space = np.array([[5., 10], [1, 2]])
    mf_sampler = MFSobolSequence(design_space=design_space,
                                 num_fidelity=2,
                                 nested=False)

    data = mf_sampler.get_samples(num_samples=[2, 5], seed=123456)
    # test results
    results = [np.array([[8.2381979, 1.28450888],
                         [6.83417066, 1.68135563]]),
               np.array([[8.2381979, 1.28450888],
                        [6.83417066,
                         1.68135563],
                        [7.40140805,
                         1.47841079],
                        [7.55499782,
                         1.61233338],
                        [9.61076839, 1.08658168]])]
    assert results[0] == pytest.approx(data[0])
    assert results[1] == pytest.approx(data[1])


def test_sobol_nested() -> None:
    design_space = np.array([[5., 10], [1, 2]])
    mf_sampler = MFSobolSequence(design_space=design_space,
                                 num_fidelity=2,
                                 nested=True)

    data = mf_sampler.get_samples(num_samples=[2, 5], seed=123456)
    # test results
    results = [np.array([[8.2381979, 1.28450888],
                         [6.83417066, 1.68135563]]),
               np.array([[8.2381979, 1.28450888],
                         [6.83417066,
                          1.68135563],
                         [7.40140805,
                          1.47841079],
                         [7.55499782,
                          1.61233338],
                         [9.61076839, 1.08658168]])]
    assert results[0] == pytest.approx(data[0])
    assert results[1] == pytest.approx(data[1])
