import numpy as np

from mfpml.models.basis_functions import Linear, Ordinary, Quadratic


def test_ordinary():
    basis = Ordinary()
    x = np.array([1, 2, 3]).reshape(-1, 1)
    result = basis(x)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == x.shape[0]
    # test if the first column is all 1
    assert np.all(result[:, 0] == 1)


def test_linear():
    basis = Linear()
    x = np.array([1, 2, 3]).reshape(-1, 1)
    result = basis(x)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == x.shape[0]
    # test if the first column is all 1
    assert np.all(result[:, 0] == 1)


def test_quadratic():
    basis = Quadratic()
    x = np.array([1, 2, 3]).reshape(-1, 1)
    result = basis(x)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == x.shape[0]
    # test if the first column is all 1
    assert np.all(result[:, 0] == 1)
