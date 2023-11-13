import numpy as np
import pytest

from mfpml.design_of_experiment.space import DesignSpace

pytestmark = pytest.mark.smoke


# Continuous space tests


# @pytest.mark.smoke
def test_design_space() -> None:
    space_dict = {"x1": [0.0, 1.0], "x2": [0.0, 1.0]}
    space_array = np.array([[0.0, 1.0], [0.0, 1.0]])

    space = DesignSpace(
        names=["x1", "x2"], low_bound=[0.0, 0.0], high_bound=[1.0, 1.0]
    )
    assert space_dict == space.design_space
    assert space_array == pytest.approx(space.input_domain)


def test_design_space_creation() -> None:
    # Test design space creation
    names = ['x1', 'x2']
    low_bound = [0, 0]
    high_bound = [1, 1]

    design_space = DesignSpace(names, low_bound, high_bound)

    # Test design_space property
    expected_design_space = {'x1': [0, 1], 'x2': [0, 1]}
    assert design_space.design_space == expected_design_space


def test_input_domain() -> None:
    # Test input_domain property
    names = ['x1', 'x2']
    low_bound = [0, 0]
    high_bound = [1, 1]

    design_space = DesignSpace(names, low_bound, high_bound)

    expected_input_domain = np.array([[0, 1], [0, 1]])
    np.testing.assert_array_equal(
        design_space.input_domain, expected_input_domain)


def test_inconsistent_lengths() -> None:
    # Test inconsistent lengths of input lists
    names = ['x1', 'x2']
    # Inconsistent with the length of names and high_bound
    low_bound = [0, 0, 0]
    high_bound = [1, 1]

    with pytest.raises(AssertionError):
        design_space = DesignSpace(names, low_bound, high_bound)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
