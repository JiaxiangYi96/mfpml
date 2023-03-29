import numpy as np
import pytest

from mfpml.design_of_experiment.space import DesignSpace

pytestmark = pytest.mark.smoke


# Continuous space tests


def test_design_space():
    space_dict = {"x1": [0.0, 1.0], "x2": [0.0, 1.0]}
    space_array = np.array([[0.0, 1.0], [0.0, 1.0]])

    space = DesignSpace(
        names=["x1", "x2"], low_bound=[0.0, 0.0], high_bound=[1.0, 1.0]
    )
    assert space_dict == space.design_space
    assert space_array == pytest.approx(space.input_domain)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
