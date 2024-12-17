import numpy as np
import pytest

from mfpml.optimization.evolutionary_algorithms import DE, PSO
from mfpml.problems.sf_functions import Ackley

pytestmark = pytest.mark.smoke


# @pytest.mark.smoke
def test_pso() -> None:
    function = Ackley(num_dim=2)
    design_space = function._input_domain
    optimum = function.optimum
    optimum_scheme = function.optimum_scheme
    # initialize optimizer
    optimizer = PSO(num_gen=1000, num_pop=50)
    opt_results, best_y, best_x = optimizer.run_optimizer(
        func=function.f,
        num_dim=function.num_dim,
        design_space=design_space,
        save_step_results=False,
        print_info=False,
        stopping_error=None,
    )

    assert best_y == pytest.approx(optimum)
    assert best_x == pytest.approx(optimum_scheme)


def test_DE() -> None:
    function = Ackley(num_dim=2)
    design_space = function._input_domain
    optimum = function.optimum
    optimum_scheme = function.optimum_scheme
    # initialize optimizer
    optimizer = DE(num_gen=1000, num_pop=50, strategy="DE/best/1/bin")
    opt_results, best_y, best_x = optimizer.run_optimizer(
        func=function.f,
        num_dim=function.num_dim,
        design_space=design_space,
        save_step_results=False,
        print_info=False,
        stopping_error=None,
    )

    assert best_y == pytest.approx(optimum)
    assert best_x == pytest.approx(optimum_scheme)

