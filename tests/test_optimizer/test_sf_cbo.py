# third party packages
import numpy as np
import pytest


from mfpml.optimization.sf_cons_acqusitions import CEI
from mfpml.optimization.sf_cons_bo import BayesConsOpt
from mfpml.problems.sf_cons_functions import Branin_1


def test_sf_cbo():

    # initialize the BayesOpt class
    optimizer = BayesConsOpt(problem=Branin_1(),
                             acquisition=CEI(),
                             num_init=10,
                             verbose=False,)
    # note by changing acquisition, to lcb and ei, we can get different results
    optimizer.run_optimizer(max_iter=20, stopping_error=0.01)
    best_y = optimizer.best
    assert (best_y - Branin_1().optimum) < 1.0
