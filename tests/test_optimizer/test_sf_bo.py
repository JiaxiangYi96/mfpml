# third party packages
import numpy as np
import pytest

# local funstions
from mfpml.models.kriging import Kriging
from mfpml.optimization.sf_uncons_acqusitions import EI, LCB, PI
from mfpml.optimization.sf_uncons_bo import BayesUnConsOpt
from mfpml.problems.sf_functions import Forrester

# define function
func = Forrester()

# initialize sampler
# import necessary functions from mfpml

# for better illustration we want to reduce samples of kriging model
x = np.array([0.12, 0.36, 0.8]).reshape((-1, 1))
y = func.f(x)
# train the model
kriging = Kriging(design_space=func._input_domain)
kriging.train(x, y)

# regenerate test samples within[0,1]
test_x = np.linspace(0, 1, 101, endpoint=True).reshape(-1, 1)
test_y = func.f(test_x)

# get kriging prediction
kriging_pre, kriging_std = kriging.predict(test_x, return_std=True)

# initialize the acqusition function
ei = EI()
lcb = LCB()
pi = PI()


def test_sf_acquisitions():
    # get the acquisition value for the test samples
    ei_value = -ei.eval(test_x, kriging)
    lcb_value = lcb.eval(test_x, kriging)
    pi_value = -pi.eval(test_x, kriging)

    # assert shape
    assert ei_value.shape == test_y.shape
    assert lcb_value.shape == test_y.shape
    assert pi_value.shape == test_y.shape


def test_sg_bo():

    # initialize the BayesOpt class
    bo = BayesUnConsOpt(problem=func)
    # note by changing acquisition, to lcb and ei, we can get different results
    bo.run_optimizer(init_x=x,
                     init_y=y,
                     max_iter=10,
                     surrogate=kriging,
                     acquisition=pi,
                     print_info=False)
    best_x = bo.best_x
    best_y = bo.best
    assert (best_y - func.optimum) < 1e-1
