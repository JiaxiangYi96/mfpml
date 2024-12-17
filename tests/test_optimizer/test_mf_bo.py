# third party packages

import pytest

from mfpml.design_of_experiment.mf_samplers import MFLatinHyperCube
from mfpml.models.hierarchical_kriging import HierarchicalKriging
from mfpml.optimization.mf_acqusitions import (
    AugmentedEI, VFEI, VFLCB, ExtendedPI)
from mfpml.optimization.mf_uncons_bo import mfUnConsBayesOpt
from mfpml.problems.mf_functions import Forrester_1a

# define problem
func = Forrester_1a()


# define sampler
sampler = MFLatinHyperCube(design_space=func.input_domain, num_fidelity=2)


#
mf_model = HierarchicalKriging(design_space=func.input_domain)
acf1 = VFEI()
acf2 = AugmentedEI()
acf3 = VFLCB()
acf4 = ExtendedPI()


def test_mf_bo_vfei():

    # initialize the BayesOpt class
    opti = mfUnConsBayesOpt(problem=func,
                            acquisition=acf1,
                            num_init=[4, 12],
                            verbose=False,
                            seed=4)
    # note by changing acquisition, to lcb and ei, we can get different results

    opti.run_optimizer(max_iter=20, stopping_error=0.01, cost_ratio=5.0)
    best_y = opti.best
    assert (best_y - func.optimum) < 1e-1


def test_mf_bo_augmentedEI():

    # initialize the BayesOpt class
    opti = mfUnConsBayesOpt(problem=func,
                            acquisition=acf2,
                            num_init=[4, 12],
                            verbose=False,
                            seed=4)
    # note by changing acquisition, to lcb and ei, we can get different results

    opti.run_optimizer(max_iter=20, stopping_error=0.01, cost_ratio=5.0)
    best_y = opti.best
    assert (best_y - func.optimum) < 1e-1


def test_mf_bo_vflcb():

    # initialize the BayesOpt class
    opti = mfUnConsBayesOpt(problem=func,
                            acquisition=acf3,
                            num_init=[4, 12],
                            verbose=False,
                            seed=4)
    # note by changing acquisition, to lcb and ei, we can get different results

    opti.run_optimizer(max_iter=20, stopping_error=0.01, cost_ratio=5.0)
    best_y = opti.best
    assert (best_y - func.optimum) < 1e-1


def test_mf_bo_extendedPI():

    # initialize the BayesOpt class
    opti = mfUnConsBayesOpt(problem=func,
                            acquisition=acf4,
                            num_init=[4, 12],
                            verbose=False,
                            seed=4)
    # note by changing acquisition, to lcb and ei, we can get different results

    opti.run_optimizer(max_iter=20, stopping_error=0.01, cost_ratio=5.0)
    best_y = opti.best
    assert (best_y - func.optimum) < 1e-1
