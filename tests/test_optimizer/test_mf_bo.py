# third party packages
import numpy as np
import pytest

from mfpml.design_of_experiment.mf_samplers import MFLatinHyperCube
from mfpml.models.co_kriging import CoKriging
from mfpml.models.hierarchical_kriging import HierarchicalKriging
from mfpml.models.scale_kriging import ScaledKriging
from mfpml.optimization.mf_acqusitions import (augmentedEI, extendedPI, vfei,
                                               vflcb)
from mfpml.optimization.mf_uncons_bo import BayesUnConsOpt
from mfpml.problems.mf_functions import Forrester_1a, mf_Hartman3

# define problem
func = Forrester_1a()


# define sampler
sampler = MFLatinHyperCube(design_space=func._design_space, seed=7)

# get initial samples
sample_x = sampler.get_samples(
    num_hf_samples=3 * func.num_dim, num_lf_samples=6 * func.num_dim
)
sample_y = func(sample_x)

#
mf_model = ScaledKriging(design_space=func._input_domain)
acf1 = vfei()
acf2 = augmentedEI()
acf3 = vflcb()
acf4 = extendedPI()


def test_mf_bo_vfei():

    # initialize the BayesOpt class
    opti = BayesUnConsOpt(problem=func)
    # note by changing acquisition, to lcb and ei, we can get different results

    opti.run_optimizer(
        mf_surrogate=mf_model,
        acquisition=acf1,
        max_iter=20,
        init_x=sample_x,
        init_y=sample_y,
    )
    best_y = opti.best_objective()
    assert (best_y - func.optimum) < 1e-1


def test_mf_bo_augmentedEI():

    # initialize the BayesOpt class
    opti = mfBayesOpt(problem=func)
    # note by changing acquisition, to lcb and ei, we can get different results

    opti.run_optimizer(
        mf_surrogate=mf_model,
        acquisition=acf2,
        max_iter=20,
        init_x=sample_x,
        init_y=sample_y,
    )
    best_y = opti.best_objective()
    assert (best_y - func.optimum) < 1e-1


def test_mf_bo_vflcb():

    # initialize the BayesOpt class
    opti = mfBayesOpt(problem=func)
    # note by changing acquisition, to lcb and ei, we can get different results

    opti.run_optimizer(
        mf_surrogate=mf_model,
        acquisition=acf3,
        max_iter=20,
        init_x=sample_x,
        init_y=sample_y,
    )
    best_y = opti.best_objective()
    assert (best_y - func.optimum) < 1e-1


def test_mf_bo_extendedPI():

    # initialize the BayesOpt class
    opti = mfBayesOpt(problem=func)
    # note by changing acquisition, to lcb and ei, we can get different results

    opti.run_optimizer(
        mf_surrogate=mf_model,
        acquisition=acf4,
        max_iter=20,
        init_x=sample_x,
        init_y=sample_y,
    )
    best_y = opti.best_objective()
    assert (best_y - func.optimum) < 1e-1
