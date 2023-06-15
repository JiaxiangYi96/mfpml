import os
import sys
from time import process_time_ns

import numpy as np

from mfpml.design_of_experiment.mf_samplers import LatinHyperCube
from mfpml.models.mf_surrogates import HierarchicalKriging
from mfpml.optimization.mf_acqusitions import (
    augmentedEI,
    extendedPI,
    vfei,
    vflcb,
)
from mfpml.optimization.mfBO import MFSOBO
from mfpml.problems.mf_functions import Forrester_1a, mf_Hartman3

func = Forrester_1a()
print(func.optimum)
sampler = LatinHyperCube(design_space=func._design_space, seed=7)

sample_x = sampler.get_samples(
    num_hf_samples=3 * func.num_dim, num_lf_samples=6 * func.num_dim
)
sample_y = func(sample_x)

HK = HierarchicalKriging(design_space=func._input_domain)
acf1 = vfei()
acf2 = augmentedEI()
acf3 = vflcb()
acf4 = extendedPI()

opti = MFSOBO(problem=func)

opti.run_optimizer(
    mf_surrogate=HK,
    acqusition=acf3,
    max_iter=10,
    init_X=sample_x,
    init_Y=sample_y,
)
opti.historical_plot()
