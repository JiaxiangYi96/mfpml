import os
import sys
from time import process_time_ns

import numpy as np

from mfpml.design_of_experiment.multifidelity_samplers import MFLatinHyperCube
from mfpml.models.co_kriging import CoKriging
from mfpml.models.hierarchical_kriging import HierarchicalKriging
from mfpml.models.mf_scale_kriging import ScaledKriging
from mfpml.optimization.mf_acqusitions import (augmentedEI, extendedPI, vfei,
                                               vflcb)
from mfpml.optimization.mfbo import mfBayesOpt
from mfpml.problems.multifidelity_functions import Forrester_1a, mf_Hartman3

func = Forrester_1a()
print(func.optimum)
sampler = MFLatinHyperCube(design_space=func._design_space, seed=7)

sample_x = sampler.get_samples(
    num_hf_samples=3 * func.num_dim, num_lf_samples=6 * func.num_dim
)
sample_y = func(sample_x)

HK = ScaledKriging(design_space=func._input_domain)
acf1 = vfei()
acf2 = augmentedEI()
acf3 = vflcb()
acf4 = extendedPI()

opti = mfBayesOpt(problem=func)

opti.run_optimizer(
    mf_surrogate=HK,
    acquisition=acf1,
    max_iter=10,
    init_x=sample_x,
    init_y=sample_y,
)
opti.historical_plot(figsize=(5, 4))
