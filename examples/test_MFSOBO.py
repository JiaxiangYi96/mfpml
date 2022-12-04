import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import numpy as np

from mfpml.problems.mf_functions import Forrester_1a, mf_Hartman3
from mfpml.design_of_experiment.mf_samplers import LatinHyperCube
from mfpml.models.mf_surrogates import HierarchicalKriging
from mfpml.optimization.mf_acqusitions import vfei, vflcb
from mfpml.optimization.mfBO import MFSOBO

func = mf_Hartman3()
sampler = LatinHyperCube(design_space=func._design_space, seed=7)

sample_x = sampler.get_samples(num_hf_samples=3*func.num_dim, num_lf_samples=6*func.num_dim)
sample_y = func(sample_x)

HK = HierarchicalKriging(design_space=func._input_domain)
# acf = vfei()
acf = vflcb()
opti = MFSOBO(problem=func)

opti.run_optimizer(mf_surrogate=HK, acqusition=acf, max_iter=10, init_X=sample_x, init_Y=sample_y)

opti.historical_plot()