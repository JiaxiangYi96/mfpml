import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import numpy as np

from mfpml.problems.mf_functions import Forrester_1a, mf_Hartman3
from mfpml.design_of_experiment.mf_samplers import LatinHyperCube
from mfpml.models.mf_surrogates import HierarchicalKriging
from mfpml.optimization.mf_acqusitions import vfei
from mfpml.optimization.mfBO import MFSOBO

func = mf_Hartman3()
sampler = LatinHyperCube(design_space=func.design_space, seed=7)

sample_x = sampler.get_samples(num_hf_samples=3*func.num_dim, num_lf_samples=6*func.num_dim)
# sample_x = {'hf': np.random.rand(3,1), 
#             'lf': np.random.rand(6,1)}
sample_y = func(sample_x)

HK = HierarchicalKriging(bounds=func.bounds)
acf = vfei()
opti = MFSOBO(problem=func, mf_surrogate=HK)

opti._first_run(sample_x, sample_y)
opti.run(acqusition=acf, max_iter=5)

opti.plotIterativeBest()