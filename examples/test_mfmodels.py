import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from mfpml.base.space import DesignSpace 
from mfpml.design_of_experiment.sf_samplers import * 
from mfpml.problems.sf_functions import *
from mfpml.models.mf_surrogates import Kriging, HierarchicalKriging, ScaledKriging 

design_space = {'x1': [0., 1.], 'x2': [0., 1.]}

sampler = LatinHyperCube(design_space=design_space, seed=12)
sampler.get_samples(num_samples=10)
sample_x = sampler.samples.to_numpy()
print(sample_x)

func = Branin() 
sample_y = func.f(sample_x)
print(sample_y)