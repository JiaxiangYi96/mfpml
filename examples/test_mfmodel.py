
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import numpy as np

from mfpml.models.mf_surrogates import HierarchicalKriging, ScaledKriging
from mfpml.design_of_experiment.mf_samplers import LatinHyperCube
from mfpml.problems.mf_functions import mf_Hartman3

func = mf_Hartman3() 
design_space = func.design_space

sampler = LatinHyperCube(design_space=design_space, seed=7)
sample_x = sampler.get_samples(num_hf_samples=5, num_lf_samples=10) 
sample_x['hf'] = sample_x['hf'].to_numpy()
sample_x['lf'] = sample_x['lf'].values
test_x = sampler.get_samples(num_hf_samples=3, num_lf_samples=3) 
test_x = test_x['hf'].values
print('sample x:', sample_x)
print('test x:', test_x)

sample_y = func(sample_x)
test_y = func.hf(test_x)
print('sample y:', sample_y) 
print('test y', test_y)

HK = HierarchicalKriging(kernel_mode='KRG', n_dim=func.num_dim)
HK.train(sample_x, sample_y)
pre, std = HK.predict(test_x, return_std=True) 

print('True obj values of y and pre: \n', np.concatenate([test_y, pre], axis=1))