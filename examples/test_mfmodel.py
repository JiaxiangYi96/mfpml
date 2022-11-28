import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF 


from mfpml.models.mf_surrogates import Kriging, HierarchicalKriging, ScaledKriging
from mfpml.design_of_experiment.mf_samplers import LatinHyperCube
from mfpml.problems.mf_functions import mf_Hartman3

func = mf_Hartman3() 
design_space = func.design_space

sampler = LatinHyperCube(design_space=design_space, seed=7)
sample_x = sampler.get_samples(num_hf_samples=10, num_lf_samples=15) 
test_x = sampler.get_samples(num_hf_samples=3, num_lf_samples=3) 
test_x = test_x['hf']
# print('sample x:', sample_x)
# print('test x:', test_x)

sample_y = func(sample_x)
test_y = func.hf(test_x)
#print('sample y:', sample_y) 
#print('test y', test_y)

print(func.bounds)
sfK = Kriging(bounds=func.bounds)
rbf = 1. * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) 
gaussian_process = GaussianProcessRegressor(kernel=rbf, n_restarts_optimizer=9)
HK = HierarchicalKriging(bounds=func.bounds)
ScK = ScaledKriging(bounds=func.bounds)

sfK.train(sample_x['hf'], sample_y['hf'])
gaussian_process.fit(sample_x['hf'], sample_y['hf'])
HK.train(sample_x, sample_y)
ScK.train(sample_x, sample_y)

sf_pre, sf_std = sfK.predict(test_x, return_std=True)
sf_pre_2, sf_std_2 = gaussian_process.predict(test_x, return_std=True)
sf_pre_2 = sf_pre_2.reshape(-1, 1)
sf_std_2 = sf_std_2.reshape(-1, 1)
mf_pre, mf_std = HK.predict(test_x, return_std=True) 
mf_pre_2, mf_std_2 = ScK.predict(test_x, return_std=True)

print('True obj values and pres of single-fidelity models: \n', np.concatenate([test_y, sf_pre, sf_pre_2], axis=1))
print('Std values of pres: \n', np.concatenate([sf_std, sf_std_2], axis=1))
print('Pres of multi-fidelity of models: \n', np.concatenate([mf_pre, mf_pre_2], axis=1))
print('Std values of pres: \n', np.concatenate([mf_std, mf_std_2], axis=1))