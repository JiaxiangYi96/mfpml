import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF 

from mfpml.base.space import DesignSpace 
from mfpml.design_of_experiment.sf_samplers import * 
from mfpml.problems.sf_functions import *
from mfpml.models.mf_surrogates import Kriging, HierarchicalKriging, ScaledKriging 

design_space = {'x1': [0., 1.], 'x2': [0., 1.]}

sampler = LatinHyperCube(design_space=design_space, seed=12)
sampler.get_samples(num_samples=10)
sample_x = sampler.samples.to_numpy() 
sampler.get_samples(num_samples=5)
test_x = sampler.samples.to_numpy()
#print('sample x: \n', sample_x)

func = Branin() 
sample_y = func.f(sample_x)
test_y = func.f(test_x)
#print(sample_y) 

print('test_x: \n', test_x)
krg = Kriging(kernel_mode='KRG', n_dim=sampler.num_dim)
krg.train(sample_x, sample_y)
pre_y, std_y = krg.predict(test_x, return_std=True) 

rbf = 1. * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) 
gaussian_process = GaussianProcessRegressor(kernel=rbf, n_restarts_optimizer=9)
gaussian_process.fit(sample_x, sample_y)
gpr_y, gpr_std = gaussian_process.predict(test_x, return_std=True)
gpr_y = gpr_y.reshape(-1, 1)
gpr_std = gpr_std.reshape(-1, 1)

print(np.concatenate([test_y, pre_y, gpr_y], axis=1))
print(np.concatenate([std_y, gpr_std], axis=1))