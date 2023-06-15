import sys, os


import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


from mfpml.models.kriging import Kriging
from mfpml.models.mf_surrogates import (
    HierarchicalKriging,
    ScaledKriging,
    CoKriging,
)
from mfpml.design_of_experiment.mf_samplers import LatinHyperCube
from mfpml.problems.mf_functions import Forrester_1a, mf_Hartman3
from mfpml.optimization.evolutionary_algorithms import PSO

func = Forrester_1a()
# func = mf_Hartman3()

sampler = LatinHyperCube(design_space=func._design_space, seed=17)
sample_x = sampler.get_samples(num_hf_samples=9, num_lf_samples=15)
test_x = sampler.get_samples(num_hf_samples=3, num_lf_samples=3)
test_x = test_x["hf"]
# print('sample x:', sample_x)
# print('test x:', test_x)

sample_y = func(sample_x)
test_y = func.hf(test_x)
# print('sample y:', sample_y)
# print('test y', test_y)
pso_opt = PSO(num_gen=200, num_pop=40)

sfK = Kriging(design_space=func._input_domain)
lfKrg = Kriging(design_space=func._input_domain)
rbf = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gaussian_process = GaussianProcessRegressor(kernel=rbf, n_restarts_optimizer=9)
HK = HierarchicalKriging(design_space=func._input_domain)
ScK = ScaledKriging(design_space=func._input_domain)
scK_rho = ScaledKriging(
    design_space=func._input_domain, rho_optimize=True, rho_method="bumpiness"
)
coK = CoKriging(design_space=func._input_domain)

sfK.train(sample_x["hf"], sample_y["hf"])
gaussian_process.fit(sample_x["hf"], sample_y["hf"])
HK.train(sample_x, sample_y)
ScK.train(sample_x, sample_y)
scK_rho.train(sample_x, sample_y)
coK.train(sample_x, sample_y)

sf_pre, sf_std = sfK.predict(test_x, return_std=True)
sf_pre_2, sf_std_2 = gaussian_process.predict(test_x, return_std=True)
sf_pre_2 = sf_pre_2.reshape(-1, 1)
sf_std_2 = sf_std_2.reshape(-1, 1)
mf_pre, mf_std = HK.predict(test_x, return_std=True)
mf_pre_2, mf_std_2 = ScK.predict(test_x, return_std=True)
mf_pre_3, mf_std_3 = scK_rho.predict(test_x, return_std=True)
mf_pre_4, mf_std_4 = coK.predict(test_x, return_std=True)

print(
    "True obj values and pres of single-fidelity models: \n",
    np.concatenate([test_y, sf_pre, sf_pre_2], axis=1),
)
print("Std values of pres: \n", np.concatenate([sf_std, sf_std_2], axis=1))
print(
    "Pres of multi-fidelity of models: \n",
    np.concatenate([mf_pre, mf_pre_2, mf_pre_3, mf_pre_4], axis=1),
)
print(
    "Std values of pres: \n",
    np.concatenate([mf_std, mf_std_2, mf_std_3, mf_std_4], axis=1),
)
