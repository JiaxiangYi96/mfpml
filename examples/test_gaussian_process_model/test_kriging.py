import os
import sys

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import max_error, mean_squared_error, r2_score

# from mfpml.models.mf_surrogates import (
#     HierarchicalKriging,
#     ScaledKriging,
#     CoKriging,
# )
from mfpml.design_of_experiment.singlefideliy_samplers import LatinHyperCube
from mfpml.models.kriging import Kriging
from mfpml.optimization.evolutionary_algorithms import PSO
from mfpml.problems.singlefidelity_functions import (Branin, Hartman3, Sasena,
                                                     Sixhump)

func = Hartman3()
# func = mf_Hartman3()

sampler = LatinHyperCube(design_space=func._design_space, seed=17)
sample_x = sampler.get_samples(num_samples=50)
test_x = sampler.get_samples(num_samples=1000)

sample_y = func.f(sample_x)
test_y = func.f(test_x)
# print('sample y:', sample_y)
# print('test y', test_y)
pso_opt = PSO(num_gen=500, num_pop=50)

sfK = Kriging(design_space=func._input_domain)
lfKrg = Kriging(design_space=func._input_domain)
rbf = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gaussian_process = GaussianProcessRegressor(
    kernel=rbf, n_restarts_optimizer=20
)
# HK = HierarchicalKriging(design_space=func._input_domain)
# ScK = ScaledKriging(design_space=func._input_domain)
# scK_rho = ScaledKriging(
#     design_space=func._input_domain, rho_optimize=True, rho_method="bumpiness"
# )
# coK = CoKriging(design_space=func._input_domain)

sfK.train(sample_x, sample_y)
gaussian_process.fit(sample_x, sample_y)
# HK.train(sample_x, sample_y)
# ScK.train(sample_x, sample_y)
# scK_rho.train(sample_x, sample_y)
# coK.train(sample_x, sample_y)

sf_pre, sf_std = sfK.predict(test_x, return_std=True)
sf_pre_2, sf_std_2 = gaussian_process.predict(test_x, return_std=True)
sf_pre_2 = sf_pre_2.reshape(-1, 1)
sf_std_2 = sf_std_2.reshape(-1, 1)


mse = mean_squared_error(sf_pre, test_y)
r2 = r2_score(sf_pre, test_y)
mae = max_error(sf_pre, test_y)
print(mse, r2, mae)
mse1 = mean_squared_error(sf_pre_2, test_y)
r21 = r2_score(sf_pre_2, test_y)
mae1 = max_error(sf_pre_2, test_y)
print(mse1, r21, mae1)
