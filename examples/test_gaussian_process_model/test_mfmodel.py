import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from mfpml.design_of_experiment.multifidelity_samplers import (
    MFLatinHyperCube, MFSobolSequence)
from mfpml.models.co_kriging import CoKriging
from mfpml.models.hierarchical_kriging import HierarchicalKriging
from mfpml.models.mf_scale_kriging import ScaledKriging
# from mfpml.models.mf_gprs import HierarchicalKriging, ScaledKriging
# from mfpml.models.sf_gpr import Kriging
from mfpml.optimization.evolutionary_algorithms import PSO
from mfpml.problems.multifidelity_functions import Forrester_1b, mf_Hartman3

func = Forrester_1b()
# func = mf_Hartman3()

sampler = MFSobolSequence(design_space=func._design_space, seed=4)
sample_x = sampler.get_samples(num_hf_samples=4, num_lf_samples=12)
# print(sample_x)
# sample_lx = np.linspace(0, 1.0, 15, endpoint=True).reshape(-1, 1)
# sample_hx = np.array([0.2, 0.6, 0.9]).reshape(-1, 1)
# sample_x = {"hf": sample_hx, "lf": sample_lx}
# test_x = sampler.get_samples(num_hf_samples=100, num_lf_samples=100)
test_x = np.linspace(0, 1, 1000).reshape(-1, 1)
# sampler.plot_samples()
# print('sample x:', sample_x)
# print('test x:', test_x)

sample_y = func(sample_x)
test_hy = func.hf(test_x)
test_ly = func.lf(test_x)
# print('sample y:', sample_y)
# print('test y', test_y)
pso_opt = PSO(num_gen=200, num_pop=20)

# sfK = Kriging(design_space=func._input_domain)
# lfKrg = Kriging(design_space=func._input_domain)
# rbf = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
# gaussian_process = GaussianProcessRegressor(kernel=rbf, n_restarts_optimizer=9)
HK = HierarchicalKriging(design_space=func._input_domain, optimizer=pso_opt)
# ScK = ScaledKriging(design_space=func._input_domain,
#                     rho_optimizer=True, rho_method="bumpiness",
#                     rho_bound=[0.0, 10.0], optimizer=pso_opt)
# scK_rho = ScaledKriging(
#     design_space=func._input_domain, rho_optimize=True, rho_method="bumpiness"
# )
# coK = CoKriging(design_space=func._input_domain)

# sfK.train(sample_x["hf"], sample_y["hf"])
# gaussian_process.fit(sample_x["hf"], sample_y["hf"])
HK.train(sample_x, sample_y)
# ScK.train(sample_x, sample_y)
# scK_rho.train(sample_x, sample_y)
# coK.train(sample_x, sample_y)
# print(ScK.rho)
# print(coK.kernel.param)
# sf_pre, sf_std = sfK.predict(test_x, return_std=True)
# sf_pre_2, sf_std_2 = gaussian_process.predict(test_x, return_std=True)
# sf_pre_2 = sf_pre_2.reshape(-1, 1)
# sf_std_2 = sf_std_2.reshape(-1, 1)
# mf_pre, mf_std = HK.predict(test_x, return_std=True)
mf_pre_2, mf_std_2 = HK.predict(test_x, return_std=True)
# mf_pre_3, mf_std_3 = scK_rho.predict(test_x, return_std=True)
# mf_pre_4, mf_std_4 = coK.predict(test_x, return_std=True)
# print(mf_std_4)
# prediction for low fidelity
lf_pred, lf_std = HK.lf_model.predict(test_x, return_std=True)
# print(mf_std_4)
fig, ax = plt.subplots()
plt.plot(test_x, test_hy, "r--", label="True HF")
plt.plot(test_x, test_ly, "g--", label="True LF")
plt.plot(test_x, mf_pre_2, "b-", label="CoKriging")
plt.fill_between(
    test_x[:, 0],
    mf_pre_2[:, 0] - 1.96 * mf_std_2[:, 0],
    mf_pre_2[:, 0] + 1.96 * mf_std_2[:, 0],
    alpha=0.4,
    color="b",
    label="95% CI",
)
plt.plot(test_x, lf_pred, "k-", label="LF model")
plt.fill_between(
    test_x[:, 0],
    lf_pred[:, 0] - 1.96 * lf_std[:, 0],
    lf_pred[:, 0] + 1.96 * lf_std[:, 0],
    alpha=0.4,
    color="y",
    label="95% CI lf",
)
plt.plot(sample_x["hf"], sample_y["hf"], "kx", label="HF samples")
plt.plot(sample_x["lf"], sample_y["lf"], "k+", label="LF samples")
plt.legend()
plt.show()

# print(
#     "True obj values and pres of single-fidelity models: \n",
#     np.concatenate([test_y, sf_pre, sf_pre_2], axis=1),
# )
# print("Std values of pres: \n", np.concatenate([sf_std, sf_std_2], axis=1))
# print(
#     "Pres of multi-fidelity of models: \n",
#     np.concatenate([mf_pre, mf_pre_2, mf_pre_3, mf_pre_4], axis=1),
# )
# print(
#     "Std values of pres: \n",
#     np.concatenate([mf_std, mf_std_2, mf_std_3, mf_std_4], axis=1),
# )
