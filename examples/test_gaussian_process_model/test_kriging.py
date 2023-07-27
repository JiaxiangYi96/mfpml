import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import max_error, mean_squared_error, r2_score

from mfpml.design_of_experiment.singlefideliy_samplers import LatinHyperCube
from mfpml.models.sf_gpr import Kriging
from mfpml.optimization.evolutionary_algorithms import PSO
from mfpml.problems.singlefidelity_functions import Forrester

func = Forrester()


sampler = LatinHyperCube(design_space=func._design_space, seed=17)
sample_x = sampler.get_samples(num_samples=5)
test_x = np.linspace(0, 1, 101, endpoint=True).reshape(-1, 1)
sample_y = func.f(sample_x)
test_y = func.f(test_x)

pso_opt = PSO(num_gen=500, num_pop=50)

sfK = Kriging(design_space=func._input_domain)


sfK.train(sample_x, sample_y)


sf_pre, sf_std = sfK.predict(test_x, return_std=True)


# plot the prediction
fig, ax = plt.subplots()
ax.plot(test_x, test_y, "k-", label="true no noise")
ax.plot(test_x, sf_pre, "b--", label="Predicted")
ax.plot(sample_x, sample_y, "*", label="samples")
ax.fill_between(
    test_x.reshape(-1),
    (sf_pre - 1.96 * sf_std).reshape(-1),
    (sf_pre + 1.96 * sf_std).reshape(-1),
    alpha=0.3,
    color="y",
    label="95% confidence interval",
)
ax.legend(loc="best")
plt.show()
