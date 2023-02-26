import sys
import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt

folder_path = "/home/yaga/Documents/GitHub/mfpml"
sys.path.insert(0, folder_path)

# import local functions
from mfpml.design_of_experiment.sf_samplers import LatinHyperCube
from mfpml.problems.sf_functions import Forrester
from mfpml.design_of_experiment.plot_figures import plot_sf_sampling
from mfpml.models.gaussian_processes import StandardGPRModel

# user-defined parameters
num_sample = 10
training_iter = 200
num_test = 1000
learning_rate = 0.1
# define the design space
function = Forrester()
design_space = function.design_space

# sampling
sampler = LatinHyperCube(design_space=design_space, seed=None)
sample_x = sampler.get_samples(num_samples=num_sample)
sample_y = function.f(sample_x)




train_x = torch.from_numpy(sample_x)
train_y = torch.from_numpy(sample_y).flatten()

# gpr model
gprmodel = StandardGPRModel()

gprmodel.train(
    train_x=train_x, train_y=train_y, training_iteration=200, print_info=False
)

# prediction
# define the test points
num_test = 1000
# test_x = np.linspace(0, 1, num_test)
test_x_temp = np.linspace(0, 1, num_test)
test_y__temp = function.f(test_x_temp)
test_x = torch.from_numpy(test_x_temp)
test_y = torch.from_numpy(test_y__temp).flatten()
pred = gprmodel.predict(test_x=test_x)
# get mean 
pred_mean = pred.mean.numpy()
lower, upper = pred.confidence_region()
pred_std = pred.variance.numpy()
lower_1, upper_1 = pred_mean - 2 * pred_std, pred_mean + 2 * pred_std
# plot the fitting
with torch.no_grad():
    f, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.plot(train_x.numpy(), train_y.numpy(), "k*")
    ax.plot(test_x.numpy(), test_y.numpy(), "r--")
    ax.plot(test_x.numpy(), pred_mean, "b")
    ax.fill_between(test_x, lower.numpy(), upper.numpy(), alpha=0.5)
    ax.plot(test_x.numpy(), pred_mean, "m")
    ax.fill_between(test_x, lower_1, upper_1, alpha=0.5)
    # ax.legend(["Observed Data", "True response", "Mean", "Confidence"])
    plt.show()
