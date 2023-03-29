import sys
import torch
import gpytorch
import math
import numpy as np
import matplotlib.pyplot as plt

folder_path = "/home/yaga/Documents/GitHub/mfpml"
sys.path.insert(0, folder_path)

# import local functions
from mfpml.design_of_experiment.sf_samplers import LatinHyperCube
from mfpml.problems.sf_functions import Forrester
from mfpml.design_of_experiment.plot_figures import plot_sf_sampling
from mfpml.models.pytorch_gpy import GPR

# Training data is 100 points in [0,1] inclusive regularly spaced
train_x = torch.linspace(0, 1, 100)
# True function is sin(2*pi*x) with Gaussian noise
train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(
    train_x.size()
) * math.sqrt(0.04)
# gpr model
gprmodel = GPR()

gprmodel.train(
    train_x=train_x, train_y=train_y, training_iteration=200, print_info=True
)

# prediction
# define the test points
num_test = 1000
# test_x = np.linspace(0, 1, num_test)
test_x = torch.linspace(0, 1, 1000)
# True function is sin(2*pi*x) with Gaussian noise
test_y = torch.sin(test_x * (2 * math.pi)) + torch.randn(
    test_x.size()
) * math.sqrt(0.04)
pred = gprmodel.predict(test_x=test_x)
# get mean
pred_mean = pred.mean.numpy()
lower, upper = pred.confidence_region()
# plot the fitting
with torch.no_grad():
    f, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.plot(train_x.numpy(), train_y.numpy(), "k*")
    # ax.plot(test_x.numpy(), test_y.numpy(), "ro")
    ax.plot(test_x.numpy(), pred_mean, "b")
    ax.fill_between(test_x, lower.numpy(), upper.numpy(), alpha=0.5)
    ax.plot(test_x.numpy(), pred_mean, "m")
    ax.legend(["Observed Data", "True response", "Mean", "Confidence"])
    plt.show()

# # define the gpr model
# likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
#     noise=torch.zeros(num_sample),
#     learn_additional_noise=False,
# )
# model = ExactGPModel(
#     train_inputs=train_x, train_targets=train_y, likelihood=likelihood
# )
# # train the model
# model.train()
# likelihood.train()

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# # bulid the loss function
# mll = gpytorch.mlls.ExactMarginalLogLikelihood(
#     likelihood=likelihood, model=model
# )
# # set up the training process
# for i in range(training_iter):
#     # Zero gradients from previous iteration
#     optimizer.zero_grad()
#     # Output from model
#     output = model(train_x)
#     # Calc loss and backprop gradients
#     loss = -mll(output, train_y)
#     loss.backward()
#     print(
#         "Iter %d/%d - Loss: %.3f   lengthscale: %.3f "
#         % (
#             i + 1,
#             training_iter,
#             loss.item(),
#             model.covar_module.base_kernel.lengthscale.item(),
#             # model.likelihood.noise.item(),
#         )
#     )
#     optimizer.step()

# # use the model for prediction
# # use eval() to indicate entering the prediction mode
# model.eval()
# likelihood.eval()
# with torch.no_grad(), gpytorch.settings.fast_pred_var():
#     test_x_temp = np.linspace(0, 1, num_test)
#     test_y__temp = function.f(test_x_temp)
#     test_x = torch.from_numpy(test_x_temp)
#     test_y = torch.from_numpy(test_y__temp).flatten()
#     observed_pred = likelihood(model(test_x), noise=torch.zeros(num_test))


# with torch.no_grad():
#     f, ax = plt.subplots(1, 1, figsize=(5, 4))
#     lower, upper = observed_pred.confidence_region()
#     ax.plot(train_x.numpy(), train_y.numpy(), "k*")
#     ax.plot(test_x.numpy(), test_y.numpy(), "r--")
#     ax.plot(test_x.numpy(), observed_pred.mean.numpy(), "b")
#     ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
#     # ax.set_ylim([-, 3])
#     ax.legend(["Observed Data", "True response", "Mean", "Confidence"])
#     plt.show()
