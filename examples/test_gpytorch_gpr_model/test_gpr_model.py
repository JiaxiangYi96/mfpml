import sys
import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt

folder_path = "/home/yaga/Documents/GitHub/mfpml"
sys.path.insert(0, folder_path)

# import local functions
from mfpml.models.gpytorch_gpr import ExactGPModel
from mfpml.design_of_experiment.sf_samplers import RandomSampler
from mfpml.problems.sf_functions import Forrester
from mfpml.design_of_experiment.plot_figures import plot_sf_sampling

# user-defined parameters
num_sample = 10
training_iter = 2000
num_test = 1000
learning_rate = 0.1
# define the design space
function = Forrester()
design_space = function.design_space

# sampling
sampler = RandomSampler(design_space=design_space, seed=111)
sample_x = sampler.get_samples(num_samples=num_sample)
sample_y = function.f(sample_x)
# plot_sf_sampling(
#     samples=sample_x, responses=sample_y, function=function, save_figure=False
# )
train_x = torch.from_numpy(sample_x)
train_y = torch.from_numpy(sample_y).flatten()

# define the gpr model
likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
    noise=torch.zeros(num_sample),
    learn_additional_noise=False,
)
model = ExactGPModel(
    train_inputs=train_x, train_targets=train_y, likelihood=likelihood
)
# train the model
model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# bulid the loss function
mll = gpytorch.mlls.ExactMarginalLogLikelihood(
    likelihood=likelihood, model=model
)
# set up the training process
for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, train_y)
    loss.backward()
    print(
        "Iter %d/%d - Loss: %.3f   lengthscale: %.3f "
        % (
            i + 1,
            training_iter,
            loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            # model.likelihood.noise.item(),
        )
    )
    optimizer.step()

# use the model for prediction
# use eval() to indicate entering the prediction mode
model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x_temp = np.linspace(0, 1, num_test)
    test_y__temp = function.f(test_x_temp)
    test_x = torch.from_numpy(test_x_temp)
    test_y = torch.from_numpy(test_y__temp).flatten()
    observed_pred = likelihood(model(test_x), noise=torch.zeros(num_test))

with torch.no_grad():
    f, ax = plt.subplots(1, 1, figsize=(5, 4))
    lower, upper = observed_pred.confidence_region()
    ax.plot(train_x.numpy(), train_y.numpy(), "k*")
    ax.plot(test_x.numpy(), test_y.numpy(), "r--")
    ax.plot(test_x.numpy(), observed_pred.mean.numpy(), "b")
    ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    # ax.set_ylim([-, 3])
    ax.legend(["Observed Data", "True response", "Mean", "Confidence"])
    plt.show()
