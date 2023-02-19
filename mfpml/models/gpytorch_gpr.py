# define gpr model via gpytorch
from sklearn.covariance import log_likelihood
import torch
import gpytorch


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_inputs, train_targets, likelihood):
        super().__init__(train_inputs, train_targets, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class StandardGPRModel:
    def __init__(self, model, likelihood, optimizer, loss_function) -> None:
        self.model = model
        self.likelihood = likelihood
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None

    def _get_likelihood(self):
        pass

    def _get_loss(self):
        pass

    def train():
        pass

    def predict():
        pass
