# define gpr model via gpytorch
from cgi import test
from tkinter.tix import Tree
import numpy as np
import torch
import gpytorch

# basic class of standard Gasussian process regression model
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


class Kriging:
    """This is the standard gaussian process regression model, in which the fixnoise exact gp model is used.
    In this case, the prediction mean of the standard GP model will cross the actual samples
    """

    def __init__(
        self,
        optimizer: str = "Adam",
        loss_function: str = "mll",
    ) -> None:
        """Initilization of Kriging model

        Parameters
        ----------
        optimizer : str, optional
            string to indicate the optimizer , by default "Adam"
        loss_function : str, optional
            string to indicate the loss function, by default "mll"
        """
        # define components of gpr model
        self.model = None
        self.likelihood = None
        self.optimizer = None
        self.loss = None

        # string to indicate different components of gpr model
        self.optimizer_ = optimizer
        self.loss_function_ = loss_function

        # data information
        self.train_x = None
        self.train_y = None
        self.test_x = None

    def train(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        design_space: dict,
        normlize: bool = False,
        training_iteration: int = 200,
        learning_rate: float = 0.1,
        print_info: bool = False,
    ):
        """training the Kriging model

        Parameters
        ----------
        train_x : torch.Tensor
            training inputs
        train_y : torch.Tensor
            traning outputs
        training_iteration : int, optional
            traning iteration, by default 200
        learning_rate : float, optional
            learning rate, by default 0.1
        print_info : bool, optional
            print the traning information or not , by default False
        """
        self.normlize = normlize
        self.design_space = design_space
        if self.normlize is True:
            self.train_x = self._normalize_data(train_x)
        else:
            self.train_x = train_x

        self.train_y = train_y

        # number of samples
        self.num_samples = self.train_x.size(dim=0)

        # initialize gpr model
        self.likelihood = self._likelihood(num_samples=self.num_samples)
        self.model = ExactGPModel(
            train_inputs=self.train_x,
            train_targets=train_y,
            likelihood=self.likelihood,
        )
        # open the training mode
        self.model.train()
        self.likelihood.train()

        # get the optimizer
        self.optimizer = self._optimizer(learning_rate=learning_rate)
        self.loss = self._loss()

        # set up the training process
        for i in range(training_iteration):
            # Zero gradients from previous iteration
            self.optimizer.zero_grad()
            # Output from model
            output = self.model(self.train_x)
            # Calc loss and backprop gradients
            loss = -self.loss(output, train_y)
            loss.backward()
            if print_info is True:
                print(
                    "Iter %d/%d - Loss: %.3f   lengthscale: %.3f "
                    % (
                        i + 1,
                        training_iteration,
                        loss.item(),
                        self.model.covar_module.base_kernel.lengthscale.item(),
                    )
                )
            self.optimizer.step()

    def predict(self, test_x: torch.Tensor) -> np.ndarray:
        """prediction

        Parameters
        ----------
        test_x : torch.Tensor
            test inputs

        Returns
        -------
        np.ndarray
            likelihood of prediction
        """
        self.model.eval()
        self.likelihood.eval()
        # get the number of the test points
        num_test = test_x.size(dim=0)
        if self.normlize is True:
            self.test_x = self._normalize_data(x=test_x)
        else:
            self.test_x = test_x

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(
                self.model(self.test_x), noise=torch.zeros(num_test)
            )
        # get the prediction for
        return observed_pred

    def _normalize_data(self, x: torch.Tensor) -> torch.Tensor:
        """normalize the train inputs so that there are located in the space
        from zero to one
        """
        for i, bounds in enumerate(self.design_space.values()):
            x[:, i] = (x[:, i] - bounds[0]) / (bounds[1] - bounds[0])
        return x

    def _likelihood(self, num_samples: int) -> gpytorch.likelihoods:
        """define the likelihood function for training gpr model

        Parameters
        ----------
        num_samples : int
            number of sampples

        Returns
        -------
        _type_
            likelihood of gaussian process regression model
        """
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
            noise=torch.zeros(num_samples),
            learn_additional_noise=False,
        )

        return likelihood

    def _loss(self):
        """get the loss function

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        NameError
            if the name is not correct, then report an error
        """
        if self.loss_function_ == "mll":
            loss = gpytorch.mlls.ExactMarginalLogLikelihood(
                likelihood=self.likelihood, model=self.model
            )
        else:
            raise NameError("the loss function is not defined \n")
        return loss

    def _optimizer(self, learning_rate: float = 0.1):
        """get optimizer

        Parameters
        ----------
        learning_rate : float
            learning rate, by default 0.1

        Returns
        -------
        _type_
            _description_
        """

        if self.optimizer_ == "Adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=learning_rate
            )
        elif self.optimizer_ == "LBFGS":
            optimizer = torch.optim.LBFGS(
                self.model.parameters(), lr=learning_rate
            )
        else:
            raise NameError("the optimizer is not defined \n")

        return optimizer


class GPR:
    """This is the standard gpr model, which can handle noise outputs"""

    def __init__(
        self,
        optimizer: str = "Adam",
        loss_function: str = "mll",
    ) -> None:
        # define components of gpr model
        self.model = None
        self.likelihood = None
        self.optimizer = None
        self.loss = None

        # string to indicate different components of gpr model
        self.optimizer_ = optimizer
        self.loss_function_ = loss_function

        # data information
        self.train_x = None
        self.train_y = None
        self.test_x = None

    def train(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        training_iteration: int = 200,
        learning_rate: float = 0.1,
        print_info: bool = False,
    ):

        self.train_x = train_x
        self.train_y = train_y
        # number of samples
        self.num_samples = self.train_x.size(dim=0)
        # TODO normalize the inputs and outputs

        # initialize gpr model
        self.likelihood = self._likelihood()
        self.model = ExactGPModel(
            train_inputs=train_x,
            train_targets=train_y,
            likelihood=self.likelihood,
        )
        # open the training mode
        self.model.train()
        self.likelihood.train()

        # get the optimizer
        self.optimizer = self._optimizer(learning_rate=learning_rate)
        self.loss = self._loss()

        # set up the training process
        for i in range(training_iteration):
            # Zero gradients from previous iteration
            self.optimizer.zero_grad()
            # Output from model
            output = self.model(train_x)
            # Calc loss and backprop gradients
            loss = -self.loss(output, train_y)
            loss.backward()
            if print_info is True:
                print(
                    "Iter %d/%d - Loss: %.3f   lengthscale: %.3f "
                    % (
                        i + 1,
                        training_iteration,
                        loss.item(),
                        self.model.covar_module.base_kernel.lengthscale.item(),
                    )
                )
            self.optimizer.step()

    def predict(self, test_x: torch.Tensor) -> np.ndarray:
        self.model.eval()
        self.likelihood.eval()
        # get the number of the test points
        num_test = test_x.size(dim=0)
        self.test_x = test_x

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(
                self.model(test_x), noise=torch.zeros(num_test)
            )
        # get the prediction for
        return observed_pred

    def _normalize_data(self):
        """normalize the train inputs so that there are located in the space
        from zero to one
        """

        pass

    def _likelihood(self) -> gpytorch.likelihoods:
        """define the likelihood function for training gpr model

        Parameters
        ----------
        num_samples : int
            number of sampples

        Returns
        -------
        _type_
            likelihood of gaussian process regression model
        """
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        return likelihood

    def _loss(self):
        if self.loss_function_ == "mll":
            loss = gpytorch.mlls.ExactMarginalLogLikelihood(
                likelihood=self.likelihood, model=self.model
            )
        else:
            raise NameError("the loss function is not defined \n")
        return loss

    def _optimizer(self, learning_rate: float = 0.1):
        """get optimizer

        Parameters
        ----------
        learning_rate : float
            learning rate, by default 0.1

        Returns
        -------
        _type_
            _description_
        """

        if self.optimizer_ == "Adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=learning_rate
            )
        else:
            raise NameError("the optimizer is not defined \n")

        return optimizer
