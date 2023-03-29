import re
from turtle import st
import torch
from torch import nn
from torch.utils import data


class MLP(nn.Module):
    def __init__(
        self,
        num_inputs=1,
        hiden_layers=[20, 20, 20],
        num_outputs=1,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        self.num_inputs = num_inputs
        self.num_hidden = len(hiden_layers)
        self.num_outputs = num_outputs
        self.hiden_layers = hiden_layers
        self.activation = activation
        #
        self.net = self._create_nn_architecture()

    def forward(self, x):
        return self.net(x)

    def _create_nn_architecture(self):

        # create the first layer
        layers = nn.Sequential(
            nn.Flatten(), nn.Linear(self.num_inputs, self.hiden_layers[0])
        )
        layers.append(self._get_activation())
        for ii in range(1, self.num_hidden):
            layers.append(
                nn.Linear(self.hiden_layers[ii - 1], self.hiden_layers[ii])
            )
            layers.append(self._get_activation())
        # add the last layer
        layers.append(nn.Linear(self.hiden_layers[-1], self.num_outputs))
        return layers

    def _get_activation(self):
        if self.activation == "relu":
            return nn.ReLU()
        elif self.activation == "tanh":
            return nn.Tanh()
        else:
            raise ValueError(
                "the activation is not implemented in this framework!"
            )


class MLPModel(MLP):
    def __init__(
        self,
        num_inputs=1,
        hiden_layers=[20, 20, 20],
        num_outputs=1,
        activation: str = "relu",
    ) -> None:
        super().__init__(num_inputs, hiden_layers, num_outputs, activation)

    @staticmethod
    def load_array(data_arrays, batch_size, is_train=True):
        """Construct a PyTorch data iterator."""
        dataset = data.TensorDataset(*data_arrays)
        return data.DataLoader(dataset, batch_size, shuffle=is_train)

    def train(
        self,
        train_x,
        train_y,
        test_x,
        test_y,
        design_space,
        normlize,
        train_epoch,
        batch_size,
        print_info,
    ):
        self.normlize = normlize
        self.design_space = design_space
        if self.normlize is True:
            self.train_x = self._normalize_data(train_x)
            self.test_x = self._normalize_data(test_x)
        else:
            self.train_x = train_x
            self.test_x = test_x

        self.train_y = train_y
        # create data iterater
        data_iter = self.load_array((self.train_x, self.train_y), batch_size)
        next(iter(data_iter))
        # apply initial weights
        self.net.apply(self._init_weight)

        # begin the training process
        for epoch in range(train_epoch):
            for batch_x, batch_y in data_iter:
                loss = self.loss_function(self.net(batch_x), batch_y)
                self.optimizer.zero_grad()  # sets gradients to zero
                loss.backward(retain_graph=True)  # back propagation
                self.optimizer.step()  # parameter update
            # calculate training loss
            train_loss = self._loss_calulate(x=self.train_x, y=train_y)
            # calculate test loss
            test_loss = self._loss_calulate(x=self.test_x, y=test_y)
            if print_info is True:
                if epoch % 500 == 0:
                    print(
                        f"epoch {epoch + 1}, training loss {train_loss:f}, test loss {test_loss:f}"
                    )

    def predict(self, test_x: torch.Tensor):

        self.net.eval()
        return self.net(test_x)

    def _normalize_data(self, x: torch.Tensor) -> torch.Tensor:
        """normalize the train inputs so that there are located in the space
        from zero to one
        """
        for i, bounds in enumerate(self.design_space.values()):
            x[:, i] = (x[:, i] - bounds[0]) / (bounds[1] - bounds[0])
        return x

    def _loss_calulate(self, x: torch.Tensor, y: torch.Tensor):
        if x is None:
            return None
        else:
            return self.loss_function(self.net(x), y)

    def define_loss(self, loss_metric="mse"):
        if loss_metric == "mse":
            self.loss_function = nn.MSELoss()
        else:
            pass

    def define_optimizer(
        self,
        optimizer: str = "adam",
        learning_rate: float = 0.01,
        weight_decay: float = 0,
    ):
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                self.net.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        else:
            raise NameError("Optimizer if not defined \n")

    @staticmethod
    def _init_weight(m):
        """initial weight

        Parameters
        ----------
        m : nn.Module
            neural network architecture (layer)
        """

        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.05)
