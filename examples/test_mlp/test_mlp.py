import sys

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch

# import local functions
from mfpml.mfpml.design_of_experiment.singlefideliy_samplers import LatinHyperCube
from mfpml.models.mlp import MLPModel
from mfpml.problems.singlefidelity_functions import Forrester

# user-defined parameters
num_sample = 20
# training_iter = 200
num_test = 1000
# learning_rate = 0.1
# define the design space
function = Forrester()
design_space = function.design_space

# sampling
sampler = LatinHyperCube(design_space=design_space, seed=None)
sample_x = sampler.get_samples(num_samples=num_sample)
sample_y = function.f(sample_x)
train_x = torch.from_numpy(sample_x).to(torch.float32)
train_y = torch.from_numpy(sample_y).to(torch.float32)

# get the test data
test_x_temp = sampler.get_samples(num_samples=num_test)
test_y_temp = function.f(test_x_temp)
test_x = torch.from_numpy(test_x_temp).to(torch.float32)
test_y = torch.from_numpy(test_y_temp).to(torch.float32)

# nn model
mlp_model = MLPModel(
    num_inputs=1, hiden_layers=[20, 20, 20], num_outputs=1, activation="tanh"
)
# define the loss function
mlp_model.define_loss(loss_metric="mse")
mlp_model.define_optimizer(
    optimizer="adam", learning_rate=0.001, weight_decay=0.00001
)

print(mlp_model.net)

mlp_model.train(
    train_x=train_x,
    train_y=train_y,
    test_x=test_x,
    test_y=test_y,
    design_space=design_space,
    normlize=False,
    train_epoch=10000,
    batch_size=num_sample,
    print_info=True,
)
pred_y = mlp_model.predict(test_x=mlp_model.test_x)

fig, ax = plt.subplots()
ax.plot(test_x.detach(), test_y.detach(), ".", label="function")
ax.plot(test_x.detach(), pred_y.detach(), "*", label="prediction")
ax.plot(train_x.detach(), train_y.detach(), "^", label="samples")
ax.legend()
plt.show()
