import sys
import torch
import numpy as np

import pickle

folder_path = "/home/yaga/Documents/GitHub/mfpml"
sys.path.insert(0, folder_path)

# import local functions
from mfpml.design_of_experiment.sf_samplers import LatinHyperCube
from mfpml.problems.sf_functions import (
    Forrester,
    Branin,
    Sasena,
    Sixhump,
    Hartman3,
    Hartman6,
    GoldPrice,
)
from mfpml.design_of_experiment.plot_figures import plot_sf_sampling
from mfpml.models.gaussian_processes import Kriging
from sklearn.metrics import r2_score, mean_squared_error, max_error


# number of repetition
num_trial = 10
num_increment = 5
record_results = np.zeros((num_increment, num_trial, 3))

for ii in range(num_increment):

    # define the tested function
    function = Branin()
    # get the number of variables
    num_dim = function.num_dim
    # user-defined parameters
    num_sample = 10 * num_dim + ii * num_dim * 5
    # define the design space
    design_space = function.design_space

    for jj in range(num_trial):
        # sampling
        sampler = LatinHyperCube(design_space=design_space, seed=None)
        sample_x = sampler.get_samples(num_samples=num_sample)
        sample_y = function.f(sample_x)
        train_x = torch.from_numpy(sample_x)
        train_y = torch.from_numpy(sample_y).flatten()

        # gpr model
        gprmodel = Kriging()
        gprmodel.train(
            train_x=train_x.clone(),
            train_y=train_y.clone(),
            design_space=function.design_space,
            normlize=True,
            training_iteration=500,
            learning_rate=0.02,
            print_info=False,
        )
        # prediction
        num_test = 1000
        test_x = sampler.get_samples(num_samples=num_test)
        test_y = function.f(test_x.copy()).flatten()
        test_x = torch.from_numpy(test_x)
        # pred_mean = gprmodel.predict(test_x=test_x.clone()).mean.numpy()
        pred = gprmodel.predict(test_x=test_x.clone())
        pred_mean = pred.mean.numpy()

        # calculate different metrics to record the accuracy of the prediction
        mse = mean_squared_error(pred_mean, test_y)
        r2 = r2_score(pred_mean, test_y)
        mae = max_error(pred_mean, test_y)

        # record the results
        record_results[ii, jj, :] = np.array([mse, r2, mae])

# save results
results = {"results": record_results}
with open("results.pickle", "wb") as f:
    pickle.dump(results, f)
