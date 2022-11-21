from mfpml.problems.mf_functions import Forrester
from mfpml.Models.GaussianProcessRegressor import SklearnGPRmodel
from mfpml.design_of_experiment.LantinHypeCube import LatinHyperCube
from mfpml.optimization.SFBayesOpt import SFBayesOpt

function = Forrester()
design_space = function.design_space

## Sampling
doe_sampler = LatinHyperCube()
samples = doe_sampler.sampling(num_samples=6,
                               design_space=design_space,
                               seed=123456)

samples['y'] = function(samples['inputs'].to_numpy())

#
# plot_1d_sampling(data=samples, function=function, save=True)

GPmodel = SklearnGPRmodel()
GPmodel.train_model(samples['x'].to_numpy(), samples['y'].to_numpy(), num_dim=1)
# GPmodel.train_model(GPmodel.x, GPmodel.y, num_dim=GPmodel.num_dim)
# fit the models

# lcb = LCB()
# cc = lcb(x=[0.5, 0.4], models=GPmodel)
# print(cc)

#
# iteration = 0
#
# while iteration < 10:
#     GPmodel.fit(samples['x'].to_numpy(), samples['y'].to_numpy(), num_dim=1)
#     BayesOpt = SFBayesOpt(models=GPmodel,
#                           acquisition='LCB',
#                           design_space=design_space,
#                           optimizer='DE')
#     next_loc = BayesOpt.suggested_location()
#     print(BayesOpt.best_design_scheme())
#     print(BayesOpt.best_objective())
#     # refine the GPR models
#     next_y = function(next_loc)
#     samples.loc[len(samples.index)] = [next_loc, next_y]
#     iteration = iteration + 1
#
# plot_1d_model_prediction(data=samples,
#                          models=GPmodel,
#                          function=function,
#                          save=True)

BayesOpt = SFBayesOpt(model=GPmodel,
                      acquisition='PI',
                      design_space=design_space,
                      optimizer='DE',
                      Analytical=True)
results = BayesOpt.run_optimizer(fun=function,
                                 iteration=20)
BayesOpt.historical_plot()

# plot_1d_model_prediction(data=samples,
#                          models=BayesOpt.models,
#                          function=function,
#                          save=True)
