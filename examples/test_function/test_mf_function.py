# configure the environmental path
import os
import sys

# import local libraries
# path of local project
folder_path = "/home/jiaxiangyi/mfpml"
sys.path.insert(0, folder_path)
from mfpml.design_of_experiment.mf_samplers import SobolSequence
from mfpml.problems.mf_functions import *
from mfpml.utils.plot_figures import plot_mf_sampling

# define function
function = Forrester_1b()
design_space = function.design_space
## test sampling part
sampler = SobolSequence(design_space=design_space, seed=12, nested=False)
samples = sampler.get_samples(num_lf_samples=10, num_hf_samples=4)

# sampler.plot_samples(figure_name='test_sampling', save_plot=True)

# sample_x = samples["hf"].to_numpy()
sample_y = {}
sample_y["hf"] = function.hf(samples["hf"])
sample_y["lf"] = function.lf(samples["lf"])

plot_mf_sampling(samples=samples, responses=sample_y, function=function, save_figure=True)
