# configure the environmental path
import os
import sys

from collections import OrderedDict

from MfPml.DoE.DesignSpace import CreateDesignSpace
from MfPml.DoE.LantinHypeCube import LatinHyperCube
from MfPml.Functions.MultiFidelityUnconstrained import Forrester

function = Forrester()

design_space = function.design_space
# print(design_space)
# define the number of samples
num_points = 10
name_outputs = ["y"]
doe_sampler = LatinHyperCube()
samples = doe_sampler.sampling(num_samples=num_points,
                               design_space=design_space,
                               out_names=name_outputs,
                               seed=123456)

samples['y'] = function(samples['x'])

function.plot_function(with_low_fidelity=True)
# print(function.get_optimum())
# print(function.get_optimum_variable())
# print(function.get_design_space())
