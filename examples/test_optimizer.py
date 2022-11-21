import os
import sys

from collections import OrderedDict

from mfpml.design_of_experiment.design_space import CreateDesignSpace
from mfpml.design_of_experiment.LantinHypeCube import LatinHyperCube
from mfpml.problems.mf_functions import Forrester
from mfpml.optimization import DE

function = Forrester()
design_space = function.design_space

bound = [(function.low_bound[0], function.high_bound[0])]

results = DE().optimize(fun=function, design_space=design_space)
