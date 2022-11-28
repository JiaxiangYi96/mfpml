import os
import sys
from unittest import result

folder_path = "/home/jiaxiangyi/mfpml"
sys.path.insert(0, folder_path)


from mfpml.problems.sf_functions import *
from mfpml.utils.pso import PSO

function = Ackley(num_dim=10)
design_space = function._input_domain
# print(design_space)
# print(function.optimum)
# define the optimizer
pso_opt = PSO(num_gen=200, num_pop=200)
opt_results = pso_opt.run_optimizer(
    func=function.f, num_dim=function.num_dim, design_space=design_space, print_info=True
)
print(opt_results)
print(function.optimum)
pso_opt.plot_optimization_history()
# print(pso_opt.gen_best)
