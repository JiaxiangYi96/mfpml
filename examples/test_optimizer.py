import os
import sys
import time

folder_path = "/home/jiaxiangyi/mfpml"
sys.path.insert(0, folder_path)


from mfpml.problems.sf_functions import *
from mfpml.utils.pso import PSO

start_time = time.time()
function = Ackley(num_dim=10)
design_space = function._input_domain
# print(design_space)
# print(function.optimum)
# define the optimizer
pso_opt = PSO(num_gen=50, num_pop=50)
opt_results = pso_opt.run_optimizer(
    func=function.f, num_dim=function.num_dim, design_space=design_space, print_info=False
)
print(opt_results)
print(function.optimum)
# pso_opt.plot_optimization_history()
# print(pso_opt.gen_best)
end_time = time.time()
print(f"time usages:{end_time-start_time}")
