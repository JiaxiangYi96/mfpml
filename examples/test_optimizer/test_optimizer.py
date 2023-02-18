import os
import sys
import time

folder_path = "/home/yaga/Documents/GitHub/mfpml"
sys.path.insert(0, folder_path)


from mfpml.problems.sf_functions import Ackley
from mfpml.optimization.evolutionary_algorithms import DE, PSO

start_time = time.time()
function = Ackley(num_dim=10)
design_space = function._input_domain

# define the optimizer
pso_opt = DE(num_gen=1000, num_pop=50)
opt_results = pso_opt.run_optimizer(
    func=function.f,
    num_dim=function.num_dim,
    design_space=design_space,
    print_info=False,
)
print(opt_results)
print(function.optimum)
pso_opt.plot_optimization_history(figure_name="de")
# print(pso_opt.gen_best)
end_time = time.time()
print(f"time usages:{end_time-start_time}")
