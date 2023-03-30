import os
import sys
import time

from mfpml.optimization.evolutionary_algorithms import DE, PSO
from mfpml.problems.sf_functions import Ackley

function = Ackley(num_dim=3)
design_space = function._input_domain

# define the optimizer
pso_opt = DE(num_gen=100, num_pop=50)
opt_results, best_y, best_x = pso_opt.run_optimizer(
    func=function.f,
    num_dim=function.num_dim,
    design_space=design_space,
    save_step_results=True,
    print_info=False,
    stopping_error=0.1,
)
print(best_x, best_y, opt_results)
print(function.optimum)
pso_opt.plot_optimization_history(save_figure=False)
