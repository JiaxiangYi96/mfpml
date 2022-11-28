import numpy as np 
from collections import OrderedDict

class MFSOBO: 
    """
    Multi-fidelity single objective Bayesian optimization
    """
    def __init__(self, problem, mf_surrogate): 

        self.problem = problem
        self.mf_model = mf_surrogate

        self.iter = 0 
        self.n_hf = 0 
        self.n_lf = 0
        self.best = None 
        self.best_x = None
        self.log = OrderedDict()

    def _first_run(self, X, Y): 

        self.mf_model.train(X, Y)
        self.n_hf = X['hf'].shape[0]
        self.n_lf = X['lf'].shape[0]
        self.best = np.min(Y['hf']) 
        index = np.argmin(Y['hf'])
        self.best_x = X['hf'][index, :]
        self.log[0] = (X, Y)
        print('Initial: #eval HF: {:d}, #eval LF: {:d}, Best: {:6}'.format(self.n_hf, self.n_lf, self.best))

    def run(self, acqusition, max_iter=float('inf'), max_cost=float('inf')): 

        iter = 0
        equal_cost = 0
        while iter<max_iter and equal_cost<max_iter:
            update_x = acqusition.opt(fmin=self.best, mf_surrogate=self.mf_model, bounds=self.problem.bounds)
            update_y = self.problem(update_x)
            self.update_para(iter=iter, update_x=update_x, update_y=update_y)
            self.mf_model.update(update_x, update_y)
            self._printCurrent(iter)
            iter += 1 
            equal_cost = self.n_hf + self.n_lf / self.problem.cr


    def _printCurrent(self, iter): 
        
        print('#iter: {:d}, #eval HF: {:d}, #eval LF: {:d}, Best: {:6}'.format(iter, self.n_hf, self.n_lf, self.best))
    
    def update_para(self, iter, update_x, update_y): 

        self.iter += iter + 1
        self.log[self.iter] = (update_x, update_y)
        if update_x['hf'] is not None: 
            min_y = np.min(update_y['hf'])
            min_index = np.argmin(update_y['hf'], axis=1)
            self.n_hf += update_x['hf'].shape[0]
            if min_y < self.best: 
                self.best = min_y
                self.best_x = update_x['hf'][min_index]
        elif update_x['lf'] is not None: 
            self.n_lf += update_x['lf'].shape[0]
    

