import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from collections import OrderedDict

class MFSOBO: 
    """
    Multi-fidelity single objective Bayesian optimization
    """
    def __init__(
        self,
        mf_surrogate) -> None: 
        """Initialize the multi-fidelity Bayesian optimization

        Parameters
        ----------
        mf_surrogate : _type_
            _description_
        """
        self.problem = problem
        self.mf_model = mf_surrogate

    def _
        self.iter = 0 
        self.n_hf = 0 
        self.n_lf = 0
        self.best = None
        self.best_history = []
        self.best_x = None
        self.log = OrderedDict()

    def run(self, acqusition, max_iter=float('inf'), max_cost=float('inf')): 

        iter = 0
        equal_cost = 0
        while iter<max_iter and equal_cost<max_cost:
            update_x = acqusition.opt(fmin=self.best, mf_surrogate=self.mf_model, bounds=self.problem.bounds)
            update_y = self.problem(update_x)
            self.update_para(iter=iter, update_x=update_x, update_y=update_y)
            self.mf_model.update(update_x, update_y)
            self._printCurrent(iter)
            iter += 1 
            equal_cost = self.n_hf + self.n_lf / self.problem.cr

    def plotIterativeBest(self, save_figure=False): 
        fig, ax = plt.subplots()
        ax.plot(range(self.iter+1), self.best_history, label="Iterative figure for best solutions")
        ax.legend()
        ax.set(xlabel=r"$iteration$")
        ax.set(ylabel=r"$function values$")
        if save_figure is True:
            fig.savefig(self.__class__.__name__, dpi=300)
        plt.show()

    def _first_run(self, X: dict, Y: dict, print_info: bool = True): 
        """Initialize parameters in the Bayesian optimization

        Parameters
        ----------
        X : dict
            dict with two keys, 'hf' contains np.ndarray of 
            high-fidelity sample points and 'lf' contains 
            low-fidelity
        Y : dict
            dict with two keys, 'hf' contains high-fidelity
            responses and 'lf' contains low-fidelity ones
        print_info : bool, optional
            whether to print information, by default True
        """
        self.mf_model.train(X, Y)
        self.n_hf = X['hf'].shape[0]
        self.n_lf = X['lf'].shape[0]
        self.best = np.min(Y['hf'])
        self.best_history.append(self.best)
        index = np.argmin(Y['hf'])
        self.best_x = X['hf'][index, :]
        self.log[0] = (X, Y)
        print('Initial: #eval HF: {:d}, #eval LF: {:d}, Best: {:6}'.format(self.n_hf, self.n_lf, self.best))

    def _print_info(self, iter: int) -> None: 
        """Print optimization information

        Parameters
        ----------
        iter : int
            current iteration
        """
        print(f'Iteration: {iter}, '
              f'Eval HF: {self._get_num_hf()}, '
              f'Eval LF: {self._get_num_lf()}, '
              f'Current optimum: {self.best_obj()}') 

    def _get_num_hf(self) -> int: 
        """Return the number of high-fidelity samples

        Returns
        -------
        num_hf: int
            number of high-fidelity samples
        """
        return self.n['hf']

    def _get_num_lf(self) -> int: 
        """Return the number of low-fidelity samples

        Returns
        -------
        num_lf: int
            number of low-fidelity samples
        """
        return self.n['lf']
    
    def update_para(self, iter, update_x, update_y): 

        self.iter += 1
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
        self.best_history.append(self.best)

