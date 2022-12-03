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
        func: any) -> None: 
        """Initialize the multi-fidelity Bayesian optimization

        Parameters
        ----------
        func : any
            optimization problem
        """
        self.func = func

    def change_func(self, func: any) -> None: 
        """Change the function for optimization

        Parameters
        ----------
        func : any
            optimization problem
        """
        self.func = func

    def _initialization(self) -> None:
        """Initialize parameters
        """
        self.iter = 0 
        self.params = {}
        self.best = None
        self.history_best = []
        self.best_x = None
        self.log = OrderedDict()
        self.params['design_space'] = self.func._input_domain()

    def run_optimizer(self, 
                    acqusition: any, 
                    max_iter: float = float('inf'), 
                    max_cost: float = float('inf'), 
                    print_info: bool = True, 
                    resume: bool = False) -> None: 
        """Multi-fidelity Bayesian optimization 

        Parameters
        ----------
        acqusition : any
            instance of multi-fidelity acqusition
        max_iter : float, optional
            stop condition of iteration, by default float('inf')
        max_cost : float, optional
            stop condition of cost, by default float('inf')
        print_info : bool, optional
            whether to print information during the optimization 
            process, by default True
        resume : bool, optional
            whether to proceed optimization with the last run
            , by default True
        """
        if not resume: 
            self._initialization()
        iter = 0
        equal_cost = 0
        while iter<max_iter and equal_cost<max_cost:
            update_x = acqusition.opt(fmin=self.best, mf_surrogate=self.mf_model, bounds=self.problem.bounds)
            update_y = self.problem(update_x)
            self.update_para(update_x, update_y)
            self.mf_model.update(update_x, update_y)
            iter += 1 
            if print_info:
                self._print_info(iter)
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

    def first_run(self, X: dict, Y: dict, print_info: bool = True): 
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
        return self.params['n_hf']

    def _get_num_lf(self) -> int: 
        """Return the number of low-fidelity samples

        Returns
        -------
        num_lf: int
            number of low-fidelity samples
        """
        return self.params['n_lf']

    def best_objective(self) -> float:
        """Get the best design of current iteration

        Returns
        -------
        float
            Minimum observed objective
        """
        return self.params['fmin']

    def best_design_scheme(self) -> np.ndarray:
        """Get the best design scheme of current iteration

        Returns
        -------
        np.ndarray
            Best design scheme
        """
        return self.params['best_scheme']
    
    def _update_para(self, update_x, update_y): 

        self.iter += 1
        self.log[self.iter] = (update_x, update_y)
        if update_x['hf'] is not None: 
            min_y = np.min(update_y['hf'])
            min_index = np.argmin(update_y['hf'], axis=1)
            self.params['n_hf'] += update_x['hf'].shape[0]
            if min_y < self.params['fmin']: 
                self.best = min_y
                self.params['fmin'] = min_y
                self.params['best_scheme'] = update_x['hf'][min_index]
        elif update_x['lf'] is not None: 
            self.params['n_lf'] += update_x['lf'].shape[0]
        self.history_best.append(self.params['fmin'])

