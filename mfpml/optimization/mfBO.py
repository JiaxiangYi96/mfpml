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
        problem: any) -> None: 
        """Initialize the multi-fidelity Bayesian optimization

        Parameters
        ----------
        problem : any
            optimization problem
        """
        self.func = problem

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
        self.params['design_space'] = self.func._input_domain
        self.params['cr'] = self.func.cr

    def run_optimizer(self, 
                    mf_surrogate: any, 
                    acqusition: any, 
                    max_iter: float = float('inf'), 
                    max_cost: float = float('inf'), 
                    print_info: bool = True, 
                    resume: bool = False, 
                    **kwargs) -> None: 
        """Multi-fidelity Bayesian optimization 

        Parameters
        ----------
        mf_surrogate : any
            instance of multi-fidelity model
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
            if 'init_X' not in kwargs:
                ValueError('initial samples "init_X" is not assigned.')
            if 'init_Y' not in kwargs:
                ValueError('initial responses "init_Y" is not assigned.')
            self._first_run(mf_surrogate=mf_surrogate, X=kwargs['init_X'], Y=kwargs['init_Y'])
        iter = 0
        while iter<max_iter and self.params['cost']<max_cost:
            update_x = acqusition.query(mf_surrogate=mf_surrogate, params=self.params)
            update_y = self.func(update_x)
            self._update_para(update_x, update_y)
            mf_surrogate.update_model(update_x, update_y)
            iter += 1 
            if print_info:
                self._print_info(iter)

    def historical_plot(self, save_figure: bool = False, name: str = 'historical best observed values') -> None:
        """Plot historical figure of best observed function values

        Parameters
        ----------
        save_figure : bool, optional
            save figure or not, by default False
        name : bool, optional
            name of the figure, by default 'historical best observed values'
        """
        with plt.style.context(['ieee', 'science']):
            fig, ax = plt.subplots()
            ax.plot(range(self.iter+1), self.history_best, '-o', 
                    label="Iterative figure for best solutions")
            ax.legend()
            ax.set(xlabel=r"$Iteration$")
            ax.set(ylabel=r"$Function values$")
            if save_figure is True:
                fig.savefig(name, dpi=300)
            plt.show(block=True)
            plt.interactive(False)

    def _first_run(self, mf_surrogate: any, X: dict, Y: dict, print_info: bool = True): 
        """Initialize parameters in the Bayesian optimization

        Parameters
        ----------
        mf_surrogate : any
            instance of multi-fidelity model
        X : dict
            initial samples 
        Y : dict
            initial responses
        print_info : bool, optional
            whether to print information, by default True
        """
        mf_surrogate.train(X, Y)
        self.params['n_hf'] = X['hf'].shape[0]
        self.params['n_lf'] = X['lf'].shape[0]
        self.params['cost'] = self.params['n_hf'] + self.params['n_lf'] / self.params['cr']
        self.params['fmin'] = np.min(Y['hf'])
        index = np.argmin(Y['hf'])
        self.params['best_scheme'] = X['hf'][index, :]
        self.history_best.append(self.params['fmin'])
        self.log[0] = (X, Y)
        if print_info:
            self._print_info(0)

    def _update_para(self, update_x: dict, update_y: dict) -> None: 
        """Update parameters

        Parameters
        ----------
        update_x : dict
            update samples
        update_y : dict
            update responses
        """
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
        self.params['cost'] = self.params['n_hf'] + self.params['n_lf'] / self.params['cr']
        self.history_best.append(self.params['fmin'])

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
              f'Current optimum: {self.best_objective()}') 

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
    
