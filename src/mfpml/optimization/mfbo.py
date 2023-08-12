# import
from collections import OrderedDict
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


class mfBayesOpt:
    """
    Multi-fidelity single objective Bayesian optimization
    """

    def __init__(self, problem: Any) -> None:
        """Initialize the multi-fidelity Bayesian optimization

        Parameters
        ----------
        problem : Any
            optimization problem
        """
        self.problem = problem

    def change_problem(self, problem: Any) -> None:
        """Change the function for optimization

        Parameters
        ----------
        problem : Any
            optimization problem
        """
        self.problem = problem

    def _initialization(self) -> None:
        """Initialize parameters
        """
        self.iter = 0
        self.best = None
        self.history_best = []
        self.best_x = None
        self.log = OrderedDict()
        self.params = {}
        self.params['design_space'] = self.problem._input_domain
        self.params['cr'] = self.problem.cr

    def run_optimizer(self,
                      init_x: dict,
                      init_y: dict,
                      mf_surrogate: Any,
                      acquisition: Any,
                      max_iter: float = float('inf'),
                      max_cost: float = float('inf'),
                      print_info: bool = True,
                      resume: bool = False,
                      **kwargs) -> None:
        """Multi-fidelity Bayesian optimization

        Parameters
        ----------
        mf_surrogate : Any
            instance of multi-fidelity model
        acquisition : Any
            instance of multi-fidelity acquisition
        max_iter : float, optional
            stop condition of iteration, by default float('inf')
        max_cost : float, optional
            stop condition of cost, by default float('inf')
        print_info : bool, optional
            whether to print information during the optimization
            process, by default True
        resume : bool, optional
            whether to proceed optimization with the last run
            , by default False
        init_x : np.ndarray, optional
            initial samples, by default None
        init_y : np.ndarray, optional

        """
        if not resume:
            self._initialization()
            self._first_run(mf_surrogate=mf_surrogate,
                            init_x=init_x, init_y=init_y)
        iter = 0
        while iter < max_iter and self.params['cost'] < max_cost:
            # get the next point to evaluate
            update_x = acquisition.query(
                mf_surrogate=mf_surrogate, params=self.params)
            update_y = self.problem(update_x)
            # update mf bo info
            self._update_para(update_x, update_y)
            # update surrogate model
            mf_surrogate.update_model(update_x, update_y)
            iter += 1
            if print_info:
                self._print_info(iter)

    def historical_plot(self, save_figure: bool = False,
                        name: str = 'historical_best.png', **kwarg) -> None:
        """Plot historical figure of best observed function values

        Parameters
        ----------
        save_figure : bool, optional
            save figure or not, by default False
        name : bool, optional
            name of the figure, by default 'historical_best.png'
        """
        # with plt.style.context(['ieee', 'science']):
        fig, ax = plt.subplots(**kwarg)
        ax.plot(range(self.iter+1), self.history_best, '-o', color='b',
                label="historical_best")
        ax.grid()
        ax.legend(loc='best')
        ax.set(xlabel=r"$Iter$")
        ax.set(ylabel=r"$f(x)$")
        if save_figure is True:
            fig.savefig(name, dpi=300, bbox_inches='tight')
        plt.show()

    def _first_run(self, mf_surrogate: Any,
                   init_x: dict,
                   init_y: dict,
                   print_info: bool = True) -> None:
        """Initialize parameters in the Bayesian optimization

        Parameters
        ----------
        mf_surrogate : any
            instance of multi-fidelity model
        init_x : dict
            initial samples
        init_y : dict
            initial responses
        print_info : bool, optional
            whether to print information, by default True
        """
        mf_surrogate.train(init_x, init_y)
        self.params['n_hf'] = init_x['hf'].shape[0]
        self.params['n_lf'] = init_x['lf'].shape[0]
        self.params['cost'] = self.params['n_hf'] + \
            self.params['n_lf'] / self.params['cr']
        self.params['fmin'] = np.min(init_y['hf'])
        index = np.argmin(init_y['hf'])
        self.params['best_scheme'] = init_x['hf'][index, :]
        self.history_best.append(self.params['fmin'])
        self.log[0] = (init_x, init_y)
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
        # update log
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
        self.params['cost'] = self.params['n_hf'] + \
            self.params['n_lf'] / self.params['cr']
        self.history_best.append(self.params['fmin'])

    def _print_info(self, iter: int) -> None:
        """Print optimization information

        Parameters
        ----------
        iter : int
            current iteration
        """
        print('===========================================')
        print(f'iter: {iter}, '
              f'eval HF: {self._get_num_hf()}, '
              f'eval LF: {self._get_num_lf()}, '
              f'found optimum: {self.best_objective():.5f}, ')

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
