from collections import OrderedDict
from typing import Any

import numpy as np
from matplotlib import pyplot as plt

from ..models.kriging import Kriging


class BayesOpt(object):

    def __init__(self, problem: Any) -> None:

        # problem
        self.problem = problem

    def change_problem(self, problem: Any) -> None:
        self.problem = problem

    def _initialization(self) -> None:
        """initialization for single fidelity bayesian optimization
        """
        self.iter = 0
        self.best = None
        self.history_best = []
        self.best_x = None
        self.log = OrderedDict()
        self.params = {}
        self.params['design_space'] = self.problem.input_domain

    # run optimizer
    def run_optimizer(self,
                      init_x: np.ndarray,
                      init_y: np.ndarray,
                      surrogate: Kriging,
                      acquisition: Any,
                      max_iter: int = 10,
                      print_info: bool = True,
                      optimum: float = None,
                      stopping_error: float = 1.0,
                      ) -> tuple[np.ndarray, np.ndarray]:

        # get additional info for optimization
        self.known_optimum = optimum
        self.stopping_error = stopping_error
        self.surrogate = surrogate
        # initialization
        self._initialization()
        self._first_run(surrogate=self.surrogate,
                        sample_x=init_x,
                        sample_y=init_y,
                        print_info=print_info)

        # main loop
        iter = 0
        error = self.__update_error()
        while iter < max_iter and error >= self.stopping_error:
            # update the next point
            update_x = acquisition.query(surrogate=self.surrogate,
                                         params=self.params)
            update_x = np.atleast_2d(update_x)
            # get update y
            update_y = self.problem.f(update_x)
            # update paras
            self._update_para(update_x=update_x, update_y=update_y)
            # update surrogate
            self.surrogate.update_model(update_x=update_x,
                                        update_y=update_y)
            iter = iter + 1
            # print info
            if print_info:
                self.__print_info(iter=iter)
            # update error
            error = self.__update_error()

        return self.best, self.best_x

    def _update_para(self, update_x: np.ndarray,
                     update_y: np.ndarray) -> None:

        self.iter = self.iter + 1
        # update log
        self.log[self.iter] = (update_x, update_y)
        # update best
        min_y = np.min(update_y)
        min_index = np.argmin(update_y, axis=1)
        #  update number samples
        self.params['num_samples'] += update_x.shape[0]
        # update best
        if min_y < self.params['fmin']:
            self.params['fmin'] = min_y
            self.params['best_scheme'] = update_x[min_index, :]
        self.best = self.params['fmin']
        self.best_x = self.params['best_scheme']
        # update history best
        self.history_best.append(self.params['fmin'])

    def __update_error(self) -> Any | float:
        """update error between the known optimum and the current optimum

        Returns
        -------
        float
            error
        """
        if self.known_optimum is not None and self.known_optimum != 0:
            error = np.abs((self.best - self.known_optimum) /
                           self.known_optimum)
        elif self.known_optimum == 0.0:
            error = np.abs(self.best - self.known_optimum)
        else:
            error = 1.0

        return error

    def plot_optimization_history(self, save_fig: bool = False,
                                  fig_name="bo_history.png", **kwarg) -> None:
        """plot optimization history
        """

        _, ax = plt.subplots(**kwarg)
        ax.plot(self.history_best, 'b-o', lw=2, label="optimum")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$f(x)$")
        ax.legend(loc="best")
        ax.grid()
        if save_fig:
            plt.savefig(fig_name, dpi=300)
        plt.show()

    def _first_run(self, surrogate: Any,
                   sample_x: np.ndarray,
                   sample_y: np.ndarray,
                   print_info: bool = True,
                   ) -> None:
        # train the surrogate model
        surrogate.train(sample_x=sample_x, sample_y=sample_y)
        self.params['num_samples'] = sample_x.shape[0]
        self.params['fmin'] = sample_y.min()
        self.params['best_scheme'] = sample_x[sample_y.argmin(), :]
        self.history_best.append(sample_y.min())
        self.best = self.params['fmin']
        self.best_x = self.params['best_scheme']
        self.log[0] = (sample_x, sample_y)
        if print_info:
            self.__print_info(iter=0)

    def __print_info(self, iter: int) -> None:
        """print optimum information to screen

        Parameters
        ----------
        iter : int
            iteration number
        """
        # print the best y of current iteration
        print(f"iter:{iter} =====================================")
        print(f"best_y: {self.best:4f}")
        print(f"best_x: {self.best_x}")
        if self.stopping_error < 1.0:
            print(f"error: {self.__update_error():4f}")
