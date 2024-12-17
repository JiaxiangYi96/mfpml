from abc import ABC
from typing import Any, List, Tuple

import numpy as np

from ..design_of_experiment import LatinHyperCube as LHS
from ..models.gaussian_process import GaussianProcessRegression as Kriging
from ..problems.functions import Functions
from .sf_uncons_acqusitions import SFUnConsAcq


class BayesUnConsOpt(ABC):
    """Bayesian optimization for single fidelity unconstrained problems
    """

    def __init__(self,
                 problem: Functions,
                 acquisition: SFUnConsAcq,
                 num_init: int,
                 verbose: bool = False,
                 seed: int = 1996) -> None:

        self.problem = problem
        self.acquisition = acquisition
        self.num_init = num_init
        self.seed = seed
        self.verbose = verbose
        # set random seed
        np.random.seed(self.seed)

        # record optimization info
        self.best: float = np.inf
        self.best_x: np.ndarray = None
        self.history_best: List[float] = []
        self.history_best_x: List[np.ndarray] = []
        self.known_optimum: float = None
        self.stopping_error: float = 1.0

        # initialization
        self._initialization()

    def run_optimizer(self,
                      max_iter: int = 10,
                      stopping_error: float = 1.0,
                      ) -> Tuple[float, np.ndarray]:

        # get additional info for optimization
        if self.problem.optimum is not None:
            self.known_optimum = self.problem.optimum
        else:
            self.known_optimum = None

        # error-based stopping criterion
        self.stopping_error = stopping_error
        # main loop
        iteration = 0
        error = self.__update_error()
        print(f"error: {error}")
        while iteration < max_iter and error >= self.stopping_error:
            # update the next point
            update_x = self.acquisition.query(surrogate=self.surrogate)
            update_x = np.atleast_2d(update_x)
            # get update y
            update_y = self.problem.f(update_x)
            # update samples
            self.sample = np.vstack((self.sample, update_x))
            self.response = np.vstack((self.response, update_y))
            # update surrogate
            self.surrogate.train(X=self.sample,
                                 Y=self.response)
            # update paras
            self._update_para()
            iteration = iteration + 1
            if self.verbose:
                self.__print_info(iteration=iteration)
            error = self.__update_error()

        return self.best, self.best_x

    def _initialization(self) -> None:
        """initialization for single fidelity bayesian optimization
        """
        # get initial samples
        sampler = LHS(design_space=self.problem.input_domain)
        self.init_sample = sampler.get_samples(
            num_samples=self.num_init, seed=self.seed)
        self.init_response = self.problem.f(self.init_sample)

        # initialize the surrogate model
        self.surrogate = Kriging(design_space=self.problem.input_domain,
                                 noise_prior=0.0,
                                 optimizer_restart=10,)
        self.surrogate.train(X=self.init_sample,
                             Y=self.init_response)

        # update the sample and response
        self.sample = self.init_sample.copy()
        self.response = self.init_response.copy()

        # initialize the acquisition function
        self.best = np.min(self.init_response)
        self.best_x = self.init_sample[np.argmin(self.init_response), :]

        # record the optimization history
        self.history_best.append(self.best)
        self.history_best_x.append(self.best_x)

        if self.verbose:
            self.__print_info(iteration=0)

    def _update_para(self) -> None:
        # update best
        min_y = np.min(self.response)
        min_index = np.argmin(self.response)
        # update best
        self.best = min_y
        self.best_x = self.sample[min_index, :]
        # record the optimization history
        self.history_best.append(self.best)
        self.history_best_x.append(self.best_x)

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

    def __print_info(self, iteration: int) -> None:
        """print optimum information to screen

        Parameters
        ----------
        iteration : int
            iteration number
        """
        # print the best y of current iteration
        if iteration == 0:
            print("============= Best objective =========")
            print(f"best_y: {self.best:4f}")
            print(f"best_x: {self.best_x}")

        else:
            print(
                f"=========== Best objective at iter {iteration} =========")
            print(f"best_y: {self.best:4f}")
            print(f"best_x: {self.best_x}")
            if self.stopping_error < 1.0:
                print(f"error: {self.__update_error():4f}")
