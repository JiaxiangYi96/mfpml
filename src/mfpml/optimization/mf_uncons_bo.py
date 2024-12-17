# import
from typing import Any, List

import numpy as np

from ..design_of_experiment.mf_samplers import MFLatinHyperCube as MFLHS
from ..models.co_kriging import CoKriging as CK
from ..models.hierarchical_kriging import HierarchicalKriging as HK
from ..models.scale_kriging import ScaledKriging as SK
from ..problems.functions import Functions
from .mf_acqusitions import MFUnConsAcq


class mfUnConsBayesOpt:
    """
    Multi-fidelity single objective Bayesian optimization
    """

    def __init__(self,
                 problem: Functions,
                 acquisition: MFUnConsAcq,
                 num_init: List,
                 surrogate_name: str = 'HierarchicalKriging',
                 seed: int = 1996,
                 verbose: bool = True) -> None:
        """Initialize the multi-fidelity Bayesian optimization

        Parameters
        ----------
        problem : Any
            optimization problem
        """
        self.problem = problem
        self.acquisition = acquisition
        self.surrogate_name = surrogate_name
        self.num_init = num_init
        self.seed = seed
        self.verbose = verbose

        # set the random seed
        np.random.seed(seed)

        # record optimization info
        self.best: float = np.inf
        self.best_x: np.ndarray = None
        self.history_best: List[float] = []
        self.history_best_x: List[np.ndarray] = []
        self.known_optimum: float = None
        self.stopping_error: float = 1.0
        self.fidelity_query_order: List = []

        # initialization
        self._initialization()

    def run_optimizer(self,
                      max_iter: int = 10,
                      cost_ratio: float = 1.0,
                      stopping_error: float = 1.0) -> None:
        """Run the optimization

        Parameters
        ----------
        max_iter : int, optional
            maximum number of iterations, by default 10
        cost_ratio : float, optional
            cost ratio between high-fidelity and low-fidelity, by default 1.0
        stopping_error : float, optional
            stopping error, by default 1.0
        """
        if self.problem.optimum is not None:
            self.known_optimum = self.problem.optimum
        else:
            self.known_optimum = None

        # error-based stopping criterion
        self.stopping_error = stopping_error
        # main loop
        iteration = 0
        error = self.__update_error()
        print(f"Error: {error}")
        while iteration < max_iter and error > stopping_error:
            # update the surrogate model
            update_x, fidelity = self.acquisition.query(
                mf_surrogate=self.surrogate,
                cost_ratio=cost_ratio,
                fmin=self.best)
            if fidelity == 0:
                update_y = self.problem.hf(np.atleast_2d(update_x))
                # update the samples
                self.sample[0] = np.vstack((self.sample[0], update_x))
                self.response[0] = np.vstack((self.response[0], update_y))
            else:
                update_y = self.problem.lf(np.atleast_2d(update_x))
                # update the samples
                self.sample[1] = np.vstack((self.sample[1], update_x))
                self.response[1] = np.vstack((self.response[1], update_y))
            # update the surrogate model
            self.surrogate.train(self.sample, self.response)
            print("update_x: ", update_x)
            print("update_y: ", update_y)
            print("fidelity: ", fidelity)
            # update the params
            self._update_para(update_x=update_x,
                              update_y=update_y, fidelity=fidelity)

            # iteration
            iteration += 1
            if self.verbose:
                self._print_info(iteration)
            # update the error
            error = self.__update_error()

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

    def _update_para(self, update_x, update_y, fidelity) -> None:
        """Update parameters

        Parameters
        ----------
        update_x : dict
            update samples
        update_y : dict
            update responses
        """

        # add the fidelity query order
        self.fidelity_query_order.append(fidelity)
        # update the best scheme
        if fidelity == 0:
            if update_y[0][0] < self.best:
                self.best = update_y[0][0]
                self.best_x = update_x

        # record the optimization history
        self.history_best.append(self.best)
        self.history_best_x.append(self.best_x)

    def _initialization(self) -> None:
        """Initialize parameters
        """
        sampler = MFLHS(design_space=self.problem.input_domain, num_fidelity=2)
        self.init_sample = sampler.get_samples(
            num_samples=self.num_init, seed=self.seed)
        self.init_response = self.problem(self.init_sample)

        # initialize the surrogate model
        if self.surrogate_name == 'HierarchicalKriging':
            self.surrogate = HK(design_space=self.problem.input_domain,
                                noise_prior=0.0,
                                optimizer_restart=10)
        elif self.surrogate_name == 'ScaledKriging':
            self.surrogate = SK(design_space=self.problem.input_domain,
                                noise_prior=0.0,
                                optimizer_restart=10)
        elif self.surrogate_name == 'CoKriging':
            self.surrogate = CK(design_space=self.problem.input_domain,
                                noise_prior=0.0,
                                optimizer_restart=10)
        else:
            raise ValueError('Unknown surrogate model')

        # train the model
        self.surrogate.train(self.init_sample, self.init_response)

        # update the sample and response
        self.sample = self.init_sample.copy()
        self.response = self.init_response.copy()

        # initialize best scheme
        self.best = np.min(self.response[0])
        self.best_x = self.sample[0][np.argmin(self.response[0]), :]

        # record the optimization history
        self.history_best.append(self.best)
        self.history_best_x.append(self.best_x)

        if self.verbose:
            self._print_info(0)

    def _print_info(self, iteration: int) -> None:
        """Print optimization information

        Parameters
        ----------
        iter : int
            current iteration
        """

        if iteration == 0:
            print(
                "======= Best objective value  ==========")
            print(f"Best objective value: {self.best:.4f}")
            print(f"Best scheme: {self.best_x}")

        else:
            print(
                "=========== Iteration {} ===========".format(iteration))
            print(f"Best objective value: {self.best}")
            print(f"Best scheme: {self.best_x}")
            print("Fidelity query order: ", self.fidelity_query_order[-1])
            if self.stopping_error < 1.0:
                print(f"Stopping error: {self.__update_error():4f}")
