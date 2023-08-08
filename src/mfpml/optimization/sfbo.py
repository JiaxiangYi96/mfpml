from typing import Any

import numpy as np
from matplotlib import pyplot as plt

from ..design_of_experiment.singlefideliy_samplers import LatinHyperCube
from ..models.sf_gpr import Kriging
from .evolutionary_algorithms import DE
from .sf_acqusitions import EI, LCB, PI


class SFBO(object):

    def __init__(self, problem: Any,
                 design_space: dict,
                 seed: int = 123) -> None:
        """single fidelity bayesian optimization

        Parameters
        ----------
        problem : Any
            objective function
        design_space : dict
            design space of the objective function
        seed : int, optional
            seed for initial sampling, by default 123
        """
        # problem
        self.problem = problem
        # design space
        self.design_space = design_space

        # number dimension
        self.num_dim = len(self.design_space)
        # get input domain
        self.input_domain = self._get_input_domain()
        # sampler
        self.sampler = LatinHyperCube(design_space=self.design_space,
                                      seed=seed)
        # initial Kriging model
        self.model = Kriging(design_space=self.input_domain)

    def _get_input_domain(self) -> np.ndarray:
        """get input domain, i.e., the bounds of the design space"""

        for key, value in enumerate(self.design_space.values()):
            if key == 0:
                input_domain = np.array(value)
            else:
                input_domain = np.vstack((input_domain, value))
            # change to 2d array
            input_domain = np.atleast_2d(input_domain)
        return input_domain

    def _acquisition(self, acq: str = "EI") -> None:
        """choose acquisition function

        Parameters
        ----------
        acq : str, optional
            name of acquisition function, by default "EI"
        """
        if acq == "EI":
            self.acq_func = EI(model=self.model)
        elif acq == "LCB":
            self.acq_func = LCB(model=self.model)
        elif acq == "PI":
            self.acq_func = PI(model=self.model)

    # run optimizer
    def run_optimizer(self, num_init: int = 5,
                      num_iter: int = 10,
                      acquisition: str = 'EI',
                      print_info: bool = True,
                      optimum: float = None,
                      stopping_error: float = 1.0,
                      ) -> tuple[np.ndarray, np.ndarray]:
        """run bayesian optimization

        Parameters
        ----------
        num_init : int, optional
            initial number of samples, by default 5
        num_iter : int, optional
            number of iterations, by default 10
        acquisition : str, optional
            name of acquisition function, by default 'EI'
        print_info : bool, optional
            print optimization information, by default True
        optimum : float, optional
            known optimum (for some numerical problems), by default None
        stopping_error : float, optional
            stopping error, by default 1.0

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            optimal information
        """
        self.num_init = num_init
        self.num_iter = num_iter
        self.known_optimum = optimum
        self.stopping_error = stopping_error
        # get initial samples
        self.samples = self.sampler.get_samples(num_samples=self.num_init)
        self.responses = self.problem(self.samples)

        # main loop
        iter = 0
        error = self.__update_error()
        while iter < num_iter and error >= self.stopping_error:
            # update the best values
            self.__get_optimum(iter=iter)
            self.model.train(sample_x=self.samples, sample_y=self.responses)
            # acquisition function
            self._acquisition(acq=acquisition)
            # optimization
            self.optimizer = DE(num_gen=200, num_pop=50,
                                strategy='DE/best/1/bin')
            # optimize acquisition function
            _, _, optimal_x = self.optimizer.run_optimizer(
                func=self.acq_func,
                num_dim=self.num_dim,
                design_space=self.input_domain)
            # calculate optimal y
            optimal_y = self.problem(np.atleast_2d(optimal_x))

            # cancatenate samples
            self.samples = np.concatenate(
                (self.samples, optimal_x.reshape((1, -1))), axis=0)
            self.responses = np.concatenate(
                (self.responses, optimal_y.reshape((-1, 1))), axis=0)
            iter = iter + 1
            # print info
            if print_info:
                self.__print_info(iter=iter)
            # update error
            error = self.__update_error()
        # get the last optimum
        self.__get_optimum(iter=iter)

        return self.best_y[-1, :], self.best_y[-1, :]

    def __print_info(self, iter: int) -> None:
        """print optimum information to screen

        Parameters
        ----------
        iter : int
            iteration number
        """
        # print the best y of current iteration
        print(f"iter:{iter} =====================================")
        print(f"best_y: {self.responses.min():4f}")
        print(f"best_x: {self.samples[self.responses.argmin(), :]}")
        if self.stopping_error < 1.0:
            print(f"error: {self.__update_error():4f}")

    def __update_error(self) -> Any | float:
        """update error between the known optimum and the current optimum

        Returns
        -------
        float
            error
        """
        if self.known_optimum is not None and self.known_optimum != 0:
            error = np.abs((self.responses.min() - self.known_optimum) /
                           self.known_optimum)
        elif self.known_optimum == 0.0:
            error = np.abs(self.responses.min() - self.known_optimum)
        else:
            error = 1.0

        return error

    def __get_optimum(self, iter: int) -> None:
        """update historical optimum information

        Parameters
        ----------
        iter : int
            iteration
        """
        if iter == 0:
            self.best_y = np.atleast_2d(self.responses.min())
            self.best_x = np.atleast_2d(
                self.samples[self.responses.argmin(), :])
        else:
            self.best_y = np.concatenate(
                (self.best_y, np.atleast_2d(self.responses.min())), axis=0)
            # get the best index of current iteration
            self.best_x = np.concatenate(
                (self.best_x,
                 np.atleast_2d(self.samples[self.responses.argmin(), :])),
                axis=0)

    def plot_optimization_history(self, save_fig: bool = False,
                                  fig_name="bo_history.png") -> None:
        """plot optimization history
        """

        fig, ax = plt.subplots()
        ax.plot(self.best_y, lw=2, label="optimum")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$f(x)$")
        ax.legend(loc="best")
        ax.grid()
        if save_fig:
            plt.savefig(fig_name, dpi=300)
        plt.show()

    @property
    def optimum(self) -> tuple[float, np.ndarray]:
        return self.best_y[-1, 0]

    @property
    def optimum_scheme(self) -> np.ndarray:
        return self.best_x[-1, :]
