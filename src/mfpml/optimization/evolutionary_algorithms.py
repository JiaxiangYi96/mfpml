import pickle
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from mfpml.problems.functions import FunctionWrapper


class EABase:
    def run_optimizer(
        self,
        func: Any,
        num_dim: int,
        design_space: np.ndarray,
        print_info: bool = False,
        args: Any = (),
    ) -> dict:
        """main function

        Parameters
        ----------
        func : any
            objective function
        num_dim : int
            dimension of objective
        design_space : np.ndarray
            design space

        Returns
        -------
        results: dict
            optimization results by evolutionary algorithms
        """

        raise NotImplementedError("Subclasses should implement this method.")

    def plot_optimization_history(
        self, figure_name: str = "optimization", save_figure: bool = True
    ) -> None:
        fig, ax = plt.subplots()
        ax.plot(
            np.linspace(0, self.num_gen, self.num_gen + 1, endpoint=True),
            self.gen_best,
            label="optimum",
        )
        ax.legend()
        ax.set(xlabel="iteration")
        ax.set(ylabel="optimum")
        plt.xlim(left=0, right=self.num_gen + 1)
        if save_figure is True:
            fig.savefig(figure_name, dpi=300)
        plt.show()

    def _location_initialzer(self) -> np.ndarray:
        """location/position initialization"""
        x = (
            np.random.random((self.num_pop, self.num_dim))
            * (self.high_bound - self.low_bound)
            + self.low_bound
        )

        return x

    def _gen_best_x(self, pop: np.ndarray, obj: np.ndarray) -> np.ndarray:
        """_summary_

        Parameters
        ----------
        pop : np.ndarray
            _description_
        obj : np.ndarray
            _description_

        Returns
        -------
        np.ndarray
            _description_
        """

        return pop[np.argmin(obj), :].reshape(-1, self.num_dim)

    def _gen_best_obj(self, obj: np.ndarray) -> np.ndarray:
        """_summary_

        Parameters
        ----------
        obj : np.ndarray
            _description_

        Returns
        -------
        np.ndarray
            _description_
        """
        return obj.min().reshape(-1, 1)

    def _loc_cons(self, x: np.ndarray) -> None:
        # TODO check correctness
        # generate new location

        for ii in range(self.num_dim):
            temp = x[:, ii]
            x_replace = self._location_initialzer()
            temp[
                (temp < self.low_bound[ii]) | (temp > self.high_bound[ii])
            ] = x_replace[:, ii][
                (temp < self.low_bound[ii]) | (temp > self.high_bound[ii])
            ]
            x[:, ii] = temp
        return x

    def _save_results(self) -> None:
        results = {
            "best_x": self.gen_best_x,
            "best_y": self.gen_best,
            "best_x_historical": self.gen_best_x,
            "best_y_historical": self.gen_best,
            "all_samples": self.samples,
            "all_responses": self.responses,
            "design_space": self.design_space,
            "pop": self.num_pop,
            "gen": self.num_gen,
        }

        with open("step_results.pickle", "wb") as f:
            pickle.dump(results, f)

    def print_info(self, iter: int) -> None:
        print(
            f"iteration:{iter},\
                optimum:{self.gen_best[iter,:].flatten()},\
                    best_x:{self.gen_best_x[iter,:].flatten()}\n")


class PSO(EABase):
    def __init__(
        self,
        num_pop: int,
        num_gen: int,
        cognition_rate: float = 1.5,
        social_rate: float = 1.5,
        weight: float = 0.8,
        max_velocity_rate: float = 0.2,
    ) -> None:
        """Initialization of PSO

        Parameters
        ----------
        num_pop : int
            number of population
        num_gen : int
            number of generation
        cognition_rate : float, optional
            connition rate, by default 1.5
        social_rate : float, optional
            social rate, by default 1.5
        weight : float, optional
            weight of previous speed , by default 0.8
        max_velocity_rate: float
            maximum velocity rate
        """
        self.num_pop = num_pop
        self.num_gen = num_gen
        self.cognition_rate = cognition_rate
        self.social_ratio = social_rate
        self.weight = weight
        self.max_v_rate = max_velocity_rate

    def run_optimizer(
        self,
        func: Any,
        num_dim: int,
        design_space: np.ndarray,
        print_info: bool = False,
        save_step_results: bool = False,
        stopping_error: float = None,
        args: Any = (),
    ) -> dict:
        """main function of pso algorithm

        Parameters
        ----------
        func : any
            objective function
        num_dim : int
            dimension of objective
        design_space : np.ndarray
            design space

        Returns
        -------
        results: dict
            optimization results of pso
        """

        # update some params
        self.stopping_error = stopping_error
        self.func = FunctionWrapper(function=func, args=args)
        self.num_dim = num_dim
        self.design_space = design_space
        self.low_bound = design_space[:, 0]
        self.high_bound = design_space[:, 1]
        # define the best information
        self.gen_best = np.zeros((self.num_gen + 1, 1))
        self.gen_best_x = np.zeros((self.num_gen + 1, num_dim))
        self.pop_best = np.zeros((self.num_pop, 1))
        self.pop_best_x = np.zeros((self.num_pop, self.num_dim))

        # define an array to save all data
        self.samples = np.zeros([self.num_gen + 1, self.num_pop, num_dim])
        self.responses = np.zeros([self.num_gen + 1, self.num_pop, 1])

        # run
        self._pso_initializer()
        # main iteration
        for iter in range(self.num_gen):
            # print info
            if print_info:
                self.print_info(iter=iter)
            # update the location information
            self.__update_pop(iter=iter)
            # update the obj
            self.__update_obj(iter=iter)
            # update the population optimum information
            self.__update_pop_optimum()
            # update the generation optimum information
            self.__update_gen_optimum(iter=iter)
            if save_step_results is True:
                self._save_results()

            if self.stopping_error is not None:
                if self.gen_best[iter + 1, 0] < self.stopping_error:
                    break

        # get the final results
        results = {
            "best_x": self.gen_best_x[-1, :],
            "best_y": self.gen_best[-1, :],
            "best_x_historical": self.gen_best_x,
            "best_y_historical": self.gen_best,
            "all_samples": self.samples,
            "all_responses": self.responses,
            "design_space": design_space,
            "pop": self.num_pop,
            "gen": self.num_gen,
        }

        return (
            results,
            self.gen_best[iter + 1, 0],
            self.gen_best_x[iter + 1, :],
        )

    def __velocity_initialzer(self, trick: str = "random") -> None:
        """velocity initialization

        Parameters
        ----------
        trick : str
            "random" or "zero"
        """
        self.v_max = (self.high_bound - self.low_bound) * self.max_v_rate
        if trick == "random":
            self.v = (
                np.random.random((self.num_pop, self.num_dim)) * self.v_max
            )
        elif trick == "zero":
            self.v = np.zeros((self.num_pop, self.num_dim))
        else:
            raise Exception(
                "The trick for velocity initialization is not defined!\n"
            )

    def _pso_initializer(self) -> None:
        """calculate the objective values for the initial population"""
        self.x = self._location_initialzer()
        self.__velocity_initialzer()
        self.obj = self.func(self.x)
        self.obj = np.reshape(self.obj, (self.num_pop, 1))

        # find the best values for individuals
        self.pop_best = self.obj
        self.pop_best_x = self.x
        # find the best value for population
        self.gen_best[0, :] = self._gen_best_obj(obj=self.obj)
        self.gen_best_x[0, :] = self._gen_best_x(pop=self.x, obj=self.obj)

        # save samples
        self.samples[0, :, :] = self.x.copy()
        self.responses[0, :, :] = self.obj.copy()

    def __update_pop(self, iter: int) -> None:
        """update the population information

        Parameters
        ----------
        iter : int
            iteration
        """
        self.v = (
            self.weight * self.v
            + self.cognition_rate * (self.pop_best_x - self.x)
            + self.social_ratio * (self.gen_best_x[iter, :] - self.x)
        )
        self.__velocity_cons()
        self.x = self.x + self.v
        self.x = self._loc_cons(x=self.x)
        # save pop info
        self.samples[iter + 1, :, :] = self.x.copy()

    def __update_obj(self, iter: int) -> None:
        self.obj = self.func(self.x)
        self.obj = np.reshape(self.obj, (self.num_pop, 1))
        self.responses[iter + 1, :, :] = self.obj.copy()

    def __update_pop_optimum(self) -> None:
        """update the population optimum information"""
        index = np.where(self.obj <= self.pop_best)
        self.pop_best[index[0], :] = self.obj[index[0], :]
        self.pop_best_x[index[0], :] = self.x[index[0], :]
        del index

    def __update_gen_optimum(self, iter: int) -> None:
        """update the generation optimum information"""
        if self.gen_best[iter, 0] >= self._gen_best_obj(obj=self.obj):
            self.gen_best[iter + 1, 0] = self._gen_best_obj(obj=self.obj)
            self.gen_best_x[iter + 1, :] = self._gen_best_x(
                pop=self.x, obj=self.obj
            )
        else:
            self.gen_best[iter + 1, 0] = self.gen_best[iter, 0]
            self.gen_best_x[iter + 1, :] = self.gen_best_x[iter, :]

    def __velocity_cons(self) -> None:
        for ii in range(self.num_dim):
            index_vel = np.where(np.abs(self.v[:, ii]) > self.v_max[ii])
            self.v[index_vel[0], ii] = (
                self.v[index_vel[0], ii]
                * self.v_max[ii]
                / np.abs(self.v[index_vel[0], ii])
            )


class DE(EABase):
    def __init__(
        self,
        num_pop: int,
        num_gen: int,
        step_size: float = 0.5,
        crossover_rate: float = 0.5,
        strategy: str = "DE/rand/1/bin",
    ) -> None:
        self.num_pop = num_pop
        self.num_gen = num_gen
        self.step_size = step_size
        self.crossover_rate = crossover_rate
        self.strategy = strategy

    def run_optimizer(
        self,
        func: Any,
        num_dim: int,
        design_space: np.ndarray,
        print_info: bool = False,
        save_step_results: bool = False,
        stopping_error: float = None,
        args: Any = (),
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        # update some params
        self.stopping_error = stopping_error
        self.func = FunctionWrapper(function=func, args=args)
        self.num_dim = num_dim
        self.design_space = design_space
        self.low_bound = design_space[:, 0]
        self.high_bound = design_space[:, 1]

        # some unique parameters for defferential evoluntionary algorithms
        self.gen_best = np.zeros((self.num_gen + 1, 1))
        self.gen_best_x = np.zeros((self.num_gen + 1, num_dim))
        # define an array to save all data
        self.samples = np.zeros([self.num_gen + 1, self.num_pop, num_dim])
        self.responses = np.zeros([self.num_gen + 1, self.num_pop, 1])

        # de initialization
        self._de_initializer()
        # main iteration
        for iter in range(self.num_gen):
            # print info
            if print_info:
                self.print_info(iter=iter)
            # update the location information
            self.__update_pop(iter=iter)
            # update the obj
            self.__update_obj(iter=iter)
            # update the generation optimum information
            self.__update_gen_optimum(iter=iter)
            if save_step_results is True:
                self._save_results()
            if self.stopping_error is not None:
                if self.gen_best[iter + 1, 0] < self.stopping_error:
                    break
        # get the final results
        results = {
            "best_x": self.gen_best_x[-1, :],
            "best_y": self.gen_best[-1, :],
            "best_x_historical": self.gen_best_x,
            "best_y_historical": self.gen_best,
            "all_samples": self.samples,
            "all_responses": self.responses,
            "design_space": design_space,
            "pop": self.num_pop,
            "gen": self.num_gen,
        }
        return (
            results,
            self.gen_best[iter + 1, 0],
            self.gen_best_x[iter + 1, :],
        )

    def _de_initializer(self) -> None:
        """calculate the objective values for the initial population"""
        self.x = self._location_initialzer()
        self.v = np.zeros([self.num_pop, self.num_dim])
        self.u = np.zeros([self.num_pop, self.num_dim])
        self.obj = self.func(self.x)
        self.obj = np.reshape(self.obj, (self.num_pop, 1))
        # find the best value for population
        self.gen_best[0, :] = self._gen_best_obj(obj=self.obj)
        self.gen_best_x[0, :] = self._gen_best_x(pop=self.x, obj=self.obj)
        # save samples
        self.samples[0, :, :] = self.x.copy()
        self.responses[0, :, :] = self.obj.copy()

    def __update_pop(self, iter: int) -> None:
        if self.strategy == "DE/rand/1/bin":
            for ii in range(self.num_pop):
                r = np.array(
                    [np.random.choice(self.num_pop, 3, replace=False)]
                )
                # selected inidivials is not same as the index
                if ii < self.num_pop - 1:
                    r[r == ii] = ii + 1
                elif ii == self.num_pop - 1:
                    r[r == ii] = 0
                else:
                    raise Exception("index exceed population number \n")
                # update the pop
                self.v[ii, :] = self.x[r[0, 0], :] + self.step_size * (
                    self.x[r[0, 1], :] - self.x[r[0, 2], :]
                )
                c = np.random.choice(self.num_dim, 1)
                for jj in range(self.num_dim):
                    # generate the random number for crossover
                    cr = np.random.random(1)
                    if cr < self.crossover_rate or c == jj:
                        self.u[ii, jj] = self.v[ii, jj]
                    else:
                        self.u[ii, jj] = self.x[ii, jj]
            # location constrains
            self.u = self._loc_cons(x=self.u)

        elif self.strategy == "DE/best/1/bin":
            for ii in range(self.num_pop):
                r = np.array(
                    [np.random.choice(self.num_pop, 2, replace=False)]
                )
                # selected inidivials is not same as the index
                if ii < self.num_pop - 1:
                    r[r == ii] = ii + 1
                elif ii == self.num_pop - 1:
                    r[r == ii] = 0
                else:
                    raise Exception("index exceed population number \n")
                # update the pop
                # get the best scheme
                x_best = self.gen_best_x[iter, :].copy()
                self.v[ii, :] = x_best + self.step_size * (
                    self.x[r[0, 0], :] - self.x[r[0, 1], :]
                )
                c = np.random.choice(self.num_dim, 1)
                for jj in range(self.num_dim):
                    # generate the random number for crossover
                    cr = np.random.random(1)
                    if cr < self.crossover_rate or c == jj:
                        self.u[ii, jj] = self.v[ii, jj]
                    else:
                        self.u[ii, jj] = self.x[ii, jj]
            # location constrains
            self.u = self._loc_cons(x=self.u)
        else:
            raise ValueError("This strategy is not defined! \n ")

    def __update_obj(self, iter: int) -> None:
        self.obj_u = self.func(self.u)
        self.obj_u = np.reshape(self.obj_u, (self.num_pop, 1))
        # decide the next parent population
        for ii in range(self.num_pop):
            if self.obj_u[ii, :] < self.obj[ii, :]:
                self.x[ii, :] = self.u[ii, :]
                self.obj[ii, :] = self.obj_u[ii, :]
        self.obj = np.reshape(self.obj, (self.num_pop, 1))

        # save pop info
        self.samples[iter + 1, :, :] = self.x.copy()
        self.responses[iter + 1, :, :] = self.obj.copy()

    def __update_gen_optimum(self, iter: int) -> None:
        """update the generation optimum information"""
        if self.gen_best[iter, 0] >= self._gen_best_obj(obj=self.obj):
            self.gen_best[iter + 1, 0] = self._gen_best_obj(obj=self.obj)
            self.gen_best_x[iter + 1, :] = self._gen_best_x(
                pop=self.x, obj=self.obj
            )
        else:
            self.gen_best[iter + 1, 0] = self.gen_best[iter, 0]
            self.gen_best_x[iter + 1, :] = self.gen_best_x[iter, :]
