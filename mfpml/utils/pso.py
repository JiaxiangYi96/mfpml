from operator import index

import matplotlib.pyplot as plt
import numpy as np


class PSO:
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

    def run_optimizer(self, func: any, num_dim: int, design_space: np.ndarray, print_info: bool = False) -> dict:

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
        self.func = func
        self.num_dim = num_dim
        self.design_space = design_space
        self.low_bound = design_space[:, 0]
        self.high_bound = design_space[:, 1]
        # define the best information
        self.gen_best = np.zeros((self.num_gen + 1, 1))
        self.gen_best_x = np.zeros((self.num_gen + 1, num_dim))
        self.pop_best = np.zeros((self.num_pop, 1))
        self.pop_best_x = np.zeros((self.num_pop, self.num_dim))
        ## run
        self._pso_initializer()
        # main iteration
        for iter in range(self.num_gen):
            # print info
            if print_info == True:
                self.__print_info(iter=iter)
            # # velocity rule
            # self.__velocity_cons()
            # # location rule
            # self.__loc_cons()
            # update the location information
            self.__update_pop(iter=iter)
            # update the obj
            self.__update_obj()
            # update the population optimum information
            self.__update_pop_optimum()
            # update the generation optimum information
            self.__update_gen_optimum(iter=iter)

        results = {"best_x": self.gen_best_x[-1, :], "best_y": self.gen_best[-1, :]}
        return results

    def plot_optimization_history(self, figure_name: str = "pso_optimization", save_figure: bool = True) -> None:
        with plt.style.context(["ieee", "science", "high-contrast"]):
            fig, ax = plt.subplots()
            ax.plot(np.linspace(0, self.num_gen, self.num_gen + 1, endpoint=True), self.gen_best, label="optimum")
            ax.legend()
            ax.set(xlabel="iteration")
            ax.set(ylabel="optimum")
            plt.xlim(left=0, right=self.num_gen + 1)
            if save_figure is True:
                fig.savefig(figure_name, dpi=300)
            plt.show()

    def __location_initialzer(self) -> None:
        """location/position initialization"""
        self.x = np.random.random((self.num_pop, self.num_dim)) * (self.high_bound - self.low_bound) + self.low_bound

    def __velocity_initialzer(self, trick: str = "random") -> None:
        """velocity initialization

        Parameters
        ----------
        trick : str
            "random" or "zero"
        """
        self.v_max = (self.high_bound - self.low_bound) * self.max_v_rate
        if trick == "random":
            self.v = np.random.random((self.num_pop, self.num_dim)) * self.v_max
        elif trick == "zero":
            self.v = np.zeros((self.num_pop, self.num_dim))
        else:
            raise Exception("The trick for velocity initialization is not defined!\n")

    def _pso_initializer(self) -> None:
        """calculate the objective values for the initial population"""
        self.__location_initialzer()
        self.__velocity_initialzer()
        self.obj = self.func(self.x)
        self.obj = np.reshape(self.obj, (self.num_pop, 1))

        # find the best values for individuals
        self.pop_best = self.obj
        self.pop_best_x = self.x

        # find the best value for population
        self.gen_best[0, :] = self.__gen_best_obj(obj=self.obj)
        self.gen_best_x[0, :] = self.__gen_best_x(pop=self.x, obj=self.obj)

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
        self.__loc_cons()

    def __update_obj(self) -> None:

        self.obj = self.func(self.x)
        self.obj = np.reshape(self.obj, (self.num_pop, 1))

    def __update_pop_optimum(self) -> None:
        """update the population optimum information"""
        index = np.where(self.obj <= self.pop_best)
        self.pop_best[index[0], :] = self.obj[index[0], :]
        self.pop_best_x[index[0], :] = self.x[index[0], :]

        # self.pop_best[self.obj <= self.pop_best] = self.obj[self.obj <= self.pop_best].reshape(-1, 1)
        # self.pop_best_x[self.obj <= self.pop_best] = self.x[self.obj <= self.pop_best].reshape(-1, self.num_dim)

    def __update_gen_optimum(self, iter: int) -> None:
        """update the generation optimum information"""
        if self.gen_best[iter, 0] >= self.__gen_best_obj(obj=self.obj):
            self.gen_best[iter + 1, 0] = self.__gen_best_obj(obj=self.obj)
            self.gen_best_x[iter + 1, :] = self.__gen_best_x(pop=self.x, obj=self.obj)
        else:
            self.gen_best[iter + 1, 0] = self.gen_best[iter, 0]
            self.gen_best_x[iter + 1, :] = self.gen_best_x[iter, :]

    def __velocity_cons(self) -> None:
        # TODO check correctness
        for ii in range(self.num_dim):
            index = np.where(np.abs(self.v[:, ii]) > self.v_max[ii])
            self.v[index[0], ii] = self.v[index[0], ii] * self.v_max[ii] / np.abs(self.v[index[0], ii])
            # for jj in range(self.num_pop):
            #     while np.abs(self.v[jj, ii]) > self.v_max[ii]:
            #         self.v[jj, ii] = self.v[jj, ii] * self.v_max[ii] / np.abs(self.v[jj, ii])

        # self.v[self.v > self.v_max] = self.v[self.v > self.v_max] * self.v_max / abs(self.v[self.v > self.v_max])

    def __loc_cons(self) -> None:
        # TODO check correctness
        for ii in range(self.num_dim):
            for jj in range(self.num_pop):
                while self.x[jj, ii] > self.high_bound[ii] or self.x[jj, ii] < self.low_bound[ii]:
                    self.x[jj, ii] = (
                        np.random.random((1)) * (self.high_bound[ii] - self.low_bound[ii]) + self.low_bound[ii]
                    )

    def __gen_best_x(self, pop: np.ndarray, obj: np.ndarray) -> np.ndarray:
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

    def __gen_best_obj(self, obj: np.ndarray) -> np.ndarray:
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

    def __print_info(self, iter: int) -> None:

        print(
            f"iteration:{iter}, optimum:{self.gen_best[iter,:].flatten()}, best_x:{self.gen_best_x[iter,:].flatten()}\n"
        )
