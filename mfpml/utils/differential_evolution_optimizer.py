import numpy as np
import scipy
from scipy.optimize import differential_evolution


class DE:

    def __init__(self):
        self.fun = None
        self.bounds = None
        self.best_x = None
        self.best_y = None
        self.best_fun = None
        self.result = None

    def run_optimizer(self, fun: any, design_space: dict, seed: int = 1) -> dict:
        """

        Parameters
        ----------
        fun: Function
            objective function
        design_space: dict
            Design space of the objective function
        seed: int
            seeds

        Returns
        -------
        Results: scipy.OptimizeResult
            optimization results after differential evolution

        """
        self.fun = fun
        self._get_bounds(design_space=design_space)
        self.result = differential_evolution(func=fun,
                                             bounds=self.bounds,
                                             vectorized=True,
                                             updating='deferred',
                                             seed=seed)
        results = self._get_results()

        return results

    def _get_results(self) -> dict:
        """

        Returns
        -------
        results: dict
            optimization results

        """
        self.best_x = self.result.x
        self.best_y = self.result.fun
        self.nfem = self.result.nfev
        results = {'best_x': self.best_x,
                   'best_y': self.best_y,
                   'nfem': self.nfem}

        return results

    def _get_bounds(self, design_space):
        """
        Get the bounds of design space which satsify the
        requirement of DE of scipy

        Parameters
        ----------
        design_space: dict
            design space

        Returns
        -------

        """
        bounds = list()
        for value in design_space.values():
            bounds.extend([value])

        self.bounds = bounds

        # self.bounds = Bounds(bounds)
