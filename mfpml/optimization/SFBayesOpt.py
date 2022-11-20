import numpy as np
import matplotlib.pyplot as plt
from mfpml.utils.differential_evolution_optimizer import *
from mfpml.optimization.aquisitions import *


class SFBayesOpt:

    def __init__(self, model: any,
                 acquisition: str,
                 design_space: dict,
                 optimizer: str,
                 Analytical: bool = True) -> None:
        """
        Initialization of Single fidelity Bayesian Optimization
        Parameters
        ----------
        model: any
            GPR models or any Bayesian machine learning models that can provide predicted mean and variance
        acquisition: str
            name of acquisition function
        design_space: dict
            design space
        optimizer: str
            name of optimizer, Input of optimizer should be design variables and the output should be the best location
        Analytical:bool
            Analytical objective function or not
        """
        self.model = model
        self.acquisition = acquisition
        self.optimizer = optimizer
        self.design_space = design_space
        self.historical_optimal_design = self.best_design_scheme().reshape(-1, model.num_dim)
        self.historical_optimum = self.best_objective().reshape(-1, 1)
        # self.historical_optimal_design = np.ndarray(self.best_design_scheme()).reshape(-1, models.num_dim)
        # self.historical_optimum = np.ndarray(self.best_objective()).reshape(-1, 1)
        self.Analytical = Analytical
        self.iteration = None

    def run_optimizer(self, fun: any,
                      iteration: int,
                      print_info: bool = True) -> dict:
        self.iteration = iteration
        if self.Analytical is False:
            raise Exception('Not Analytical function \n')
        else:
            iter = 0
            while iter < iteration:
                self._print_info(iter=iter)
                self._update_model()
                self._update_samples(add_x=self.suggested_location(),
                                     add_y=fun(self.suggested_location()))
                self._historical_info()
                iter = iter + 1

        results = {'optimum': self.historical_optimum,
                   'optimal_design': self.historical_optimal_design}

        return results

    def suggested_location(self) -> np.ndarray:
        """

        Returns
        -------
        next_loc:array
            suggested location by the acquisition function

        """
        optimizer = self._get_optimizer()
        acquisition_function = self._get_acquisition()

        results = optimizer.run_optimizer(fun=acquisition_function,
                                          design_space=self.design_space)
        return results['best_x']

    def _get_optimizer(self) -> any:
        """
        Get the optimizer for acquisition function
        Returns
        -------

        """
        if self.optimizer == 'DE':
            optimizer = DE()
        # TODO, add more optimizer here
        else:
            raise Exception('The optimizer is not defined in mfpml \n')

        return optimizer

    def _get_acquisition(self) -> any:
        """
        Get the acquisition function
        Returns
        -------
        acquisition_func: function
            acquisition function of Bayesian optimization

        """
        if self.acquisition == 'LCB':
            acquisition_func = LCB(model=self.model)
        elif self.acquisition == 'EI':
            acquisition_func = EI(model=self.model)
        elif self.acquisition == 'PI':
            acquisition_func = PI(model=self.model)
        # TODO, add more optimizer here
        else:
            raise Exception('Acquisition function is not defined in mfpml \n')

        return acquisition_func

    def best_objective(self) -> list:
        """
        get the best design of current iteration
        Returns
        -------
        best_objective: np.ndarray
            Best objective

        """
        return self.model.y.min()

    def best_design_scheme(self) -> np.ndarray:
        """
        get the best design scheme of current iteration
        Returns
        -------
        best_design_scheme: np.ndarray
            Best design scheme

        """
        return self.model.x[np.argmin(self.model.y), :]

    def _historical_info(self) -> None:
        """
        Update the historical best optimal infomation
        Returns
        -------

        """

        self.historical_optimal_design = np.concatenate((self.historical_optimal_design,
                                                         self.best_design_scheme().reshape(-1, self.model.num_dim)),
                                                        axis=0)
        self.historical_optimum = np.concatenate((self.historical_optimum,
                                                  self.best_objective().reshape(-1, 1)),
                                                 axis=0)

    def _update_model(self):
        """
        Update the GPR models
        Returns
        -------
        models: any
            Updated GPR models

        """

        self.model.train_model(self.model.x, self.model.y, num_dim=self.model.num_dim)

    def _update_samples(self, add_x: np.ndarray, add_y: np.ndarray) -> any:
        """
        Update samples information in GPR models

        Parameters
        ----------
        add_x: np.ndarray
            most promising location identified by acquisition function
        add_y: np.ndarray
            response of objective function at add_x

        Returns
        -------
        updated samples structure

        """

        self.model.x = np.concatenate((self.model.x, add_x.reshape(-1, self.model.num_dim)), axis=0)
        self.model.y = np.concatenate((self.model.y, add_y.reshape(-1, 1)), axis=0)

    def _print_info(self, iter: int) -> None:
        """
        print optimization information to the screen

        Parameters
        ----------
        iter: int
            current iteration of Bayesian Optimization

        Returns
        -------

        """

        print(f'Iteration:{iter},'
              f' Current optimum:{self.best_objective().tolist()},'
              f' Current optimal scheme: {self.best_design_scheme().tolist()}')

    def historical_plot(self, save: bool = True, name: str = 'historical_plot') -> None:
        """
        Plot the historical optimum
        Parameters
        ----------
        save: bool
            save figure or not
        name: str
            name of the figure

        Returns
        -------

        """

        x_plot = np.linspace(start=0,
                             stop=self.iteration,
                             num=self.iteration + 1,
                             endpoint=True)
        x_plot = x_plot.reshape((-1, 1))
        with plt.style.context(['ieee', 'science']):
            fig, ax = plt.subplots()
            ax.plot(x_plot, self.historical_optimum, '-.*', label='Optimum')
            ax.plot(x_plot, self.model.y[-self.iteration - 1:, :], '--o', label=f'Added samples')
            ax.legend()
            ax.set(xlabel=r'$Iteration$')
            ax.set(ylabel=r'$y$')
            # ax.autoscale(tight=True)
            if save is True:
                fig.savefig(name, dpi=300)
            plt.show(block=True)
            plt.interactive(False)
