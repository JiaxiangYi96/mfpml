import numpy as np
# from PlotFigures import Plots

import matplotlib.pyplot as plt


class Functions:

    def __call__(self, x: np.ndarray, fidelity: str) -> np.ndarray:
        """

        Parameters
        ----------
        x: np.ndarray
            design scheme that needs to be evaluated

        Returns
        -------
        y: np.ndarray
            responses from the functions

        """
        pass

    def get_optimum(self) -> float:
        """

        Returns
        -------
        optimum: float
            name of the class

        """
        return self.__class__.optimum

    def get_optimum_variable(self) -> list:
        """

        Returns
        -------
        optimum_variable: list
            Best design scheme

        """
        return self.__class__.optimum_scheme

    def get_design_space(self) -> dict:
        """

        Returns
        -------
        design_space:

        """

        return self.__class__.design_space

    def plot_function(self,
                      with_low_fidelity: bool,
                      save: bool = True) -> None:
        """
        Function to visualize the landscape of the function s
        Parameters
        ----------
        save : bool
            save figure or not

        Returns
        -------

        """

        num_dim = self._get_dimension()
        num_plot = 1000
        if num_dim == 1:
            # draw the samples from design space
            x_plot = np.linspace(start=self.__class__.low_bound[0],
                                 stop=self.__class__.high_bound[0],
                                 num=num_plot)
            # y_plot_hf = self.__class__.__call__(self, x=x_plot)

            with plt.style.context(['ieee', 'science']):
                fig, ax = plt.subplots()
                ax.plot(x_plot,
                        self.__class__.__call__(self,
                                                x=x_plot,
                                                fidelity='high'),
                        label=f'{self.__class__.__name__}')
                if with_low_fidelity is True:
                    low_fidelity = self._get_low_fidelity()
                    for ii in range(len(low_fidelity)):
                        ax.plot(x_plot,
                                self.__class__.__call__(self,
                                                        x=x_plot,
                                                        fidelity=low_fidelity[ii]),
                                '--', label=f'{low_fidelity[ii]}')
                ax.legend()
                ax.set(xlabel=r'$x$')
                ax.set(ylabel=r'$y$')
                ax.autoscale(tight=True)
                if save is True:
                    fig.savefig(self.__class__.__name__, dpi=300)
                plt.show(block=True)
                plt.interactive(False)
        elif num_dim == 2:
            # TODO: contour plot for 2D case
            pass

        else:
            raise ValueError("Unexpected value of 'num_dimension'!", num_dim)

    def _get_dimension(self) -> int:
        """
        Get dimension of the function

        Returns
        -------
        dimension: int
            dimension of the problem
        """
        return self.__class__.num_dim

    def _get_low_fidelity(self) -> list:
        """
        Get names of low fidelity functions
        Returns
        -------
        name: list
            name list of low fidelity functions

        """

        return self.__class__.low_fidelity
