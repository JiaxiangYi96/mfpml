# third-party
import numpy as np
import pandas as pd
from dataclasses import dataclass
from matplotlib import pyplot as plt


@dataclass
class Sampler:
    """
    Class for drawing samples from design space

    """
    seed = None
    design_space = None
    num_samples = None
    num_dim = None
    samples = None
    num_outs = None
    out_names = None

    def set_seed(self, seed: int = 123456) -> None:
        """
        This function is used to get the design space info
        and the number of dimension of this problem

        Parameters
        ----------
        seed: int
            seed for sampler, for data-replication

        Returns
        -------

        """

        self.seed = seed

    def set_design_space(self, design_space: dict = None) -> None:
        """
        This function is used to get the design space info
        and the number of dimension of this problem

        Parameters
        ----------
        design_space : dict
        design space inside a dictionary

        Returns
        -------

        """

        self.design_space = design_space
        self.num_dim = len(design_space)

    def get_samples(self, num_samples: int = None) -> np.ndarray:
        """
        Get the samples

        Parameters
        ----------
        num_samples: int
            number of samples

        Returns
        ---------

        Notes
        ---------
        The function should be completed at the sub-sclass

        """

        pass

    def create_pandas_frame(self, out_names: list) -> None:
        """
        this function is used to create pandas framework for the doe
        the output will be added at the end of the pandas dataframe but without giving names

        Parameters
        ----------
        out_names : list

        Returns
        -------
        """

        # load the number of outputs and corresponding names
        self.out_names = out_names
        self.num_outs = len(out_names)

        # transfer the variables to a pandas dataframe
        self.samples = pd.DataFrame(self.samples, columns=list(self.design_space.keys()))
        self.samples[self.out_names] = np.nan
        self.samples[self.out_names] = self.samples[self.out_names].astype(object)

    def sampling(self, num_samples: int,
                 design_space: dict,
                 seed: int,
                 out_names: list) -> pd.DataFrame:
        """
        Sampling interface to the user

        Parameters
        ----------
        num_samples: int
            number of samples
        design_space:dict
            design space of this problem
        seed: int
            seed for sampler, for replicating the same results
        out_names : list
            names of the outputs

        Returns
        -------
        samples : pandas.DataFrame
            a pandas dataframe to that contains the information of samples

        """

        self.set_seed(seed=seed)
        self.set_design_space(design_space=design_space)
        self.get_samples(num_samples=num_samples)
        self.create_pandas_frame(out_names=out_names)

        return self.samples

    def save_doe(self, name: str = 'doe') -> None:
        """
        This function is used to save the DoE to Json files

        Parameters
        ----------
        name:str
            name for the Json.file

        Returns
        -------

        """

        self.samples.to_json(name + '.json', index=True)

    def plot_samples_2D(self, name: str = None, saveplot: bool = False) -> None:

        """
        function to visualize the one dimension design of experiment
        Parameters
        ----------
        name: str
            name of the figure file
        saveplot: bool
            operator to claim save the figure or not

        Returns
        -------

        """

        with plt.style.context(['ieee', 'science']):
            fig, ax = plt.subplots()
            ax.plot(self.samples.values[:, 0], self.samples.values[:, 1], '*', label='Samples')
            ax.legend()
            ax.set(xlabel=r'$x_1$')
            ax.set(ylabel=r'$x_2$')
            ax.autoscale(tight=True)
            plt.show()
            if saveplot is True:
                fig.savefig(name, dpi=300)

    def plot_samples_1D(self, name: str = None, saveplot: bool = False) -> None:
        """
        function to visualize the one dimension design of experiment
        Parameters
        ----------
        name: str
            name of the figure file
        saveplot: bool
            operator to claim save the figure or not

        Returns
        -------

        """

        with plt.style.context(['ieee', 'science']):
            fig, ax = plt.subplots()
            ax.plot(self.samples.values[:, 0], np.zeros((self.num_samples, 1)), '.', label='Samples')
            ax.legend()
            ax.set(xlabel=r'$x_1$')
            ax.set(ylabel=r'$y$')
            ax.autoscale(tight=True)
            plt.show()
            if saveplot is True:
                fig.savefig(name, dpi=300)
