# third-party
import os
import numpy as np
import pandas as pd
from SALib.sample import sobol_sequence
from scipy.stats import qmc
from matplotlib import pyplot as plt


class Sampler:

    """Arg:
        design_space
        num_simples
    """

    def __init__(self):
        self.seed = None
        self.design_space = None
        self.num_samples = None
        self.num_dim = None
        self.samples = None
        self.num_outs = None
        self.out_names = None

    def SetSeed(self, seed: int = 123456) -> None:
        """"
            This function is used to get the design space info
            and the number of dimension of this problem
        Arg:
            design_space : a dict that contain the design space

        Note:  the function should be completed at the subsclass
        """
        self.seed = seed

        pass

    def SetDesignSpace(self, design_space: dict = None) -> None:

        """"
            This function is used to get the design space info
            and the number of dimension of this problem
        Arg:
            design_space : a dict that contain the design space

        Note:  the function should be completed at the subsclass
        """
        self.design_space = design_space
        self.num_dim = len(design_space)

    def GetSamples(self, num_samples: int = None) -> None:
        """"
            This function is used to get the samples
        Arg:
            num_samples : a dict that contain the design space

        Note:  the function should be completed at the subsclass
        """

        pass

    def CreatePandasFrame(self, out_names: list) -> None:
        """
        this function is used to create pandas framework for the doe
        the output will be added at the end of the pandas dataframe but without giving names

        Returns
        -------
        None
        """
        # load the number of outputs and corresponding names
        self.out_names = out_names
        self.num_outs = len(out_names)

        # transfer the variables to a pandas dataframe
        self.samples = pd.DataFrame(self.samples, columns=list(self.design_space.keys()))
        self.samples[self.out_names] = np.nan
        self.samples[self.out_names] = self.samples[self.out_names].astype(object)

    def SaveDoE(self, name: str = 'doe') -> None:
        """
        This function is used to save the DoE to Json files
        Returns
        -------

        """
        self.samples.to_json(name + '.json', index=True)

    def PlotSamples2D(self, name: str = None, saveplot: bool = False) -> None:
        """
        This function is used to visualize the two-dimensional sampling
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

    def PlotSamples1D(self, name: str = None, saveplot: bool = False) -> None:
        """
        function to visualize the one dimension design of experiment
        Parameters
        ----------
        name: name of figure
        saveplot: bool operator to claim save the figure or not

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


class LatinHyperCube(Sampler):

    def Sampling(self, num_samples: int,
                 design_space: dict,
                 seed: int,
                 out_names: dict) -> pd.DataFrame:
        self.SetSeed(seed=seed)
        self.SetDesignSpace(design_space=design_space)
        self.GetSamples(num_samples=num_samples)
        self.CreatePandasFrame(out_names=out_names)

        return self.samples

    def GetSamples(self, num_samples: int = None) -> np.ndarray:
        self.num_samples = num_samples
        sampler = qmc.LatinHypercube(d=self.num_dim)
        sample = sampler.random(self.num_samples)
        for i, bounds in enumerate(self.design_space.values()):
            sample[:, i] = sample[:, i] * (bounds[1] - bounds[0]) + bounds[0]

        self.samples = sample


class SobolSequence(Sampler):

    def Sampling(self, num_samples: int,
                 design_space: dict,
                 seed: int,
                 out_names: dict) -> pd.DataFrame:
        self.SetSeed(seed=seed)
        self.SetDesignSpace(design_space=design_space)
        self.GetSamples(num_samples=num_samples)
        self.CreatePandasFrame(out_names=out_names)

        return self.samples

    def GetSamples(self, num_samples: int = None) -> np.ndarray:
        self.num_samples = num_samples
        sample = sobol_sequence.sample(self.num_samples, self.num_dim)
        for i, bounds in enumerate(self.design_space.values()):
            sample[:, i] = sample[:, i] * (bounds[1] - bounds[0]) + bounds[0]
        self.samples = sample


class RandomSampler(Sampler):

    def Sampling(self, num_samples: int,
                 design_space: dict,
                 seed: int,
                 out_names: dict) -> pd.DataFrame:
        self.SetSeed(seed=seed)
        self.SetDesignSpace(design_space=design_space)
        self.GetSamples(num_samples=num_samples)
        self.CreatePandasFrame(out_names=out_names)

        return self.samples

    def GetSamples(self, num_samples: int = None) -> np.ndarray:
        self.num_samples = num_samples
        np.random.seed(self.seed)
        sample = np.random.random((self.num_samples, self.num_dim))
        for i, bounds in enumerate(self.design_space.values()):
            sample[:, i] = sample[:, i] * (bounds[1] - bounds[0]) + bounds[0]

        self.samples = sample


class FixNumberSampler(Sampler):

    def Sampling(self, num_samples: int,
                 design_space: dict,
                 out_names: dict) -> pd.DataFrame:
        self.SetDesignSpace(design_space=design_space)
        self.GetSamples(num_samples=num_samples)
        self.CreatePandasFrame(out_names=out_names)
        return self.samples

    def GetSamples(self, num_samples: int = None) -> np.ndarray:
        self.num_samples = num_samples
        fixedvalue = list(self.design_space.values())
        sample = np.repeat(fixedvalue[0], num_samples)
        self.samples = sample
