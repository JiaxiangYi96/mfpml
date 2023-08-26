
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats.qmc import LatinHypercube, Sobol

# local modulus
from mfpml.design_of_experiment.sampler import Sampler


class SingleFidelitySampler(Sampler):
    def __init__(self, design_space: dict, seed: int) -> None:
        """

        Parameters
        ----------
        design_space: dict
            design space
        seed: int
            seed
        Returns
        ----------

        """
        super(SingleFidelitySampler, self).__init__(
            design_space=design_space, seed=seed
        )

    def _create_pandas_frame(self) -> None:
        """
        create pandas frame for samples
        Returns
        -------


        """
        self._samples = pd.DataFrame(
            self._samples, columns=list(self.design_space.keys())
        )

    def get_samples(self, num_samples: int, **kwargs) -> np.ndarray:
        """
        Get the samples

        Parameters
        ----------
        num_samples: int
            number of samples

        Returns
        ---------
        samples: np.ndarray
          samples generated from sampling methods

        Notes
        ---------
        The function should be completed at the sub-sclass

        """

        raise NotImplementedError("Subclasses should implement this method.")

    def plot_samples(
        self, fig_name: str = "sf_sampler",
        save_sig: bool = False, **kwargs
    ) -> None:
        """
        Visualization of sampling method

        Parameters
        ----------
        figure_name: str
            figure name
        save_plot: bool
            save the figure or not

        Returns
        -------

        Examples
        --------
        >>> design_space = {'x1': [0, 1], 'x2': [0, 1]}
        >>> sampler = LatinHyperCube(design_space=design_space, seed=123456)
        >>> samples = sampler.get_samples(num_samples=10)
        >>> sampler.plot_samples(figure_name='sampler', save_plot=False)

        """

        if self.num_dim == 2:
            # two dimensional plot
            fig, ax = plt.subplots(**kwargs)
            ax.plot(
                self.samples.iloc[:, 0],
                self.samples.iloc[:, 1],
                "*",
                label="Samples",
            )
            ax.legend()
            ax.set(xlabel=r"$x_1$")
            ax.set(ylabel=r"$x_2$")
            plt.grid()
            plt.show()
            if save_sig is True:
                fig.savefig(fig_name, dpi=300)

        elif self.num_dim == 1:
            # one dimensional plot
            fig, ax = plt.subplots(**kwargs)
            ax.plot(
                self.samples.iloc[:, 0],
                np.zeros((self.samples.shape[0], 1)),
                ".",
                label="Samples",
            )
            ax.legend()
            ax.set(xlabel=r"$x$")
            ax.set(ylabel=r"$y$")
            ax.autoscale(tight=True)
            plt.grid()
            plt.show()
            if save_sig is True:
                fig.savefig(fig_name, dpi=300)

        else:
            raise Exception("Can not plot figure more than two dimension! \n ")

    @property
    def samples(self) -> pd.DataFrame:
        """get samples

        Returns
        -------
        samples : pd.DataFrame
            a pandas dataframe of samples
        """
        return self._samples

    @property
    def data(self) -> dict[str, pd.DataFrame]:
        """get data"""
        return {"inputs": self._samples}


class FixNumberSampler(SingleFidelitySampler):
    def __init__(self, design_space: dict, seed: int) -> None:
        """

        Parameters
        ----------
        design_space: dict
            design space
        seed: int
            seed

        Returns
        -------
        """
        super(FixNumberSampler, self).__init__(
            design_space=design_space, seed=seed
        )

    def get_samples(self, num_samples: int, **kwargs) -> dict:
        """

        Parameters
        ----------
        num_samples: int
            number of samples
        kwargs: additional info

        Returns
        -------
        data: dict
            samples

        Examples
        --------
        >>> design_space = {'x1': [0, 1], 'x2': [0, 1]}
        >>> sampler = FixNumberSampler(design_space=design_space, seed=123456)
        >>> samples = sampler.get_samples(num_samples=2)
        >>> samples
        {'inputs':    x1   x2
        0  0.5  0.5
        1  0.5  0.5}

        """
        fixed_value = list(self.design_space.values())
        samples = np.repeat(fixed_value[0], num_samples).reshape(
            (-1, self.num_dim)
        )
        self._samples = samples.copy()
        self._create_pandas_frame()

        return samples


class LatinHyperCube(SingleFidelitySampler):
    """
    Latin Hyper cube sampling
    """

    def __init__(self, design_space: dict, seed: int) -> None:
        """

        Parameters
        ----------
        design_space: dict
            design space
        seed: int
            seed
        """
        super(LatinHyperCube, self).__init__(
            design_space=design_space, seed=seed
        )

    def get_samples(self, num_samples: int, **kwargs) -> np.ndarray:
        """get samples

        Parameters
        ----------
        num_samples : int
            number of samples

        Returns
        -------
        sample : np.ndarray
            a numpy array of samples

        Examples
        --------
        >>> design_space = {'x1': [0, 1], 'x2': [0, 1]}
        >>> sampler = LatinHyperCube(design_space=design_space, seed=123456)
        >>> samples = sampler.get_samples(num_samples=2)
        >>> samples
        array([[0.5, 0.5],
                [0.5, 0.5]])
        """
        lhs_sampler = LatinHypercube(d=self.num_dim, seed=self.seed)
        sample = lhs_sampler.random(num_samples)
        for i, bounds in enumerate(self.design_space.values()):
            sample[:, i] = sample[:, i] * (bounds[1] - bounds[0]) + bounds[0]

        self._samples = sample
        self._create_pandas_frame()
        return sample


class RandomSampler(SingleFidelitySampler):
    """
    Random sampling
    """

    def __init__(self, design_space: dict, seed: int) -> None:
        """

        Parameters
        ----------
        design_space: dict
            design space
        seed: int
            seed
        """
        super(RandomSampler, self).__init__(
            design_space=design_space, seed=seed
        )

    def get_samples(self, num_samples: int, **kwargs) -> np.ndarray:
        """get samples

        Parameters
        ----------
        num_samples : int
            number of samples

        Returns
        -------
        sample : np.ndarray
            a numpy array of samples

        Examples
        --------
        >>> design_space = {'x1': [0, 1], 'x2': [0, 1]}
        >>> sampler = RandomSampler(design_space=design_space, seed=123456)
        >>> samples = sampler.get_samples(num_samples=2)
        >>> samples
        array([[0.12696982, 0.96671784],
                [0.26047601, 0.89750289]])

        """
        np.random.seed(self.seed)
        sample = np.random.random((num_samples, self.num_dim))
        for i, bounds in enumerate(self.design_space.values()):
            sample[:, i] = sample[:, i] * (bounds[1] - bounds[0]) + bounds[0]

        self._samples = sample
        self._create_pandas_frame()
        return sample


class SobolSequence(SingleFidelitySampler):
    """
    Sobol Sequence sampling
    """

    def __init__(
        self, design_space: dict, seed: int, num_skip: int = None
    ) -> None:
        """

        Parameters
        ----------
        design_space: dict
            design space
        seed: int
            seed
        num_skip: int
            cut the first several samples s
        """
        super(SobolSequence, self).__init__(
            design_space=design_space, seed=seed
        )
        if num_skip is None:
            self.num_skip = len(design_space)
        else:
            self.num_skip = num_skip

    def get_samples(self, num_samples: int, **kwargs) -> np.ndarray:
        """get samples

        Parameters
        ----------
        num_samples : int
            number of samples

        Returns
        -------
        sample : np.ndarray
            a numpy array of samples

        Examples
        --------
        >>> design_space = {'x1': [0, 1], 'x2': [0, 1]}
        >>> sampler = SobolSequence(design_space=design_space, seed=123456)
        >>> samples = sampler.get_samples(num_samples=2)
        >>> samples
        array([[0.5, 0.5],
                [0.5, 0.5]])
        """
        sobol_sampler = Sobol(d=self.num_dim, seed=self.seed)
        # generate lots of samples first
        sobol_sampler.random_base2(m=num_samples)
        _ = sobol_sampler.reset()
        _ = sobol_sampler.fast_forward(n=self.num_skip)
        # get the samples
        sample = sobol_sampler.random(n=num_samples)

        for i, bounds in enumerate(self.design_space.values()):
            sample[:, i] = sample[:, i] * (bounds[1] - bounds[0]) + bounds[0]

        self._samples = sample
        self._create_pandas_frame()

        return sample
