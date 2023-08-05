import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats.qmc import LatinHypercube, Sobol

# local modulus
from mfpml.design_of_experiment.sampler import Sampler


class MultiFidelitySampler(Sampler):
    """
    Base class for multi-fidelity sampling
    """

    def __init__(self, design_space: dict, seed: int) -> None:
        """
        Parameters
        ----------
        design_space: dict
            design_space
        seed: int
            seed s
        """
        super(MultiFidelitySampler, self).__init__(
            design_space=design_space, seed=seed
        )

        self._lf_samples = None
        self._hf_samples = None
        self.num_lf_samples = None
        self.num_hf_samples = None
        self.nested = None

    def _mf_samples_rules(self) -> None:
        """
        number of low fidelity samples should larger than high fidelity samples
        Returns
        -------
        """
        assert (self.num_lf_samples >= self.num_hf_samples), \
            "samples of low fidelity should larger than tha of high fidelity"

    def get_samples(self, **kwargs) -> dict:
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

        raise NotImplementedError("Subclasses should implement this method.")

    def _create_pandas_frame(self) -> None:
        """
        this function is used to create pandas framework for the doe
        the output will be added at the end of the pandas dataframe but
        without giving names
        Parameters
        ----------
        Returns
        -------
        samples: pd.DataFrame
            pd.DataFrame format of samples
        """

        self._lf_samples = pd.DataFrame(
            self._lf_samples, columns=list(self.design_space.keys())
        )
        self._hf_samples = pd.DataFrame(
            self._hf_samples, columns=list(self.design_space.keys())
        )

    def plot_samples(
        self, figure_name: str = 'mf.png', save_plot: bool = False, **kwargs
    ) -> None:
        """
        Visualization of mf sampling
        Parameters
        ----------
        figure_name: str
            figure name
        save_plot: bool
            save figure or not
        Returns
        -------
        """
        if self.num_dim == 2:
            fig, ax = plt.subplots(**kwargs)
            # plot the low fidelity samples
            ax.plot(
                self.lf_samples.iloc[:, 0],
                self.lf_samples.iloc[:, 1],
                "*",
                label="LF Samples",
            )
            ax.plot(
                self.hf_samples.iloc[:, 0],
                self.hf_samples.iloc[:, 1],
                "o",
                mfc="none",
                label="HF Samples",
            )
            ax.legend()
            ax.set(xlabel=r"$x_1$")
            ax.set(ylabel=r"$x_2$")
            plt.grid()
            plt.show()
            if save_plot is True:
                fig.savefig(figure_name, dpi=300)

        elif self.num_dim == 1:
            fig, ax = plt.subplots(**kwargs)
            ax.plot(
                self.lf_samples.iloc[:, 0],
                np.zeros((self.lf_samples.shape[0], 1)),
                ".",
                label="LF Samples",
            )
            ax.plot(
                self.hf_samples.iloc[:, 0],
                np.zeros((self.hf_samples.shape[0], 1)),
                "o",
                mfc="none",
                label="HF Samples",
            )
            ax.legend()
            ax.set(xlabel=r"$x$")
            ax.set(ylabel=r"$y$")
            plt.show()
            if save_plot is True:
                fig.savefig(figure_name, dpi=300)

        else:
            raise Exception("Can not plot figure more than two dimension! \n ")

        pass

    @property
    def lf_samples(self) -> pd.DataFrame:
        """
        get the low fidelity samples
        Returns
        -------
        lf_samples: pd.DataFrame
            low fidelity samples
        """
        return self._lf_samples

    @property
    def hf_samples(self) -> pd.DataFrame:
        """
        get the high fidelity samples
        Returns
        -------
        hf_samples: pd.DataFrame
            high fidelity samples
        """
        return self._hf_samples

    @property
    def data(self) -> dict:
        """
        samples for multi-fidelity
        Returns
        -------
        data: dict
            samples includes high fidelity and low fidelity
        """
        return {"hf": self._hf_samples, "lf": self._lf_samples}


class MFLatinHyperCube(MultiFidelitySampler):
    """
    Multi-fidelity Latin HyperCube sampling
    """

    def __init__(
        self, design_space: dict, nested: bool = False, seed: int = None
    ) -> None:
        super(MFLatinHyperCube, self).__init__(
            design_space=design_space, seed=seed
        )
        self.nested = nested

    def get_samples(self, num_samples: int = None, **kwargs) -> dict:
        self.num_lf_samples = kwargs["num_lf_samples"]
        self.num_hf_samples = kwargs["num_hf_samples"]
        self._mf_samples_rules()
        # get the samples of low fidelity first
        lf_sample = self.__get_lf_samples()

        # check the user wants nested samples of high fidelity or not
        if self.nested is True:
            hf_sample = self.__get_nested_hf_samples()
        else:
            hf_sample = self.__get_non_nested_hf_samples()

        # transfer samples into pd.DataFrame
        self._create_pandas_frame()

        return {"hf": hf_sample, "lf": lf_sample}

    def __get_lf_samples(self):
        """
        Generate low fidelity samples using LHS
        Returns
        -------
        """

        lhs_sampler = LatinHypercube(d=self.num_dim, seed=self.seed)

        lf_sample = lhs_sampler.random(n=self.num_lf_samples)
        for i, bounds in enumerate(self.design_space.values()):
            lf_sample[:, i] = (
                lf_sample[:, i] * (bounds[1] - bounds[0]) + bounds[0]
            )
        self._lf_samples = lf_sample

        return lf_sample

    def __get_non_nested_hf_samples(self) -> None:
        """
        use  another LHS to generate samples for high fidelity
        Returns
        -------
        """
        if self.seed is None:
            lhs_sampler = LatinHypercube(d=self.num_dim, seed=None)
        else:
            lhs_sampler = LatinHypercube(d=self.num_dim, seed=self.seed + 1)

        hf_sample = lhs_sampler.random(n=self.num_hf_samples)
        for i, bounds in enumerate(self.design_space.values()):
            hf_sample[:, i] = (
                hf_sample[:, i] * (bounds[1] - bounds[0]) + bounds[0]
            )
        self._hf_samples = hf_sample

        return hf_sample

    def __get_nested_hf_samples(self) -> None:
        pass


class MFSobolSequence(MultiFidelitySampler):
    """
    Multi-fidelity sobol sequence sampling
    """

    def __init__(
        self,
        design_space: dict,
        nested: bool = False,
        seed: int = 123456,
        num_skip: int = None,
    ) -> None:
        super(MFSobolSequence, self).__init__(
            design_space=design_space, seed=seed
        )
        self.nested = nested
        if num_skip is None:
            self.num_skip = len(design_space)
        else:
            self.num_skip = num_skip

    def get_samples(self, num_samples: int = None, **kwargs) -> dict:
        self.num_lf_samples = kwargs["num_lf_samples"]
        self.num_hf_samples = kwargs["num_hf_samples"]
        self._mf_samples_rules()
        # get the samples of low fidelity first
        lf_sample = self.__get_lf_samples()

        # check the user wants nested samples of high fidelity or not
        if self.nested is True:
            hf_sample = self.__get_nested_hf_samples()
        else:
            hf_sample = self.__get_non_nested_hf_samples()

        # transfer samples into pd.DataFrame
        self._create_pandas_frame()

        return {"hf": hf_sample, "lf": lf_sample}

    def __get_lf_samples(self) -> np.ndarray:
        """
        Returns
        -------
        """
        sobol_sampler = Sobol(d=self.num_dim, seed=self.seed)
        _ = sobol_sampler.reset()
        _ = sobol_sampler.fast_forward(n=self.num_skip)
        # get the samples
        lf_sample = sobol_sampler.random(n=self.num_lf_samples)
        for i, bounds in enumerate(self.design_space.values()):
            lf_sample[:, i] = (
                lf_sample[:, i] * (bounds[1] - bounds[0]) + bounds[0]
            )
        self._lf_samples = lf_sample

        return lf_sample

    def __get_nested_hf_samples(self):
        """
        Returns
        -------
        """
        hf_sample = self._lf_samples[0: self.num_hf_samples, :]
        self._hf_samples = hf_sample
        return hf_sample

    def __get_non_nested_hf_samples(self):
        """
        Returns
        -------
        """
        sobol_sampler = Sobol(d=self.num_dim, seed=self.seed + 1)
        _ = sobol_sampler.reset()
        _ = sobol_sampler.fast_forward(n=self.num_skip)
        # get the samples
        hf_sample = sobol_sampler.random(n=self.num_hf_samples)
        for i, bounds in enumerate(self.design_space.values()):
            hf_sample[:, i] = (
                hf_sample[:, i] * (bounds[1] - bounds[0]) + bounds[0]
            )
        self._hf_samples = hf_sample

        return hf_sample
