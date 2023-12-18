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
        design_space: np.ndarray or list
            design space
        seed: int
            seed s
        """
        super(MultiFidelitySampler, self).__init__(
            design_space=design_space, seed=seed
        )
        self.nested = None

    def _mf_samples_rules(self, num_samples: list) -> None:
        """
        number of low fidelity samples should larger than high fidelity samples
        """
        for i in range(len(num_samples)-1):
            assert (num_samples[i] <= num_samples[i+1]), \
                "samples of low fidelity should larger than tha of high fidelity"

    def get_samples(self, num_samples: list, **kwargs) -> dict:
        """
        Get the samples
        
        Parameters
        ----------
        num_samples : list
            number of samples for each fidelity level

        Notes
        ---------
        The function should be completed at the sub-sclass
        """
        data = []
        self._mf_samples_rules(num_samples)
        # get the samples of low fidelity first
        data.insert(0, self._get_nonested_samples(num_samples.pop()))
        while len(num_samples) > 0: 
            if self.nested is True:
                # TODO
                data.insert(0, self._get_nested_samples(num_samples.pop(), data[0]))
            else:
                data.insert(0, self._get_nonested_samples(num_samples.pop()))

        # transfer samples into pd.DataFrame
        # self._create_pandas_frame()

        return data

    # def _create_pandas_frame(self) -> None:
    #     """
    #     this function is used to create pandas framework for the doe
    #     the output will be added at the end of the pandas dataframe but
    #     without giving names
    #     """

    #     self._lf_samples = pd.DataFrame(
    #         self._lf_samples, columns=list(self.design_space.keys())
    #     )
    #     self._hf_samples = pd.DataFrame(
    #         self._hf_samples, columns=list(self.design_space.keys())
    #     )

    def plot_samples(
        self, fig_name: str = 'mf.png', save_fig: bool = False, **kwargs
    ) -> None:
        # TODO
        """
        Visualization of mf sampling
        Parameters
        ----------
        fig_name: str
            figure name
        save_fig: bool
            save figure or not

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
            if save_fig is True:
                fig.savefig(fig_name, dpi=300, bbox_inches="tight")

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
            if save_fig is True:
                fig.savefig(fig_name, dpi=300, bbox_inches="tight")

        else:
            raise Exception("Can not plot figure more than two dimension! \n ")

        pass


class MFLatinHyperCube(MultiFidelitySampler):
    """
    Multi-fidelity Latin HyperCube sampling
    """

    def __init__(
        self, design_space: dict, nested: bool = False, seed: int = None
    ) -> None:
        """Initialization

        Parameters
        ----------
        design_space : dict
            design space of the problem
        nested : bool, optional
            nested sampling or not, by default False
        seed : int, optional
            seed, by default None
        """
        super(MFLatinHyperCube, self).__init__(
            design_space=design_space, seed=seed
        )
        self.nested = nested

    def _get_nonested_samples(self, num_sample) -> np.ndarray:
        """
        get low fidelity samples

        Returns
        -------
        lf_samples : np.ndarray
            a numpy array contains low fidelity samples
        """
        if self.seed is None:
            sampler = LatinHypercube(d=self.num_dim, seed=None)
        else:
            sampler = LatinHypercube(d=self.num_dim, seed=self.seed)

        sample = sampler.random(n=num_sample)
        sample = sample * (self.ub - self.lb) + self.lb
        # for i, bounds in enumerate(self.design_space.values()):
        #     sample[:, i] = (
        #         sample[:, i] * (bounds[1] - bounds[0]) + bounds[0]
        #     )

        return sample

    def _get_nested_samples(self, num_sample, data_higher_fidelity) -> None:
        # TODO nested sampling
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
        """Initialization

        Parameters
        ----------
        design_space : dict
            design space of the problem
        nested : bool, optional
            nested sampling or not, by default False
        seed : int, optional
            seed, by default 123456
        num_skip : int, optional
            number of skip, by default None

        """
        super(MFSobolSequence, self).__init__(
            design_space=design_space, seed=seed
        )
        self.nested = nested
        if num_skip is None:
            self.num_skip = len(design_space)
        else:
            self.num_skip = num_skip

    def _get_nonested_samples(self, num_sample) -> np.ndarray:
        """get low fidelity samples

        Returns
        -------
        lf_samples : np.ndarray
            a numpy array contains low fidelity samples
        """
        sobol_sampler = Sobol(d=self.num_dim, seed=self.seed)
        _ = sobol_sampler.reset()
        _ = sobol_sampler.fast_forward(n=self.num_skip)
        # get the samples
        sample = sobol_sampler.random(n=num_sample)
        sample = sample * (self.ub - self.lb) + self.lb
        # for i, bounds in enumerate(self.design_space.values()):
        #     sample[:, i] = (
        #         sample[:, i] * (bounds[1] - bounds[0]) + bounds[0]
        #     )

        return sample

    def _get_nested_hf_samples(self, num_sample, data_higher_fidelity) -> np.ndarray:
        """

        Returns
        -------
        hf_sample : np.ndarray
            a numpy array contains high fidelity samples

        """
        return data_higher_fidelity[0: self.num_sample, :]



if __name__ == '__main__':

    # print(LatinHypercube(d=1, seed=None).random(n=1))

    # from mfpml.design_of_experiment.space import DesignSpace
    # design_space = DesignSpace(low_bound=[5.]*5, high_bound=[10.]*5)

    # method = MFLatinHyperCube(design_space=[[5., 10.]]*5)
    method = MFSobolSequence(design_space=[[5., 10.]]*5, nested=1)
    print(method.get_samples([3, 5, 10]))

    print()