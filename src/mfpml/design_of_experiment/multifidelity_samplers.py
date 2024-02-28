import pickle
from typing import Dict, List

import numpy as np
from scipy.stats.qmc import LatinHypercube, Sobol


class MultiFidelitySampler:
    """
    Base class for multi-fidelity sampling
    """

    def __init__(self,
                 design_space: List | np.ndarray,
                 num_fidelity: int = None,
                 nested: bool = False) -> None:
        """
        Parameters
        ----------
        design_space: List
            design space
        """

        # A list is expected that contains the design space of each fidelity
        if isinstance(design_space, np.ndarray):
            self.num_fidelity = num_fidelity
            self.num_dim = len(design_space)
            self.design_space = self.transfer_design_space_list(design_space)
        elif isinstance(design_space, List):
            self.design_space = design_space
            # get number of fidelity levels
            self.num_fidelity = len(design_space)
            # get number of dimensions
            self.num_dim = len(design_space[0])
        else:
            raise TypeError("design space should be a list or numpy array")
        # nested or not
        self.nested = nested

    def get_samples(self,
                    num_samples: List,
                    seed: int = 123456,
                    **kwargs) -> Dict:
        """
        Get the samples

        Parameters
        ----------
        num_samples : List
            number of samples for each fidelity level

        seed : int, optional
            random seed, by default 123456

        """
        # record the seed for reproducibility
        self.seed = seed

        # initialize the samples
        data = []
        # check the number of samples
        self._mf_samples_rules(num_samples)

        # get the samples of low fidelity first
        data.insert(0, self._get_non_nested_samples(
            num_sample=num_samples.pop(),
            fidelity=len(num_samples)))

        # get samples sequentially
        while len(num_samples) > 0:
            if self.nested is True:
                # nested samples
                sample = self._get_nested_samples(
                    num_sample=num_samples.pop(),
                    data_higher_fidelity=data[0])
            else:
                # non-nested samples
                sample = self._get_non_nested_samples(
                    num_sample=num_samples.pop(),
                    fidelity=len(num_samples))

            # insert the samples
            data.insert(0, sample)
        # assign data to self.data
        self.data = data

        return data

    def _mf_samples_rules(self, num_samples: List) -> None:
        """
        number of low fidelity samples should larger than high fidelity samples

        Parameters
        ----------
        num_samples : List
            number of samples for each fidelity level

        """
        for i in range(len(num_samples)-1):
            assert (num_samples[i] <= num_samples[i+1]), \
                "samples of low fidelity should larger than tha of high fidelity"

    def lb(self, fidelity: int = 0) -> np.ndarray:
        """lower bound of the design space

        Parameters
        ----------
        fidelity : int, optional
            fidelity level, by default 0

        Returns
        -------
        np.ndarray
            lower bound of the design space at fidelity level `fidelity`
        """
        return self.design_space[fidelity][:, 0]

    def ub(self, fidelity: int = 0) -> np.ndarray:
        """upper bound of the design space

        Parameters
        ----------
        fidelity : int, optional
            fidelity level, by default 0

        Returns
        -------
        np.ndarray
            upper bound of the design space at fidelity level `fidelity`
        """
        return self.design_space[fidelity][:, 1]

    def transfer_design_space_list(self, design_space) -> List:
        """transfer design space to list

        Returns
        -------
        List
            design space
        """
        return [design_space for i in range(self.num_fidelity)]

    def save_data(self, file_name: str = "mf_data") -> None:
        """
        This function is used to save the design_of_experiment to Json files

        Parameters
        ----------
        file_name:str
            name for the pickle.file

        Returns
        -------

        """

        with open(file_name + ".pickle", "wb") as file:
            pickle.dump(self.data, file)


class MFLatinHyperCube(MultiFidelitySampler):
    """
    Multi-fidelity Latin HyperCube sampling
    """

    def __init__(
        self, design_space: List,
        num_fidelity: int = None,  # type: ignore
        nested: bool = False,
    ) -> None:
        """Initialization

        Parameters
        ----------
        design_space : List
            design space of the problem
        num_fidelity : int, optional
            number of fidelity levels, by default None
        nested : bool, optional
            nested sampling or not, by default False
        """
        super(MFLatinHyperCube, self).__init__(design_space=design_space,
                                               num_fidelity=num_fidelity,
                                               nested=nested)

    def _get_non_nested_samples(self,
                                num_sample: int,
                                fidelity: int = 0) -> np.ndarray:
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
            sampler = LatinHypercube(d=self.num_dim, seed=self.seed + fidelity)

        samples = sampler.random(n=num_sample)

        samples = samples * (self.ub(fidelity) -
                             self.lb(fidelity)) + self.lb(fidelity)

        return samples

    def _get_nested_samples(self,
                            num_sample: int,
                            data_higher_fidelity) -> None:

        # generate index randomly for nested sampling without repeating
        index = np.random.choice(
            np.arange(data_higher_fidelity.shape[0]),
            size=num_sample,
            replace=False
        )

        return data_higher_fidelity[index, :]


class MFSobolSequence(MultiFidelitySampler):
    """
    Multi-fidelity Sobol Sequence sampling
    """

    def __init__(
        self,
        design_space: List | np.ndarray,
        num_fidelity: int = None,
        nested: bool = False,
        num_skip: int = None,  # type: ignore
    ) -> None:
        """Initialization

        Parameters
        ----------
        design_space : dict
            design space of the problem
        nested : bool, optional
            nested sampling or not, by default False
        num_skip : int, optional
            number of skip, by default None

        """
        super(MFSobolSequence, self).__init__(design_space=design_space,
                                              num_fidelity=num_fidelity,
                                              nested=nested)

        if num_skip is None:
            self.num_skip = len(design_space)
        else:
            self.num_skip = num_skip

    def _get_non_nested_samples(self,
                                num_sample: int,
                                fidelity: int = 0) -> np.ndarray:
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
        samples = sobol_sampler.random(n=num_sample)

        # scale the samples
        samples = samples * (self.ub(fidelity) -
                             self.lb(fidelity)) + self.lb(fidelity)

        return samples

    def _get_nested_samples(self,
                            num_sample: int,
                            data_higher_fidelity) -> np.ndarray:
        """

        Returns
        -------
        hf_sample : np.ndarray
            a numpy array contains high fidelity samples

        """
        return data_higher_fidelity[0: num_sample, :]
