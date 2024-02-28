
import pickle
from abc import ABC
from typing import Any, List

import numpy as np
from scipy.stats.qmc import LatinHypercube, Sobol


class SingleFidelitySampler(ABC):
    """
    Class for drawing samples from design space

    """

    def __init__(self, design_space: np.ndarray | List) -> None:
        """
        Initialization of sampler class

        Parameters
        ----------
        design_space: np.ndarray
            design space
        """

        # make sure the design space is a 2d array
        design_space = np.atleast_2d(np.asarray(design_space))
        #
        self.design_space = design_space
        # number of dimensions
        self.num_dim = len(design_space)

        # TODO: This is a hack to make the code work. The samples attribute
        self.samples: np.ndarray = None  # type: ignore

    def get_samples(self,
                    num_samples: int,
                    seed: int = 123456,
                    **kwargs) -> Any:
        """
        Get the samples

        Parameters
        ----------
        num_samples: int
            number of samples
        kwargs: int,int
            num_lf_samples: int
            num_hf_samples: int

        Returns
        ---------
        samples: any
            samples

        Notes
        ---------
        The function should be completed at the sub-sclass


        """

        raise NotImplementedError("Subclasses should implement this method.")

    def save_data(self, file_name: str = "data") -> None:
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
            pickle.dump(self.samples, file)

    def scale_samples(self, samples) -> np.ndarray:
        """
        Scale the samples to the design space

        Parameters
        ----------
        samples: np.ndarray
            samples

        Returns
        -------
        scaled_samples: np.ndarray
            scaled samples

        """

        scaled_samples = self.lb + samples * (self.ub - self.lb)

        return scaled_samples

    @property
    def lb(self) -> np.ndarray[Any, Any]:
        """return the lower bound of the design space

        Returns
        -------
        np.ndarray[Any, Any]
            lower bound of the design space
        """
        return self.design_space[:, 0]

    @property
    def ub(self) -> np.ndarray[Any, Any]:
        return self.design_space[:, 1]


class FixNumberSampler(SingleFidelitySampler):
    """Fix number of samples from design space

    Parameters
    ----------
    SingleFidelitySampler : class
        base class for sampling
    """

    def __init__(self, design_space: np.ndarray) -> None:
        """

        Parameters
        ----------
        design_space: np.ndarray
            design space

        """
        super(FixNumberSampler, self).__init__(design_space=design_space)

    def get_samples(self,
                    num_samples: int,
                    seed=123456,
                    **kwargs) -> np.ndarray:
        """

        Parameters
        ----------
        num_samples: int
            number of samples
        seed: int
            seed for reproducibility
        kwargs: additional info

        Returns
        -------
        samples: np.ndarray
            samples

        """
        # transfer the design space into one dimension
        space = self.design_space.flatten()

        # repeat the design space for num_samples times
        self.samples = np.tile(space, (num_samples, 1))

        return self.samples


class LatinHyperCube(SingleFidelitySampler):
    """
    Latin Hyper cube sampling via scipy
    """

    def __init__(self, design_space: np.ndarray) -> None:
        """

        Parameters
        ----------
        design_space: dict
            design space
        """
        super(LatinHyperCube, self).__init__(
            design_space=design_space,
        )

    def get_samples(self,
                    num_samples: int,
                    seed=123456,
                    **kwargs) -> np.ndarray:
        """get samples

        Parameters
        ----------
        num_samples : int
            number of samples

        Returns
        -------
        sample : np.ndarray
            a numpy array of samples

        """
        # record the seed
        self.seed = seed
        lhs_sampler = LatinHypercube(d=self.num_dim, seed=self.seed)
        samples = lhs_sampler.random(num_samples)

        # scale the samples
        self.samples = self.scale_samples(samples=samples)

        return self.samples


class RandomSampler(SingleFidelitySampler):
    """
    Random sampling
    """

    def __init__(self, design_space: np.ndarray) -> None:
        """

        Parameters
        ----------
        design_space: dict
            design space
        seed: int
            seed
        """
        super(RandomSampler, self).__init__(
            design_space=design_space,
        )

    def get_samples(self, num_samples: int,
                    seed=123456,
                    **kwargs) -> np.ndarray:
        """get samples

        Parameters
        ----------
        num_samples : int
            number of samples

        seed: int
            seed for reproducibility


        Returns
        -------
        sample : np.ndarray
            a numpy array of samples


        """
        # record the seed
        self.seed = seed
        # fix the seed for reproducibility
        np.random.seed(self.seed)
        samples = np.random.random((num_samples, self.num_dim))

        # scale the samples
        self.samples = self.scale_samples(samples=samples)

        return self.samples


class SobolSequence(SingleFidelitySampler):
    """
    Sobol Sequence sampling
    """

    def __init__(
        self, design_space: np.ndarray, num_skip: int = None
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
        super(SobolSequence, self).__init__(design_space=design_space)
        if num_skip is None:
            self.num_skip = len(design_space)
        else:
            self.num_skip = num_skip

    def get_samples(self,
                    num_samples: int,
                    seed: int = 123456, **kwargs) -> np.ndarray:
        """get samples

        Parameters
        ----------
        num_samples : int
            number of samples

        seed: int
            seed for reproducibility

        Returns
        -------
        sample : np.ndarray
            a numpy array of samples


        """
        # record the seed
        self.seed = seed

        # define the sobol sampler
        sobol_sampler = Sobol(d=self.num_dim, seed=self.seed)
        # generate lots of samples first
        sobol_sampler.random_base2(m=num_samples)
        _ = sobol_sampler.reset()
        _ = sobol_sampler.fast_forward(n=self.num_skip)
        # get the samples (an np.ndarray)
        samples = sobol_sampler.random(n=num_samples)

        # scale the samples
        self.samples = self.scale_samples(samples=samples)

        return self.samples
