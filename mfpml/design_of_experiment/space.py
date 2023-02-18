import numpy as np


class DesignSpace:

    def __init__(self, names: list, low_bound: list, high_bound: list) -> np.ndarray:
        """
        Parameters
        ----------
        names : list
            Names of added variables
        low_bound : list
            Lower bound of added design variables
        high_bound :  list
            Higher bound of added design variables
        Returns
        -------
        input_domain: list
            a list that contains design space information

        """
        self._input_domain = np.ndarray
        self._design_space = dict()

        self.__check_consistency(names=names,
                                 low_bound=low_bound,
                                 high_bound=high_bound)

        for ii, name in enumerate(names):
            self._design_space[name] = [low_bound[ii], high_bound[ii]]
        self._input_domain = np.array([low_bound, high_bound]).transpose()

    @staticmethod
    def __check_consistency(names: list, low_bound: list, high_bound: list) -> None:
        """
        check consistency
        Parameters
        ----------
        names : list
            names
        low_bound: list
            low bound
        high_bound: list
            high bound

        Returns
        -------

        """
        assert (len(names) == len(low_bound) == len(
            high_bound)), "Length of variables should be same"

    @staticmethod
    def __check_magnitude(low_bound: list, high_bound: list) -> None:
        """

        Parameters
        ----------
        low_bound: list
            low bound
        high_bound: list
            high bound

        Returns
        -------

        """
        assert (sum(np.array(high_bound) - np.array(
            low_bound)) == len(low_bound)), "Length of variables should be same"

    @property
    def design_space(self) -> dict:
        """

        Returns
        -------

        """
        return self._design_space

    @property
    def input_domain(self) -> np.ndarray:
        """

        Returns
        -------
        input domain: np.ndarray
            design space in numpy array form

        """
        return self._input_domain
