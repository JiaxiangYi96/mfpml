
from abc import ABC
from typing import Any

import numpy as np


class Functions(ABC):
    @staticmethod
    def f(x: np.ndarray) -> np.ndarray:
        """

        Parameters
        ----------
        x: np.ndarray
            design scheme that needs to be evaluated

        Returns
        -------
        y: np.ndarray
            responses from single fidelity functions

        Raises
        ------
        NotImplementedError
            Raised when not implemented.
        """

        raise NotImplementedError("Subclasses should implement this method.")

    @staticmethod
    def f_der(x: np.ndarray) -> np.ndarray:
        """

        Parameters
        ----------
        x: np.ndarray
            design scheme that needs to be evaluated

        Returns
        -------
        f_der: np.ndarray
            responses from the single fidelity functions

        Raises
        ------
        NotImplementedError
            Raised when not implemented. Subclasses should implement this
            method.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @staticmethod
    def hf(x: np.ndarray) -> np.ndarray:
        """

        Parameters
        ----------
        x: np.ndarray
            design scheme that needs to be evaluated

        Returns
        -------
        y_hf: np.ndarray
            responses from the high fidelity functions

        Raises
        ------
        NotImplementedError
            Raised when not implemented.
        """

        raise NotImplementedError("Subclasses should implement this method.")

    @staticmethod
    def lf(x: np.ndarray, factor: float) -> np.ndarray:
        """
        low fidelity function

        Parameters
        ----------
        x: np.ndarray
            design scheme that needs to be evaluated
        factor: floatS
            a factor to control generating low fidelity functions

        Returns
        -------
        y_lf: np.ndarray
            responses from the low functions

        Raises
        ------
        NotImplementedError
            Raised when not implemented.
        """

        raise NotImplementedError("Subclasses should implement this method.")

    @staticmethod
    def hf_der(x: np.ndarray) -> np.ndarray:
        """
        derivative of high fidelity function

        Parameters
        ----------
        x: np.ndarray
            design scheme that needs to be evaluated

        Returns
        -------
        hf_der: np.ndarray
            responses from the high fidelity functions

        Raises
        ------
        NotImplementedError
            Raised when not implemented.
        """

        raise NotImplementedError("Subclasses should implement this method.")

    @staticmethod
    def lf_der(x: np.ndarray) -> np.ndarray:
        """
        derivative of high fidelity function

        Parameters
        ----------
        x: np.ndarray
            design scheme that needs to be evaluated

        Returns
        -------
        lf_der: np.ndarray
            responses from the low functions

        Raises
        ------
        NotImplementedError
            Raised when not implemented.
        """

        raise NotImplementedError("Subclasses should implement this method.")

    @staticmethod
    def f_cons(x: np.ndarray) -> np.ndarray:
        """
        constrained functions of high-fidelity function

        Parameters
        ----------
        x: np.ndarray
            design scheme that needs to be evaluated

        Returns
        -------
        f_cons: np.ndarray
            constrained responses from the single fidelity functions

        Raises
        ------
        NotImplementedError
            Raised when not implemented.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @staticmethod
    def hf_cons(x: np.ndarray) -> np.ndarray:
        """
        constrained functions of high-fidelity function

        Parameters
        ----------
        x: np.ndarray
            design scheme that needs to be evaluated

        Returns
        -------
        hf_cons: np.ndarray
            constrained responses from the high fidelity functions

        Raises
        ------
        NotImplementedError
            Raised when not implemented.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @staticmethod
    def lf_cons(x: np.ndarray) -> np.ndarray:
        """
        constrained functions of low-fidelity function

        Parameters
        ----------
        x: np.ndarray
            design scheme that needs to be evaluated

        Returns
        -------
        lf_cons: np.ndarray
            constrained responses from the low-fidelity functions

        Raises
        ------
        NotImplementedError
            Raised when not implemented.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def plot_function(
        self, with_low_fidelity: bool = False, save_figure: bool = True
    ) -> None:
        """
        Function to visualize the landscape of the function s
        Parameters
        ----------
        with_low_fidelity: bool
            plot low fidelity functions or not
        save_figure : bool
            save figure or not

        Returns
        -------

        """
        raise NotImplementedError("Subclasses should implement this method.")

    @property
    def _get_dimension(self) -> int:
        """
        Get dimension of the function

        Returns
        -------
        dimension: int
            dimension of the problem
        """
        return self.__class__.num_dim

    @property
    def _get_low_fidelity(self) -> list:
        """
        Get names of low fidelity functions
        Returns
        -------
        name: list
            name list of low fidelity functions

        """

        return self.__class__.low_fidelity

    @property
    def _optimum(self) -> float:
        """

        Returns
        -------
        optimum: float
            name of the class

        """
        return self.__class__.optimum

    @property
    def _optimum_variable(self) -> list:
        """

        Returns
        -------
        optimum_variable: list
            Best design scheme

        """
        return self.__class__.optimum_scheme

    @property
    def _design_space(self) -> dict:
        """

        Returns
        -------
        design_space:

        """

        return self.__class__.design_space

    @property
    def _input_domain(self) -> np.ndarray:
        """

        Returns
        -------

        """
        return self.__class__.input_domain


class FunctionWrapper:
    """
    Object to wrap user's function, allowing pick lability
    """

    def __init__(self, function: Any, args: tuple = ()) -> None:
        """function wrapper

        Parameters
        ----------
        function : any
            function
        args : tuple, optional
            additional parameters , by default ()
        """
        self.function = function

        self.args = [] if args is None else args

    def __call__(self, x) -> Any:
        return self.function(x, *self.args)
