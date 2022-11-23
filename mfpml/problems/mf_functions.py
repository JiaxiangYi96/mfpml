import matplotlib.pyplot as plt
import numpy as np

from mfpml.base.functions import Functions


class MultiFidelityFunctions(Functions):
    def plot_function(self, save_figure: bool = True, with_low_fidelity: bool = False) -> None:
        num_dim = self._get_dimension
        num_plot = 200
        if num_dim == 1:
            # draw the samples from design space
            x_plot = np.linspace(start=self._input_domian[0, 0], stop=self._input_domain[0, 1], num=num_plot)

            with plt.style.context(["ieee", "science"]):
                fig, ax = plt.subplots()
                ax.plot(x_plot, self.f(x=x_plot), label=f"{self.__class__.__name__}")
                if with_low_fidelity is True:
                    ax.plot(
                        x_plot,
                        self.lf(x=x_plot),
                        "--",
                        label=f"{self.__class__.__name__}_low_fidelity",
                    )
                ax.legend()
                ax.set(xlabel=r"$x$")
                ax.set(ylabel=r"$y$")
                plt.xlim(left=self._input_domain[0, 0], right=self._input_domain[0, 1])
                # ax.autoscale(tight=True)
                if save_figure is True:
                    fig.savefig(self.__class__.__name__, dpi=300)
                plt.show()
        elif num_dim == 2:

            x1_plot = np.linspace(start=self._input_domain[0, 0], stop=self._input_domain[0, 1], num=num_plot)
            x2_plot = np.linspace(start=self._input_domain[1, 0], stop=self._input_domain[1, 1], num=num_plot)
            X1, X2 = np.meshgrid(x1_plot, x2_plot)
            Y = np.zeros([len(X1), len(X2)])
            # get the values of Y at each mesh grid
            for i in range(len(X1)):
                for j in range(len(X1)):
                    xy = np.array([X1[i, j], X2[i, j]])
                    xy = np.reshape(xy, (1, 2))
                    Y[i, j] = self.f(x=xy)
            with plt.style.context(["ieee", "science"]):
                fig, ax = plt.subplots()
                cs = ax.contour(X1, X2, Y, 15)
                plt.colorbar(cs)
                ax.set(xlabel=r"$x_1$")
                ax.set(ylabel=r"$x_2$")
                # plt.clabel(cs, inline=True, fontsize=10)
                if save_figure is True:
                    fig.savefig(self.__class__.__name__, dpi=300)
                plt.show()
        else:
            raise ValueError("Unexpected value of 'num_dimension'!", num_dim)


class Forrester_1a(MultiFidelityFunctions):
    """
    Forrester function
    This file contains the definition of an adapted version of the simple 1D
    example function as presented in:
        Forrester Alexander I.J, Sóbester András and Keane Andy J "Multi-fidelity
        Optimization via Surrogate Modelling", Proceedings of the Royal Society A,
        vol. 463, http://doi.org/10.1098/rspa.2007.1900

    Function definitions:
    .. math::
        f_h(x) = (6x-2)^2 \sin(12x-4)
    For the low fidelity functions, they are different among different publications, in
    this code, the version adopted from :
        Jiang, P., Cheng, J., Zhou, Q., Shu, L., & Hu, J. (2019). Variable-fidelity lower
        confidence bounding approach for engineering optimization problems with expensive
        simulations. AIAA Journal, 57(12), 5416-5430.
    """

    num_dim: int = 1
    num_obj: int = 1
    num_cons: int = 0
    input_domain: np.ndarray = np.array([[0.0, 1.0]])
    design_space: dict = {"x": [0.0, 1.0]}
    optimum: float = -6.020740
    optimum_scheme: list = [0.757248757841856]
    low_fidelity: bool = True
    cost_ratio: list = None

    @classmethod
    def is_dim_compatible(cls, num_dim):
        assert num_dim == cls.num_dim, f"Can not change dimension for {cls.__name__} function"

        return num_dim

    def __init__(self, num_dim: int = 1, cost: list = [1.0, 0.2]) -> None:
        # check the dimension
        self.is_dim_compatible(num_dim=num_dim)
        self.cost = cost
        self.cr = round(cost[0] / cost[1])

    @staticmethod
    def hf(x: np.ndarray) -> np.ndarray:
        obj = (6 * x - 2) ** 2 * np.sin(12 * x - 4)
        obj = np.reshape(obj, (x.shape[0], 1))

        return obj

    @staticmethod
    def lf(x: np.ndarray, factor: float = None) -> np.ndarray:
        obj = (6 * x - 2) ** 2 * np.sin(12 * x - 4) - 5
        obj = np.reshape(obj, (x.shape[0], 1))
        return obj


class Forrester_1b(MultiFidelityFunctions):
    """
    Forrester function
    This file contains the definition of an adapted version of the simple 1D
    example function as presented in:
        Forrester Alexander I.J, Sóbester András and Keane Andy J "Multi-fidelity
        Optimization via Surrogate Modelling", Proceedings of the Royal Society A,
        vol. 463, http://doi.org/10.1098/rspa.2007.1900

    Function definitions:
    .. math::
        f_h(x) = (6x-2)^2 \sin(12x-4)
    For the low fidelity functions, they are different among different publications, in
    this code, the version adopted from :
        Jiang, P., Cheng, J., Zhou, Q., Shu, L., & Hu, J. (2019). Variable-fidelity lower
        confidence bounding approach for engineering optimization problems with expensive
        simulations. AIAA Journal, 57(12), 5416-5430.
    """

    num_dim: int = 1
    num_obj: int = 1
    num_cons: int = 0
    input_domain: np.ndarray = np.array([[0.0, 1.0]])
    design_space: dict = {"x": [0.0, 1.0]}
    optimum: float = -6.020740
    optimum_scheme: list = [0.757248757841856]
    low_fidelity: bool = True
    cost_ratio: list = None

    @classmethod
    def is_dim_compatible(cls, num_dim):
        assert num_dim == cls.num_dim, f"Can not change dimension for {cls.__name__} function"

        return num_dim

    def __init__(self, num_dim: int = 1, cost: list = [1.0, 0.2]) -> None:
        # check the dimension
        self.is_dim_compatible(num_dim=num_dim)
        self.cost = cost
        self.cr = round(cost[0] / cost[1])

    @staticmethod
    def hf(x: np.ndarray) -> np.ndarray:
        obj = (6 * x - 2) ** 2 * np.sin(12 * x - 4)
        obj = np.reshape(obj, (x.shape[0], 1))

        return obj

    @staticmethod
    def lf(x: np.ndarray, factor: float = None) -> np.ndarray:
        obj = 0.5 * ((6 * x - 2) ** 2 * np.sin(12 * x - 4)) + 10 * (x - 0.5) - 5
        obj = np.reshape(obj, (x.shape[0], 1))
        return obj


class Forrester_1a(MultiFidelityFunctions):
    """
    Forrester function
    This file contains the definition of an adapted version of the simple 1D
    example function as presented in:
        Forrester Alexander I.J, Sóbester András and Keane Andy J "Multi-fidelity
        Optimization via Surrogate Modelling", Proceedings of the Royal Society A,
        vol. 463, http://doi.org/10.1098/rspa.2007.1900

    Function definitions:
    .. math::
        f_h(x) = (6x-2)^2 \sin(12x-4)
    For the low fidelity functions, they are different among different publications, in
    this code, the version adopted from :
        Jiang, P., Cheng, J., Zhou, Q., Shu, L., & Hu, J. (2019). Variable-fidelity lower
        confidence bounding approach for engineering optimization problems with expensive
        simulations. AIAA Journal, 57(12), 5416-5430.
    """

    num_dim: int = 1
    num_obj: int = 1
    num_cons: int = 0
    input_domain: np.ndarray = np.array([[0.0, 1.0]])
    design_space: dict = {"x": [0.0, 1.0]}
    optimum: float = -6.020740
    optimum_scheme: list = [0.757248757841856]
    low_fidelity: bool = True
    cost_ratio: list = None

    @classmethod
    def is_dim_compatible(cls, num_dim):
        assert num_dim == cls.num_dim, f"Can not change dimension for {cls.__name__} function"

        return num_dim

    def __init__(self, num_dim: int = 1, cost: list = [1.0, 0.2]) -> None:
        # check the dimension
        self.is_dim_compatible(num_dim=num_dim)
        self.cost = cost
        self.cr = round(cost[0] / cost[1])

    @staticmethod
    def hf(x: np.ndarray) -> np.ndarray:
        obj = (6 * x - 2) ** 2 * np.sin(12 * x - 4)
        obj = np.reshape(obj, (x.shape[0], 1))

        return obj

    @staticmethod
    def lf(x: np.ndarray, factor: float = None) -> np.ndarray:
        obj = (6 * x - 2) ** 2 * np.sin(12 * x - 4) - 5
        obj = np.reshape(obj, (x.shape[0], 1))
        return obj


class Forrester_1c(MultiFidelityFunctions):
    """
    Forrester function
    This file contains the definition of an adapted version of the simple 1D
    example function as presented in:
        Forrester Alexander I.J, Sóbester András and Keane Andy J "Multi-fidelity
        Optimization via Surrogate Modelling", Proceedings of the Royal Society A,
        vol. 463, http://doi.org/10.1098/rspa.2007.1900

    Function definitions:
    .. math::
        f_h(x) = (6x-2)^2 \sin(12x-4)
    For the low fidelity functions, they are different among different publications, in
    this code, the version adopted from :
        Jiang, P., Cheng, J., Zhou, Q., Shu, L., & Hu, J. (2019). Variable-fidelity lower
        confidence bounding approach for engineering optimization problems with expensive
        simulations. AIAA Journal, 57(12), 5416-5430.
    """

    num_dim: int = 1
    num_obj: int = 1
    num_cons: int = 0
    input_domain: np.ndarray = np.array([0.0, 1.0])
    design_space: dict = {"x": [0.0, 1.0]}
    optimum: float = -6.020740
    optimum_scheme: list = [0.757248757841856]
    low_fidelity: bool = True
    cost_ratio: list = None

    @classmethod
    def is_dim_compatible(cls, num_dim):
        assert num_dim == cls.num_dim, f"Can not change dimension for {cls.__name__} function"

        return num_dim

    def __init__(self, num_dim: int = 1, cost: list = [1.0, 0.2]) -> None:
        # check the dimension
        self.is_dim_compatible(num_dim=num_dim)
        self.cost = cost
        self.cr = round(cost[0] / cost[1])

    @staticmethod
    def hf(x: np.ndarray) -> np.ndarray:
        obj = (6 * x - 2) ** 2 * np.sin(12 * x - 4)
        obj = np.reshape(obj, (x.shape[0], 1))

        return obj

    @staticmethod
    def lf(x: np.ndarray, factor: float = None) -> np.ndarray:
        obj = (6 * (x + 0.2) - 2) ** 2 * np.sin(12 * (x + 0.2) - 4)
        obj = np.reshape(obj, (x.shape[0], 1))
        return obj
