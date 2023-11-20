import matplotlib.pyplot as plt
import numpy as np

from mfpml.problems.functions import Functions


# local module
class MultiFidelityFunctions(Functions):

    def plot_function(
        self, save_figure: bool = True, with_low_fidelity: bool = False
    ) -> None:
        num_dim = self._get_dimension
        num_plot = 200
        if num_dim == 1:
            # draw the samples from design space
            x_plot = np.linspace(
                start=self._input_domain[0, 0],
                stop=self._input_domain[0, 1],
                num=num_plot,
            )

            fig, ax = plt.subplots()
            ax.plot(
                x_plot, self.f(x=x_plot), label=f"{self.__class__.__name__}"
            )
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
            plt.xlim(
                left=self._input_domain[0, 0], right=self._input_domain[0, 1]
            )
            # ax.autoscale(tight=True)
            if save_figure is True:
                fig.savefig(self.__class__.__name__, dpi=300)
            plt.show()
        elif num_dim == 2:
            x1_plot = np.linspace(
                start=self._input_domain[0, 0],
                stop=self._input_domain[0, 1],
                num=num_plot,
            )
            x2_plot = np.linspace(
                start=self._input_domain[1, 0],
                stop=self._input_domain[1, 1],
                num=num_plot,
            )
            X1, X2 = np.meshgrid(x1_plot, x2_plot)
            Y = np.zeros([len(X1), len(X2)])
            # get the values of Y at each mesh grid
            for i in range(len(X1)):
                for j in range(len(X1)):
                    xy = np.array([X1[i, j], X2[i, j]])
                    xy = np.reshape(xy, (1, 2))
                    Y[i, j] = self.f(x=xy)
            # with plt.style.context(["ieee", "science"]):
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

    @classmethod
    def is_dim_compatible(cls, num_dim):
        assert (
            num_dim == cls.num_dim
        ), f"Can not change dimension for {cls.__name__} function"

        return num_dim

    def __call__(self, x: dict) -> dict:
        out = {}
        out["hf"] = None
        out["lf"] = None
        if x["hf"] is not None:
            out["hf"] = self.hf(x["hf"])
        if x["lf"] is not None:
            out["lf"] = self.lf(x["lf"])
        return out


class Forrester_1a(MultiFidelityFunctions):

    num_dim: int = 1
    num_obj: int = 1
    num_cons: int = 0
    input_domain: np.ndarray = np.array([[0.0, 1.0]])
    design_space: dict = {"x": [0.0, 1.0]}
    optimum: float = -6.020740
    optimum_scheme: list = [0.757248757841856]
    low_fidelity: bool = True
    cost_ratio: list = None

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
    def lf(x: np.ndarray, factor: float = None) -> np.ndarray:  # type: ignore
        obj = (6 * x - 2) ** 2 * np.sin(12 * x - 4) - 5
        obj = np.reshape(obj, (x.shape[0], 1))
        return obj


class Forrester_1b(MultiFidelityFunctions):
    """
    Forrester function from Jicheng

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
        obj = (
            0.5 * ((6 * x - 2) ** 2 * np.sin(12 * x - 4)) + 10 * (x - 0.5) - 5
        )
        obj = np.reshape(obj, (x.shape[0], 1))
        return obj


class Forrester_1c(MultiFidelityFunctions):
    """
    Forrester function
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


class mf_Hartman3(MultiFidelityFunctions):
    """multi fidelity Hartman3 function

    Parameters
    ----------
    MultiFidelityFunctions : parent class
        multi-fidelity function

    Returns
    -------
    _type_
        _description_
    """

    num_dim: int = 3
    num_obj: int = 1
    num_cons: int = 0
    input_domain: np.ndarray = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    design_space: dict = {"x1": [0.0, 1.0], "x2": [0.0, 1.0], "x3": [0.0, 1.0]}
    optimum: float = -3.86278214782076
    optimum_scheme: list = [0.1, 0.55592003, 0.85218259]
    low_fidelity: list = ["low"]

    def __init__(self, num_dim: int = 3, cost: list = [1.0, 0.2]) -> None:
        # check the dimension
        self.is_dim_compatible(num_dim=num_dim)
        self.cost = cost
        self.cr = round(cost[0] / cost[1])

    @staticmethod
    def hf(x: np.ndarray) -> np.ndarray:
        a = np.array(
            [
                [3.0, 10.0, 30.0],
                [0.1, 10.0, 35.0],
                [3.0, 10.0, 30.0],
                [0.1, 10.0, 35.0],
            ]
        )
        p = np.array(
            [
                [0.3689, 0.117, 0.2673],
                [0.4699, 0.4387, 0.747],
                [0.1091, 0.8732, 0.5547],
                [0.03815, 0.5743, 0.8828],
            ]
        )
        c = np.array([1, 1.2, 3, 3.2])
        num_samples = x.shape[0]
        obj = np.zeros((num_samples, 1))
        for i in range(num_samples):
            obj[i, :] = -np.dot(
                c, np.exp(-np.sum(a * (x[i, :] - p) ** 2, axis=1))
            )
        obj.reshape((x.shape[0], 1))
        return obj

    @staticmethod
    def lf(x: np.ndarray, factor: float = None) -> np.ndarray:
        obj = (
            0.585
            - 0.324 * x[:, 0]
            - 0.379 * x[:, 1]
            - 0.431 * x[:, 2]
            - 0.208 * x[:, 0] * x[:, 1]
            + 0.326 * x[:, 0] * x[:, 2]
            + 0.193 * x[:, 1] * x[:, 2]
            + 0.225 * x[:, 0] ** 2
            + 0.263 * x[:, 1] ** 2
            + 0.274 * x[:, 2] ** 2
        )
        return obj.reshape(-1, 1)


class mf_Sixhump(MultiFidelityFunctions):

    num_dim: int = 2
    num_obj: int = 1
    num_cons: int = 0
    input_domain = np.array([[-2.0, 2.0], [-2.0, 2.0]])
    design_space: dict = {"x1": [-2.0, 2.0], "x2": [-2.0, 2.0]}
    optimum: float = -1.0316
    optimum_scheme: list = [[0.0898, -0.7126], [-0.0898, 0.7126]]
    low_fidelity: list = None

    def __init__(self, num_dim: int = 2, cost: list = [1.0, 0.2]) -> None:
        """
        Initialization
        """
        self.cr = round(cost[0] / cost[1])
        self.is_dim_compatible(num_dim=num_dim)

    @staticmethod
    def hf(x: np.ndarray) -> np.ndarray:
        x1 = x[:, 0]
        x2 = x[:, 1]

        term1 = (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2
        term2 = x1 * x2
        term3 = (-4 + 4 * x2**2) * x2**2

        obj = term1 + term2 + term3
        obj = np.reshape(obj, (x.shape[0], 1))

        return obj

    @staticmethod
    def lf(x: np.ndarray) -> np.ndarray:
        x1 = x[:, 0]
        x2 = x[:, 1]
        term1 = (4 - 2.1 * (0.7 * x1) ** 2 + ((0.7 * x1) ** 4) / 3) * (
            0.7 * x1
        ) ** 2
        term2 = 0.7 * x1 * 0.7 * x2
        term3 = (-4 + 4 * (0.7 * x2) ** 2) * (0.7 * x2) ** 2
        obj1 = (term1 + term2 + term3).reshape((-1, 1))
        obj2 = (-x1 * x2 - 15).reshape(x.shape[0], 1)

        obj = obj1 + obj2

        return obj


class mf_Hartman6(MultiFidelityFunctions):

    num_dim: int = 6
    num_obj: int = 1
    num_cons: int = 0
    input_domain = np.array(
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    design_space: dict = {
        "x1": [0.0, 1.0],
        "x2": [0.0, 1.0],
        "x3": [0.0, 1.0],
        "x4": [0.0, 1.0],
        "x5": [0.0, 1.0],
        "x6": [0.0, 1.0],
    }
    optimum: float = -np.log(3.32236801141551)
    optimum_scheme: list = [
        0.20168952,
        0.15001069,
        0.47687398,
        0.27533243,
        0.31165162,
        0.65730054,
    ]
    low_fidelity: list = None

    def __init__(self, num_dim: int = 6, cost: list = [1.0, 0.1]) -> None:
        """
        Initialization
        """
        self.cr = round(cost[0] / cost[1])
        self.is_dim_compatible(num_dim=num_dim)

    @staticmethod
    def hf(x: np.ndarray) -> np.ndarray:
        """hf
        """
        a = np.array(
            [
                [10.00, 3.0, 17.00, 3.5, 1.7, 8],
                [0.05, 10.0, 17.00, 0.1, 8.0, 14],
                [3.00, 3.5, 1.70, 10.0, 17.0, 8],
                [17.00, 8.0, 0.05, 10.0, 0.1, 14],
            ]
        )
        p = np.array(
            [
                [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
                [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
            ]
        )
        c = np.array([1.0, 1.2, 3.0, 3.2])
        num_samples = x.shape[0]
        obj = np.zeros((num_samples, 1))
        for i in range(num_samples):
            obj[i, :] = -np.dot(
                c, np.exp(-np.sum(a * (x[i, :] - p) ** 2, axis=1))
            )
        obj = np.reshape(obj, (x.shape[0], 1))
        return -np.log(-obj)

    @staticmethod
    def lf(x: np.ndarray) -> np.ndarray:

        # different from lf
        a = np.array(
            [
                [10.00, 3.0, 17.00, 3.5, 1.7, 8],
                [0.05, 10.0, 17.00, 0.1, 8.0, 14],
                [3.00, 3.5, 1.70, 10.0, 17.0, 8],
                [17.00, 8.0, 0.05, 10.0, 0.1, 14],
            ]
        )
        p = np.array(
            [
                [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
                [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
            ]
        )
        c = np.array([1.1, 0.8, 2.5, 3])
        l_array = np.array([0.75, 1, 0.8, 1.3, 0.7, 1.1])
        # number of samples
        num_samples = x.shape[0]
        obj = np.zeros((num_samples, 1))
        for i in range(num_samples):
            obj[i, :] = -np.dot(
                c,
                np.exp(-np.sum(a * (l_array * x[i, :] - p) ** 2, axis=1)),
            )
        obj = np.reshape(obj, (x.shape[0], 1))
        return -np.log(-obj)


class mf_Discontinuous(MultiFidelityFunctions):
    """multi-fidelity discontinuous function,

    Parameters
    ----------
    MultiFidelityFunctions : class
        multifidelity function class

    """
    num_dim: int = 1
    num_obj: int = 1
    num_cons: int = 0
    input_domain = np.array([[0.0, 1.0]])
    design_space: dict = {"x": [0.0, 1.0]}
    optimum: float = None
    optimum_scheme: list = None
    low_fidelity: list = None

    def __init__(self, num_dim: int = 1) -> None:
        super().__init__()
        # check dimension
        self.is_dim_compatible(num_dim=num_dim)

    @staticmethod
    def lf(x: np.ndarray) -> np.ndarray:
        # num of samples
        y = np.zeros((x.shape[0], 1))
        # get objective for every point
        for ii in range(x.shape[0]):
            if x[ii, 0] <= 0.5:
                y[ii, 0] = 0.5 * (6 * x[ii, 0] - 2) ** 2 * \
                    np.sin(12 * x[ii, 0] - 4) + 10 * (x[ii, 0] - 0.5) - 5
            else:
                y[ii, 0] = 3 + 0.5 * (6 * x[ii, 0] - 2) ** 2 * \
                    np.sin(12 * x[ii, 0] - 4) + 10 * (x[ii, 0] - 0.5) - 5

        return y.reshape((-1, 1))

    @staticmethod
    def hf(x: np.ndarray) -> np.ndarray:

        y = np.zeros((x.shape[0], 1))

        for ii in range(x.shape[0]):
            if x[ii, 0] <= 0.5:
                y[ii, 0] = 2*(0.5 * (6 * x[ii, 0] - 2) ** 2 *
                              np.sin(12 * x[ii, 0] - 4) +
                              10 * (x[ii, 0] - 0.5) - 5) - 20 * x[ii, 0] + 20
            else:
                y[ii, 0] = 4 + 2*(3 + 0.5 * (6 * x[ii, 0] - 2) ** 2 * np.sin(
                    12 * x[ii, 0] - 4) + 10 * (x[ii, 0] - 0.5) - 5) - \
                    20 * x[ii, 0] + 20

        return y.reshape((-1, 1))


class ContinuousNonlinearCorrelation1D(MultiFidelityFunctions):
    """Continuous Nonlinear Correlation 1D

    Parameters
    ----------
    MultiFidelityFunctions : class
        base class

    """
    num_dim: int = 1
    num_obj: int = 1
    num_cons: int = 0
    input_domain = np.array([[0.0, 1.0]])
    design_space: dict = {"x": [0.0, 1.0]}
    optimum: float = None
    optimum_scheme: list = None
    low_fidelity: list = None

    def __init__(self, num_dim: int = 1) -> None:
        super().__init__()
        # check dimension
        self.is_dim_compatible(num_dim=num_dim)

    @staticmethod
    def lf(x: np.ndarray) -> np.ndarray:
        y = np.sin(8*np.pi*x)

        return y.reshape((-1, 1))

    @staticmethod
    def hf(x: np.ndarray) -> np.ndarray:

        y = np.sin(8*np.pi*x) ** 2 * (x-np.sqrt(2))

        return y.reshape((-1, 1))


class PhaseShiftedOscillations(MultiFidelityFunctions):
    """Phase shifted oscillations

    Parameters
    ----------
    MultiFidelityFunctions : class
        based class

    """
    num_dim: int = 1
    num_obj: int = 1
    num_cons: int = 0
    input_domain = np.array([[0.0, 1.0]])
    design_space: dict = {"x": [0.0, 1.0]}
    optimum: float = None
    optimum_scheme: list = None
    low_fidelity: list = None

    def __init__(self, num_dim: int = 1) -> None:
        super().__init__()
        # check dimension

        self.is_dim_compatible(num_dim=num_dim)

    @staticmethod
    def lf(x: np.ndarray) -> np.ndarray:
        y = np.sin(8*np.pi * x)
        return y.reshape((-1, 1))

    @staticmethod
    def hf(x: np.ndarray) -> np.ndarray:

        y = x**2 + (np.sin(8*np.pi*x+np.pi/10))**2

        return y.reshape((-1, 1))


class mf_Bohachevsky(MultiFidelityFunctions):
    """multi-fidelity Bohachevsky function

    Parameters
    ----------
    MultiFidelityFunctions : class
        base class

    """
    num_dim: int = 2
    num_obj: int = 1
    num_cons: int = 0
    input_domain = np.array(
        [
            [-5.0, 5.0],
            [-5.0, 5.0],
        ]
    )
    design_space: dict = {
        "x1": [-5.0, 5.0],
        "x2": [-5.0, 5.0],
    }
    optimum: float = 0.0
    optimum_scheme: list = [0.0, 0.0]
    low_fidelity: bool = True

    def __init__(self, num_dim: int = 2) -> None:
        super().__init__()

        # check dimension
        self.is_dim_compatible(num_dim=num_dim)

    @staticmethod
    def lf(x: np.ndarray) -> np.ndarray:

        x = np.atleast_2d(x)

        # get the values of x1 and x2
        x11 = 0.7*x[:, 0]
        x1 = x[:, 0]
        x2 = x[:, 1]

        # calculate the terms (term 1-3 is same as hf by replacing x1 with x11)
        # the fourth term is different from hf, which is x1*x2 - 12
        term1 = x11**2 + 2*x2**2
        term2 = 0.3*np.cos(3*np.pi*x11)
        term3 = 0.4*np.cos(4*np.pi*x2)
        term4 = x1*x2 - 12

        # calculate the objective function
        obj = (term1 - term2 - term3 + 0.7 + term4).reshape((-1, 1))

        return obj.reshape((-1, 1))

    @staticmethod
    def hf(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)

        # get the values of x1 and x2
        x1 = x[:, 0]
        x2 = x[:, 1]

        # calculate the terms
        term1 = x1**2 + 2*x2**2
        term2 = 0.3*np.cos(3*np.pi*x1)
        term3 = 0.4*np.cos(4*np.pi*x2)

        # calculate the objective function
        obj = (term1 - term2 - term3 + 0.7).reshape((-1, 1))
        return obj


class mf_Booth(MultiFidelityFunctions):
    """multi-fidelity Booth function,

    Parameters
    ----------
    MultiFidelityFunctions : class
        base class

    """

    num_dim: int = 2
    num_obj: int = 1
    num_cons: int = 0
    input_domain = np.array(
        [
            [-10.0, 10.0],
            [-10.0, 10.0],
        ]
    )
    design_space: dict = {
        "x1": [-10.0, 10.0],
        "x2": [-10.0, 10.0],
    }
    optimum: float = 0.0
    optimum_scheme: list = [1.0, 3.0]
    low_fidelity: bool = True

    def __init__(self, num_dim: int = 2) -> None:
        super().__init__()

        # check dimension
        self.is_dim_compatible(num_dim=num_dim)

    @staticmethod
    def lf(x: np.ndarray) -> np.ndarray:

        x = np.atleast_2d(x)

        # get the values of x1 and x2
        x11 = 0.4*x[:, 0]
        x1 = x[:, 0]
        x2 = x[:, 1]

        # calculate the terms (term 1-3 is same as hf by replacing x1 with x11)
        # the fourth term is different from hf, which is x1*x2 - 12
        term1 = (x11 + 2*x2 - 7)**2
        term2 = (2*x11 + x2 - 5)**2
        term3 = 1.7*x1*x2 - x1 + 2*x2

        # calculate the objective function
        obj = (term1 + term2 + term3).reshape((-1, 1))

        return obj.reshape((-1, 1))

    @staticmethod
    def hf(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)

        # get the values of x1 and x2
        x1 = x[:, 0]
        x2 = x[:, 1]

        # calculate the terms
        term1 = (x1 + 2*x2 - 7)**2
        term2 = (2*x1 + x2 - 5)**2

        # calculate the objective function
        obj = (term1 + term2).reshape((-1, 1))
        return obj


class mf_Borehole(MultiFidelityFunctions):
    """multi-fidelity Borehole function,

    Parameters
    ----------
    MultiFidelityFunctions : class
        base class

    """

    num_dim: int = 8
    num_obj: int = 1
    num_cons: int = 0
    input_domain = np.array(
        [
            [0.05, 0.15],
            [100, 50000],
            [63070, 115600],
            [990, 1110],
            [63.1, 116],
            [700, 820],
            [1120, 1680],
            [9855, 12045],
        ]
    )
    design_space: dict = {
        "rw": [0.05, 0.15],
        "r": [100, 50000],
        "Tu": [63070, 115600],
        "Hu": [990, 1110],
        "Tl": [63.1, 116],
        "Hl": [700, 820],
        "L": [1120, 1680],
        "Kw": [9855, 12045],
    }
    optimum: float = None
    optimum_scheme: list = None
    low_fidelity: bool = True

    def __init__(self, num_dim: int = 8) -> None:
        super().__init__()

        # check dimension
        self.is_dim_compatible(num_dim=num_dim)

    @staticmethod
    def base(x: np.ndarray, a: float, b: float) -> np.ndarray:

        # get design variables
        x = np.atleast_2d(x)

        # get the values
        rw = x[:, 0]
        r = x[:, 1]
        Tu = x[:, 2]
        Hu = x[:, 3]
        Tl = x[:, 4]
        Hl = x[:, 5]
        L = x[:, 6]
        Kw = x[:, 7]

        # calculate the terms
        term1 = a*Tu*(Hu - Hl)
        term2 = np.log(r/rw)
        term3 = b + 2*L*Tu/(term2 * rw**2 * Kw) + Tu/Tl

        # calculate the objective function
        obj = (term1 / (term2 * term3)).reshape((-1, 1))

        return obj.reshape((-1, 1))

    @staticmethod
    def lf(x: np.ndarray) -> np.ndarray:

        obj = mf_Borehole().base(x=x, a=5, b=1.5)

        return obj.reshape((-1, 1))

    @staticmethod
    def hf(x: np.ndarray) -> np.ndarray:

        obj = mf_Borehole().base(x=x, a=2*np.pi, b=1)

        return obj


class mf_CurrinExp(MultiFidelityFunctions):
    """multi-fidelity CurrinExp function,

    Parameters
    ----------
    MultiFidelityFunctions : class
        base class

    """

    num_dim: int = 2
    num_obj: int = 1
    num_cons: int = 0
    input_domain = np.array(
        [
            [0, 1],
            [0, 1],
        ]
    )
    design_space: dict = {
        "x1": [0, 1],
        "x2": [0, 1],
    }
    optimum: float = None
    optimum_scheme: list = None
    low_fidelity: bool = True

    def __init__(self, num_dim: int = 2) -> None:
        super().__init__()

        # check dimension
        self.is_dim_compatible(num_dim=num_dim)

    @staticmethod
    def lf(x: np.ndarray) -> np.ndarray:

        # reshape the input
        x = np.atleast_2d(x)

        # calculate the terms
        xx = x.copy()
        xx = xx + 0.05
        term1 = mf_CurrinExp().hf(x=xx)

        # calculate term2
        xx = x.copy()
        xx[:, 0] = xx[:, 0] + 0.05
        xx[:, 1] = xx[:, 1] - 0.05
        term2 = mf_CurrinExp().hf(x=xx)

        # calculate term3
        xx = x.copy()
        xx[:, 0] = xx[:, 0] - 0.05
        xx[:, 1] = xx[:, 1] + 0.05
        term3 = mf_CurrinExp().hf(x=xx)

        # calculate term4
        xx = x.copy()
        xx = xx - 0.05
        term4 = mf_CurrinExp().hf(x=xx)

        # calculate the objective function
        obj = ((term1 + term2 + term3 + term4)/4.0).reshape((-1, 1))

        return obj.reshape((-1, 1))

    @staticmethod
    def hf(x: np.ndarray) -> np.ndarray:

        # reshape the input
        x = np.atleast_2d(x)

        # get the values of x1 and x2
        x1 = x[:, 0]
        x2 = x[:, 1]

        # calculate the terms
        are_zero = x2 <= 1e-8  # Assumes x2 approaches 0 from positive
        term1 = np.ones(x2.shape)

        term1[~are_zero] -= np.exp(-1 / (2*x2[~are_zero]))
        # term1 = np.exp(-1/2*x2)
        term2 = 2300*x1**3 + 1900*x1**2 + 2092*x1 + 60
        term3 = 100*x1**3 + 500*x1**2 + 4*x1 + 20

        # calculate the objective function
        obj = (term1*term2 / term3).reshape((-1, 1))

        return obj


class mf_Himmelblau(MultiFidelityFunctions):
    """multi-fidelity Himmelblau function,

    Parameters
    ----------
    MultiFidelityFunctions : class
        base class
    """

    num_dim: int = 2
    num_obj: int = 1
    num_cons: int = 0
    input_domain = np.array(
        [
            [-4.0, 4.0],
            [-4.0, 4.0],
        ]
    )
    design_space: dict = {
        "x1": [-4.0, 4.0],
        "x2": [-4.0, 4.0],
    }
    optimum: float = None
    optimum_scheme: list = None
    low_fidelity: bool = True

    def __init__(self, num_dim: int = 2) -> None:
        super().__init__()

        # check dimension
        self.is_dim_compatible(num_dim=num_dim)

    @staticmethod
    def lf(x: np.ndarray) -> np.ndarray:

        x = np.atleast_2d(x)

        # get the values of x1 and x2
        x11 = 0.5*x[:, 0]
        x22 = 0.8*x[:, 1]
        x1 = x[:, 0]
        x2 = x[:, 1]

        # calculate the terms (term 1-3 is same as hf by replacing x1 with x11)
        # the fourth term is different from hf, which is x1*x2 - 12
        term1 = (x11**2 + x22 - 11)**2
        term2 = (x11 + x22**2 - 7)**2
        term3 = x2**3 - (x1 + 1)**2

        # calculate the objective function
        obj = (term1 + term2 + term3).reshape((-1, 1))

        return obj.reshape((-1, 1))

    @staticmethod
    def hf(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)

        # get the values of x1 and x2
        x1 = x[:, 0]
        x2 = x[:, 1]

        # calculate the terms
        term1 = (x1**2 + x2 - 11)**2
        term2 = (x1 + x2**2 - 7)**2

        # calculate the objective function
        obj = (term1 + term2).reshape((-1, 1))
        return obj


class mf_Park91A(MultiFidelityFunctions):
    """multi-fidelity Park91A function,

    Parameters
    ----------
    MultiFidelityFunctions : class
        base class

    """

    num_dim: int = 4
    num_obj: int = 1
    num_cons: int = 0
    input_domain = np.array(
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    design_space: dict = {
        "x1": [0.0, 1.0],
        "x2": [0.0, 1.0],
        "x3": [0.0, 1.0],
        "x4": [0.0, 1.0],
    }
    optimum: float = None
    optimum_scheme: list = None
    low_fidelity: bool = True

    def __init__(self, num_dim: int = 4) -> None:
        super().__init__()

        # check dimension
        self.is_dim_compatible(num_dim=num_dim)

    @staticmethod
    def lf(x: np.ndarray) -> np.ndarray:

        x = np.atleast_2d(x)

        # get the values of x1, x2, x3, x4
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        # x4 = x[:, 3]

        # calculate the terms
        term1 = (1 + np.sin(x1) / 10).reshape((-1, 1))
        term1 = term1 * mf_Park91A().hf(x=x)

        term2 = -2*x1 + x2**2 + x3**2 + 0.5

        # calculate the objective function
        obj = (term1).reshape((-1, 1)) + term2.reshape((-1, 1))

        return obj.reshape((-1, 1))

    @staticmethod
    def hf(x: np.ndarray) -> np.ndarray:

        x = np.atleast_2d(x)

        # get the values of x1, x2, x3, x4
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]

        # calculate the terms
        term1 = x1/2 * (np.sqrt(1 + (x2 + x3**2) * x4/x1**2) - 1)
        term2 = (x1 + 3*x4)*np.exp(1 + np.sin(x3))

        # calculate the objective function
        obj = (term1 + term2).reshape((-1, 1))

        return obj


class mf_Park91B(MultiFidelityFunctions):
    """ multi-fidelity Park91B function,

    Parameters
    ----------
    MultiFidelityFunctions : class
        base class
    """

    num_dim: int = 4
    num_obj: int = 1
    num_cons: int = 0
    input_domain = np.array(
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    design_space: dict = {
        "x1": [0.0, 1.0],
        "x2": [0.0, 1.0],
        "x3": [0.0, 1.0],
        "x4": [0.0, 1.0],
    }
    optimum: float = None  # type: ignore
    optimum_scheme: list = None  # type : ignore
    low_fidelity: bool = True

    def __init__(self, num_dim: int = 4) -> None:
        super().__init__()

        # check dimension
        self.is_dim_compatible(num_dim=num_dim)

    @staticmethod
    def lf(x: np.ndarray) -> np.ndarray:

        x = np.atleast_2d(x)

        # calculate the terms
        term1 = 1.2 * mf_Park91B().hf(x=x) - 1

        # calculate the objective function
        obj = (term1).reshape((-1, 1))

        return obj

    @staticmethod
    def hf(x: np.ndarray) -> np.ndarray:

        x = np.atleast_2d(x)

        # get the values of x1, x2, x3, x4
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]

        # calculate the terms
        term1 = 2/3 * np.exp(x1 + x2) - x4*np.sin(x3) + x3

        # calculate the objective function
        obj = (term1).reshape((-1, 1))

        return obj
