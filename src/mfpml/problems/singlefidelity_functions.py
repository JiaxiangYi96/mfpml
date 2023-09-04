

import matplotlib.pyplot as plt
import numpy as np

from mfpml.problems.functions import Functions


class SingleFidelityFunctions(Functions):
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
            # plot the function
            fig, ax = plt.subplots()
            ax.plot(
                x_plot,
                self.f(x=x_plot),
                label=f"{self.__class__.__name__}",
            )
            ax.legend()
            ax.set(xlabel=r"$x$")
            ax.set(ylabel=r"$y$")
            plt.xlim(
                left=self._input_domain[0, 0],
                right=self._input_domain[0, 1],
            )
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
            fig, ax = plt.subplots()
            cs = ax.contour(X1, X2, Y, 15)
            plt.colorbar(cs)
            ax.set(xlabel=r"$x_1$")
            ax.set(ylabel=r"$x_2$")
            if save_figure is True:
                fig.savefig(self.__class__.__name__, dpi=300)
            plt.show()
        else:
            raise ValueError("Unexpected value of 'num_dimension'!", num_dim)


class Forrester(SingleFidelityFunctions):
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
    low_fidelity: bool = None
    cost_ratio: list = None

    @classmethod
    def is_dim_compatible(cls, num_dim):
        assert (
            num_dim == cls.num_dim
        ), f"Can not change dimension for {cls.__name__} function"

        return num_dim

    def __init__(self, num_dim: int = 1) -> None:
        # check the dimension
        self.is_dim_compatible(num_dim=num_dim)

    @staticmethod
    def f(x: np.ndarray) -> np.ndarray:
        obj = (6 * x - 2) ** 2 * np.sin(12 * x - 4)
        obj = np.reshape(obj, (x.shape[0], 1))

        return obj


class Branin(SingleFidelityFunctions):

    num_dim: int = 2
    num_obj: int = 1
    num_cons: int = 0
    input_domain = np.array([[-5.0, 10.0], [0.0, 15.0]])
    design_space: dict = {"x1": [-5.0, 10.0], "x2": [0.0, 15.0]}
    optimum: float = 0.397887
    optimum_scheme: list = [[-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475]]
    low_fidelity: list = None

    @classmethod
    def is_dim_compatible(cls, num_dim):
        assert (
            num_dim == cls.num_dim
        ), f"Can not change dimension for {cls.__name__} function"

        return num_dim

    def __init__(self, num_dim: int = 2) -> None:
        """
        Initialization
        """

        self.is_dim_compatible(num_dim=num_dim)

    @staticmethod
    def f(x: np.ndarray) -> np.ndarray:
        a = 1
        b = 5.1 / (4 * np.pi**2)
        c = 5.0 / np.pi
        d = 6.0
        h = 10.0
        ff = 1 / (8 * np.pi)
        x1 = x[:, 0]
        x2 = x[:, 1]

        obj = (
            a * (x2 - b * x1**2 + c * x1 - d) ** 2
            + h * (1 - ff) * np.cos(x1)
            + h
        )
        obj = np.reshape(obj, (x.shape[0], 1))
        return obj


class GoldPrice(SingleFidelityFunctions):

    num_dim: int = 2
    num_obj: int = 1
    num_cons: int = 0
    input_domain = np.array([[-2.0, 2.0], [-2.0, 2.0]])
    design_space: dict = {"x1": [-2.0, 2.0], "x2": [-2.0, 2.0]}
    optimum: float = 3.0
    optimum_scheme: list = [0.0, 1.0]
    low_fidelity: list = None

    @classmethod
    def is_dim_compatible(cls, num_dim):
        assert (
            num_dim == cls.num_dim
        ), f"Can not change dimension for {cls.__name__} function"

        return num_dim

    def __init__(self, num_dim: int = 2) -> None:
        """
        Initialization
        """

        self.is_dim_compatible(num_dim=num_dim)

    @staticmethod
    def f(x: np.ndarray) -> np.ndarray:
        x1 = x[:, 0]
        x2 = x[:, 1]
        obj = (
            1
            + (x1 + x2 + 1) ** 2
            * (
                19
                - 14 * x1
                + 3 * x1**2
                - 14 * x2
                + 6 * x1 * x2
                + 3 * x2**2
            )
        ) * (
            30
            + (2 * x1 - 3 * x2) ** 2
            * (
                18
                - 32 * x1
                + 12 * x1**2
                + 48 * x2
                - 36 * x1 * x2
                + 27 * x2**2
            )
        )
        obj = np.reshape(obj, (x.shape[0], 1))
        return obj


class Sixhump(SingleFidelityFunctions):

    num_dim: int = 2
    num_obj: int = 1
    num_cons: int = 0
    input_domain = np.array([[-3.0, 3.0], [-2.0, 2.0]])
    design_space: dict = {"x1": [-3.0, 3.0], "x2": [-2.0, 2.0]}
    optimum: float = -1.0316
    optimum_scheme: list = [[0.0898, -0.7126], [-0.0898, 0.7126]]
    low_fidelity: list = None

    @classmethod
    def is_dim_compatible(cls, num_dim):
        assert (
            num_dim == cls.num_dim
        ), f"Can not change dimension for {cls.__name__} function"

        return num_dim

    def __init__(self, num_dim: int = 2) -> None:
        """
        Initialization
        """

        self.is_dim_compatible(num_dim=num_dim)

    @staticmethod
    def f(x: np.ndarray) -> np.ndarray:
        x1 = x[:, 0]
        x2 = x[:, 1]

        term1 = (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2
        term2 = x1 * x2
        term3 = (-4 + 4 * x2**2) * x2**2

        obj = term1 + term2 + term3
        obj = np.reshape(obj, (x.shape[0], 1))
        return obj


class Sasena(SingleFidelityFunctions):

    num_dim: int = 2
    num_obj: int = 1
    num_cons: int = 0
    input_domain = np.array([[0.0, 5.0], [0.0, 5.0]])
    design_space: dict = {"x1": [0.0, 5.0], "x2": [0.0, 5.0]}
    optimum: float = -1.4565
    optimum_scheme: list = [2.5044, 2.5778]
    low_fidelity: list = None

    @classmethod
    def is_dim_compatible(cls, num_dim):
        assert (
            num_dim == cls.num_dim
        ), f"Can not change dimension for {cls.__name__} function"

        return num_dim

    def __init__(self, num_dim: int = 2) -> None:
        """
        Initialization
        """

        self.is_dim_compatible(num_dim=num_dim)

    @staticmethod
    def f(x: np.ndarray) -> np.ndarray:
        x1 = x[:, 0]
        x2 = x[:, 1]
        obj = (
            2
            + 0.01 * (x2 - x1**2) ** 2
            + (1 - x1) ** 2
            + 2 * (2 - x2) ** 2
            + 7 * np.sin(0.5 * x1) * np.sin(0.7 * x1 * x2)
        )
        obj = np.reshape(obj, (x.shape[0], 1))
        return obj


class Hartman3(SingleFidelityFunctions):

    num_dim: int = 3
    num_obj: int = 1
    num_cons: int = 0
    input_domain = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    design_space: dict = {"x1": [0.0, 1.0], "x2": [0.0, 1.0], "x3": [0.0, 1.0]}
    optimum: float = -3.86278214782076
    optimum_scheme: list = [0.1, 0.55592003, 0.85218259]
    low_fidelity: list = None

    @classmethod
    def is_dim_compatible(cls, num_dim):
        assert (
            num_dim == cls.num_dim
        ), f"Can not change dimension for {cls.__name__} function"

        return num_dim

    def __init__(self, num_dim: int = 3) -> None:
        """
        Initialization
        """

        self.is_dim_compatible(num_dim=num_dim)

    @staticmethod
    def f(x: np.ndarray) -> np.ndarray:
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
        obj = np.reshape(obj, (x.shape[0], 1))
        return obj


class Hartman6(SingleFidelityFunctions):

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

    @classmethod
    def is_dim_compatible(cls, num_dim):
        assert (
            num_dim == cls.num_dim
        ), f"Can not change dimension for {cls.__name__} function"

        return num_dim

    def __init__(self, num_dim: int = 6) -> None:
        """
        Initialization
        """

        self.is_dim_compatible(num_dim=num_dim)

    @staticmethod
    def f(x: np.ndarray) -> np.ndarray:
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


class Thevenot(SingleFidelityFunctions):

    num_dim: int = None
    num_obj: int = 1
    num_cons: int = 0
    design_space: dict = {}
    input_domain: np.ndarray = None
    optimum: float = None
    optimum_scheme: list = None
    low_fidelity: list = None

    @classmethod
    def is_dim_compatible(cls, d) -> any:
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def __init__(self, num_dim: int, m: float = 5.0, beta: float = 15) -> None:
        """

        Parameters
        ----------
        num_dim : int
            dimension of Thevenot function
        m:
        beta
        """
        self.is_dim_compatible(d=num_dim)
        self.num_dim = num_dim
        self.input_domain = np.array(
            [[-2 * np.pi, 2 * np.pi] for _ in range(num_dim)]
        )
        self.m = m
        self.beta = beta
        # update the information of function class
        self.__update_parameters()

    def f(self, x: np.ndarray) -> np.ndarray:
        # only can compute sequentially
        if x.ndim == 1:
            res = np.exp(-np.sum((x / self.beta) ** (2 * self.m)))
            res = res - 2 * np.exp(-np.prod(x**2)) * np.prod(np.cos(x) ** 2)
            obj = res
        elif x.ndim == 2:
            obj = np.zeros((x.shape[0], 1))
            for ii in range(x.shape[0]):
                X = x[ii, :]
                res = np.exp(-np.sum((X / self.beta) ** (2 * self.m)))
                res = res - 2 * np.exp(-np.prod(X**2)) * np.prod(
                    np.cos(X) ** 2
                )
                obj[ii, 0] = res
            obj = np.reshape(obj, (x.shape[0], 1))

        return obj

    def __update_parameters(self):

        self.__class__.num_dim = self.num_dim
        for ii in range(self.input_domain.shape[0]):
            self.__class__.design_space[f"x{ii + 1}"] = self.input_domain[
                ii, :
            ].tolist()
        self.__class__.optimum_scheme = np.array(
            [0 for i in range(1, self.num_dim + 1)]
        ).tolist()
        self.__class__.optimum = self.f(
            np.array([0 for i in range(1, self.num_dim + 1)])
        )
        self.__class__.input_domain = self.input_domain


class Ackley(SingleFidelityFunctions):

    num_dim: int = None
    num_obj: int = 1
    num_cons: int = 0
    design_space: dict = {}
    input_domain: np.ndarray = None
    optimum: float = None
    optimum_scheme: list = None
    low_fidelity: list = None

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def __init__(
        self, num_dim: int, a: float = 20, b: float = 0.2, c: float = 2 * np.pi
    ) -> None:
        """

        Parameters
        ----------
        num_dim: int
            number of dimension
        a:float
            Parameters
        b:float
            Parameters
        c: float
            Parameters
        """
        self.is_dim_compatible(d=num_dim)
        self.num_dim = num_dim
        self.input_domain = np.array([[-32, 32] for _ in range(num_dim)])
        self.a = a
        self.b = b
        self.c = c
        # update the information of function class
        self.__update_parameters()

    def get_param(self) -> dict[str, float]:
        return {"a": self.a, "b": self.b, "c": self.c}

    def f(self, x: np.ndarray) -> np.ndarray:

        if x.ndim == 1:
            res = -self.a * np.exp(-self.b * np.sqrt(np.mean(x**2)))
            res = (
                res - np.exp(np.mean(np.cos(self.c * x))) + self.a + np.exp(1)
            )
            obj = res
        elif x.ndim == 2:
            obj = np.zeros((x.shape[0], 1))
            for ii in range(x.shape[0]):
                X = x[ii, :]
                res = -self.a * np.exp(-self.b * np.sqrt(np.mean(X**2)))
                res = (
                    res
                    - np.exp(np.mean(np.cos(self.c * X)))
                    + self.a
                    + np.exp(1)
                )
                obj[ii, 0] = res
        return obj

    def __update_parameters(self) -> None:
        """
        update the class variable information
        Returns
        -------

        """

        self.__class__.num_dim = self.num_dim

        for ii in range(self.input_domain.shape[0]):
            self.__class__.design_space[f"x{ii + 1}"] = self.input_domain[
                ii, :
            ].tolist()
        self.__class__.optimum_scheme = np.array(
            [0 for _ in range(1, self.num_dim + 1)]
        ).tolist()
        self.__class__.optimum = self.f(
            np.array([0 for _ in range(1, self.num_dim + 1)])
        )
        self.__class__.input_domain = self.input_domain


class AckleyN2(SingleFidelityFunctions):

    num_dim: int = 2
    num_obj: int = 1
    num_cons: int = 0
    # low_bound: list = [-32.0, -32.0]
    # high_bound: list = [32.0, 32.0]
    input_domain: np.ndarray = np.array([[-32.0, 32.0], [-32.0, 32.0]])
    design_space: dict = {"x1": [-32.0, 32.0], "x2": [-32.0, 32.0]}
    optimum: float = [0.0, 0.0]
    optimum_scheme: list = -200.0
    low_fidelity: list = None

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (
            isinstance(d, int) and (not d < 0)
        ), "The dimension d must be None or a positive integer"
        return d == 2

    def __init__(self, num_dim: int = 2) -> None:
        self.is_dim_compatible(d=num_dim)

    @staticmethod
    def f(x: np.ndarray) -> np.ndarray:

        if x.ndim == 1:
            x1, x2 = x
            res = -200 * np.exp(-0.2 * np.sqrt(x1**2 + x2**2))
            obj = res
        elif x.ndim == 2:
            obj = np.zeros((x.shape[0], 1))
            for ii in range(x.shape[0]):
                X = x[ii, :]
                x1, x2 = X
                res = -200 * np.exp(-0.2 * np.sqrt(x1**2 + x2**2))
                obj[ii, 0] = res
        return obj
