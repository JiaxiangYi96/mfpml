from abc import ABC

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

            with plt.style.context(["ieee", "science"]):
                fig, ax = plt.subplots()
                ax.plot(
                    x_plot,
                    self.f(x=x_plot),
                    label=f"{self.__class__.__name__}",
                )
                ax.legend()
                ax.set(xlabel=r"$x$")
                ax.set(ylabel=r"$y$")
                # ax.autoscale(tight=True)
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
            with plt.style.context(["ieee", "science"]):
                fig, ax = plt.subplots()
                cs = ax.contour(X1, X2, Y, 15)
                plt.colorbar(cs)
                ax.set(xlabel=r"$x_1$")
                ax.set(ylabel=r"$x_2$")
                if save_figure is True:
                    fig.savefig(self.__class__.__name__, dpi=300)
                plt.show(block=True)
                plt.interactive(False)
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
    """
    Branin Test Function for Nonlinear Optimization
    Taken from "Towards Global Optimisation 2",edited by L.C.W. Dixon and G.P.
    Szego, North-Holland Publishing Company, 1978. ISBN 0 444 85171 2

    -5 <= x1 <= 10
     0 <= x2 <= 15
    fmin = 0.397887357729739
    xmin =   9.42477796   -3.14159265  3.14159265
             2.47499998   12.27500000  2.27500000
    """

    num_dim: int = 2
    num_obj: int = 1
    num_cons: int = 0
    input_domain = np.array([[-5.0, 10.0], [0.0, 15.0]])
    # low_bound: list = [-5.0, 0.0]
    # high_bound: list = [10.0, 15.0]
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
    """
    Goldstein-Price Test Function for Nonlinear Optimization
    Taken from "Towards Global Optimisation 2",edited by L.C.W. Dixon and G.P.
    Szego, North-Holland Publishing Company, 1978. ISBN 0 444 85171 2

    -2 <= x1 <= 2
    -2 <= x2 <= 2
    fmin = 3
    xmin =[ 0, -1]

    http://www4.ncsu.edu/~definkel/research/index.html
    """

    num_dim: int = 2
    num_obj: int = 1
    num_cons: int = 0
    # low_bound: list = [-2.0, -2.0]
    # high_bound: list = [2.0, 2.0]
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
    """
    6-Hump Camel Test Function for Nonlinear Optimization

    Taken from "Towards Global Optimisation 2",edited by L.C.W. Dixon and G.P.
    Szego, North-Holland Publishing Company, 1978. ISBN 0 444 85171 2

     -3 <= x1 <= 3
     -2 <= x2 <= 2
    fmin = -1.0316284535
    xmin = 0.08984201  -0.08984201
          -0.71265640   0.71265640

    http://www4.ncsu.edu/~definkel/research/index.html

    """

    num_dim: int = 2
    num_obj: int = 1
    num_cons: int = 0
    # low_bound: list = [-3.0, -2.0]
    # high_bound: list = [3.0, 2.0]
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
    """
    Sasena function (called mystery function by Sasena)
    Sasena, M. J., ¡°Flexibility and Efficiency Enhancements for
    Constrained Global Design Optimization with Kriging Approximations,¡±
    Ph.D. Thesis, Univ. of Michigan, Ann Arbor, MI, 2002.
    0 <= x1 <= 5
    0 <= x2 <= 5
    xmin = [2.5044,2.5778]
    fmin = -1.4565
    """

    num_dim: int = 2
    num_obj: int = 1
    num_cons: int = 0
    # low_bound: list = [0.0, 0.0]
    # high_bound: list = [5.0, 5.0]
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
    """
    Hartman 3 Test Function for Nonlinear Optimization
    Taken from "Towards Global Optimisation 2",edited by L.C.W. Dixon and G.P.
    Szego, North-Holland Publishing Company, 1978. ISBN 0 444 85171 2

    0 <= x1 <= 1
    0 <= x2 <= 1
    0 <= x3 <= 1
    fmin = -3.86278214782076
    xmin = [0.1, 0.55592003,0.85218259]
    http://www4.ncsu.edu/~definkel/research/index.html
    """

    num_dim: int = 3
    num_obj: int = 1
    num_cons: int = 0
    # low_bound: list = [0.0, 0.0, 0.0]
    # high_bound: list = [1.0, 1.0, 1.0]
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
    """
    Hartman 6 Test Function for Nonlinear Optimization
    Taken from "Towards Global Optimization 2",edited by L.C.W. Dixon and G.P.
    Szego, North-Holland Publishing Company, 1978. ISBN 0 444 85171 2
    0 <= x1 <= 1
    0 <= x2 <= 1
    0 <= x3 <= 1
    0 <= x4 <= 1
    0 <= x5 <= 1
    0 <= x6 <= 1
    fmin = -3.32236801141551;
    xmin = [0.20168952;  0.15001069;  0.47687398;  0.27533243;  0.31165162;  0.65730054]
    http://www4.ncsu.edu/~definkel/research/index.html
    """

    num_dim: int = 6
    num_obj: int = 1
    num_cons: int = 0
    # low_bound: list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # high_bound: list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
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
    """
    latex_formula = r'f(\mathbf{x}) = exp(-\sum_{i=1}^{d}(x_i / \beta)^{2m}) - 2exp(-\prod_{i=1}^{d}x_i^2) \prod_{i=1}^{d}cos^ 2(x_i) '
    latex_formula_dimension = r'd \in \mathbb{N}_{+}^{*}'
    latex_formula_input_domain = r'x_i \in [-2\pi, 2\pi], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_minimum = r'f(0, ..., 0)=-1, \text{ for}, m=5, \beta=15'
    continuous = True
    convex = True
    separable = True
    differentiable = True
    mutimodal = True
    randomized_term = False
    parametric = True

    Created by Axel Thevenot (2020)
    Github repository: https://github.com/AxelThevenot/Python_Benchmark_Test_Optimization_Function_Single_Objective

    """

    num_dim: int = None
    num_obj: int = 1
    num_cons: int = 0
    # low_bound: list = None
    # high_bound: list = None
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
        """
        update the class variable information
        Returns
        -------

        """

        self.__class__.num_dim = self.num_dim
        # self.__class__.low_bound = np.array([-2 * np.pi for _ in range(self.num_dim)]).tolist()
        # self.__class__.high_bound = np.array([2 * np.pi for _ in range(self.num_dim)]).tolist()
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
    """
    name = 'Ackley'
    latex_formula = r'f(\mathbf{x}) = -a \cdot exp(-b\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2})-exp(\frac{1}{d}\sum_{i=1}^{d}cos(c \cdot x_i))+ a + exp(1)'
    latex_formula_dimension = r'd \in \mathbb{N}_{+}^{*}'
    latex_formula_input_domain = r'x_i \in [-32, 32], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_minimum = r'f((0, ..., 0)) = 0'
    continuous = True
    convex = False
    separable = True
    differentiable = True
    mutimodal = True
    randomized_term = False
    parametric = True

    Github repository: https://github.com/AxelThevenot/Python_Benchmark_Test_Optimization_Function_Single_Objective

    """

    num_dim: int = None
    num_obj: int = 1
    num_cons: int = 0
    # low_bound: list = None
    # high_bound: list = None
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
        # self.__class__.low_bound = np.array([-32.0 for _ in range(self.num_dim)]).tolist()
        # self.__class__.high_bound = np.array([32.0 for _ in range(self.num_dim)]).tolist()
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
    """
    name = 'Ackley N. 2'
    latex_formula = r'f(x, y) = -200exp(-0.2\sqrt{x^2 + y^2)}'
    latex_formula_dimension = r'd=2'
    latex_formula_input_domain = r'$x \in [-32, 32], y \in [-32, 32]$'
    latex_formula_global_minimum = r'f(0, 0)=-200'
    continuous = False
    convex = True
    separable = False
    differentiable = True
    mutimodal = False
    randomized_term = False
    parametric = False
    """

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
