import matplotlib.pyplot as plt
import numpy as np

from mfpml.problems.functions import Functions


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
    def lf(x: np.ndarray, factor: float = None) -> np.ndarray: # type: ignore
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
    r"""Implementation of the bi-fidelity Six-hump Camel-back function
    as defined in:

        Dong, H., Song, B., Wang, P. et al. Multi-fidelity information
        fusion based on prediction of kriging. Struct Multidisc Optim
        51, 1267–1280 (2015) doi:10.1007/s00158-014-1213-9

    Function definitions:

    .. math::

        f_h(x_1, x_2) = 4x_1^2 - 2.1x_1^4 + \dfrac{x_1^6}{3} + x_1x_2 - 4x_2^2 + 4x_2^4

    .. math::

        f_l(x_1, x_2) = f_h(0.7x_1, 0.7x_2) + x_1x_2 - 15
    """

    num_dim: int = 2
    num_obj: int = 1
    num_cons: int = 0
    input_domain = np.array([[-2.0, 2.0], [-2.0, 2.0]])
    design_space: dict = {"x1": [-2.0, 2.0], "x2": [-2.0, 2.0]}
    optimum: float = -1.0316
    optimum_scheme: list = [[0.0898, -0.7126], [-0.0898, 0.7126]]
    low_fidelity: list = None

    def __init__(self, num_dim: int = 2) -> None:
        """
        Initialization
        """

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

    def __init__(self, num_dim: int = 6) -> None:
        """
        Initialization
        """

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
        l = np.array([0.75, 1, 0.8, 1.3, 0.7, 1.1])
        # number of samples
        num_samples = x.shape[0]
        obj = np.zeros((num_samples, 1))
        for i in range(num_samples):
            obj[i, :] = -np.dot(
                c,
                np.exp(-np.sum(a * (l * x[i, :] - p) ** 2, axis=1)),
            )
        obj = np.reshape(obj, (x.shape[0], 1))
        return -np.log(-obj)

class mf_Discontinuous(MultiFidelityFunctions): 
    """multi-fidelity discontinuous function 

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
                y[ii, 0] = 2*(0.5 * (6 * x[ii, 0] - 2) ** 2 * \
                                np.sin(12 * x[ii, 0] - 4) + \
                                10 * (x[ii, 0] - 0.5) - 5) - 20 * x[ii, 0] + 20
            else: 
                y[ii, 0] = 4 + 2*(3 + 0.5 * (6 * x[ii, 0] - 2) ** 2 * \
                                np.sin(12 * x[ii, 0] - 4) + \
                                10 * (x[ii, 0] - 0.5) - 5) - 20 * x[ii, 0] + 20
        
        return y.reshape((-1, 1))        

class ContinuousNonlinearCorrelation1D(MultiFidelityFunctions):
    """_summary_

    Parameters
    ----------
    MultiFidelityFunctions : _type_
        _description_

    Returns
    -------
    _type_
        _description_
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
    """_summary_

    Parameters
    ----------
    MultiFidelityFunctions : _type_
        _description_

    Returns
    -------
    _type_
        _description_
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
