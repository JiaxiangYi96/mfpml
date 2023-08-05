

import numpy as np
from matplotlib import pyplot as plt

from mfpml.problems.functions import Functions


def plot_sf_sampling(
    samples: np.ndarray,
    responses: np.ndarray,
    function: Functions = None,
    save_figure: bool = False,
    **kwargs,
) -> None:
    """
    Visualize the 1D case, set the y axis as Zero
    Parameters
    ----------
    save_figure: bool
        save figure
    samples : np.ndarray
        original data for visualization
    responses: np.ndarray
        responses of samples
    function:
        the original function

    Returns
    -------

    """
    num_dim = samples.shape[1]
    if num_dim == 1:
        x_plot = np.linspace(
            start=function._input_domain[0, 0],
            stop=function._input_domain[0, 1],
            num=1000,
        )
        x_plot = x_plot.reshape((-1, 1))
        y_plot = function.f(x=x_plot)
        y_plot.reshape((-1, 1))
        fig, ax = plt.subplots(**kwargs)
        ax.plot(samples[:, 0], responses[:, 0], "*", label="Samples")
        ax.plot(x_plot, y_plot, label=f"{function.__class__.__name__}")
        ax.legend()
        ax.set(xlabel=r"$x$")
        ax.set(ylabel=r"$y$")
        ax.autoscale(tight=True)
        if save_figure is True:
            fig.savefig(function.__class__.__name__, dpi=300)
        plt.show()

    elif num_dim == 2:
        num_plot = 200
        x1_plot = np.linspace(
            start=function._input_domain[0, 0],
            stop=function._input_domain[0, 1],
            num=num_plot,
        )
        x2_plot = np.linspace(
            start=function._input_domain[1, 0],
            stop=function._input_domain[1, 1],
            num=num_plot,
        )
        X1, X2 = np.meshgrid(x1_plot, x2_plot)
        Y = np.zeros([len(X1), len(X2)])
        # get the values of Y at each mesh grid
        for i in range(len(X1)):
            for j in range(len(X1)):
                xy = np.array([X1[i, j], X2[i, j]])
                xy = np.reshape(xy, (1, 2))
                Y[i, j] = function.f(x=xy)
        fig, ax = plt.subplots(**kwargs)
        plt.scatter(
            samples[:, 0],
            samples[:, 1],
            s=15,
            color="orangered",
            label="Samples",
        )
        cs = ax.contour(X1, X2, Y, 15)
        plt.colorbar(cs)
        ax.set(xlabel=r"$x_1$")
        ax.set(ylabel=r"$x_2$")
        plt.legend(
            loc="upper center", bbox_to_anchor=(1, -0.05), edgecolor="k"
        )

        if save_figure is True:
            fig.savefig(function.__class__.__name__ + ".png", dpi=300)
        plt.show()


def plot_mf_sampling(
    samples: dict,
    responses: dict,
    function: Functions = None,
    save_figure: bool = False,
    **kwargs,
) -> None:
    """
    Visualize the 1D case, set the y axis as Zero
    Parameters
    ----------
    save_figure: bool
        save figure
    samples : np.ndarray
        original data for visualization
    responses: np.ndarray
        responses of samples
    function:
        the original function

    Returns
    -------

    """
    num_dim = samples["hf"].shape[1]
    if num_dim == 1:
        x_plot = np.linspace(
            start=function._input_domain[0, 0],
            stop=function._input_domain[0, 1],
            num=1000,
        )
        x_plot = x_plot.reshape((-1, 1))
        yh_plot = function.hf(x=x_plot)
        # yh_plot.reshape((-1, 1))
        yl_plot = function.lf(x=x_plot)
        # with plt.style.context(["ieee", "science"]):
        fig, ax = plt.subplots(**kwargs)
        ax.plot(
            samples["hf"][:, 0],
            responses["hf"][:, 0],
            "*",
            label="hf Samples",
        )
        ax.plot(
            samples["lf"][:, 0],
            responses["lf"][:, 0],
            "o",
            label="ls Samples",
        )
        ax.plot(x_plot, yh_plot, "-", label="High fidelity function")
        ax.plot(x_plot, yl_plot, "--", label="Low fidelity function")
        ax.legend()
        ax.set(xlabel=r"$x$")
        ax.set(ylabel=r"$y$")
        plt.xlim(
            left=function._input_domain[0, 0],
            right=function._input_domain[0, 1],
        )
        if save_figure is True:
            fig.savefig(function.__class__.__name__, dpi=300)
        plt.show()

    elif num_dim == 2:
        num_plot = 200
        x1_plot = np.linspace(
            start=function._input_domain[0, 0],
            stop=function._input_domain[0, 1],
            num=num_plot,
        )
        x2_plot = np.linspace(
            start=function._input_domain[1, 0],
            stop=function._input_domain[1, 1],
            num=num_plot,
        )
        X1, X2 = np.meshgrid(x1_plot, x2_plot)
        Y = np.zeros([len(X1), len(X2)])
        # get the values of Y at each mesh grid
        for i in range(len(X1)):
            for j in range(len(X1)):
                xy = np.array([X1[i, j], X2[i, j]])
                xy = np.reshape(xy, (1, 2))
                Y[i, j] = function.hf(x=xy)
        fig, ax = plt.subplots(**kwargs)
        plt.scatter(
            samples["hf"][:, 0],
            samples["hf"][:, 1],
            s=15,
            color="orangered",
            label="hf samples ",
        )
        plt.scatter(
            samples["lf"][:, 0],
            samples["lf"][:, 1],
            s=15,
            color="green",
            label="lf samples ",
        )
        cs = ax.contour(X1, X2, Y, 15)
        plt.colorbar(cs)
        ax.set(xlabel=r"$x_1$")
        ax.set(ylabel=r"$x_2$")
        plt.legend(
            loc="upper center", bbox_to_anchor=(1, -0.05), edgecolor="k"
        )

        if save_figure is True:
            fig.savefig(function.__class__.__name__ + ".png", dpi=300)
        plt.show()
