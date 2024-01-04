import numpy as np
from matplotlib import pyplot as plt

from .functions import Functions


class SFRFunctions(Functions):

    def plot_limit_state(self, save_fig: bool = True, **kwargs) -> None:

        num_dim = self._get_dimension
        num_plot = 200
        if num_dim == 2:
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
            fig, ax = plt.subplots(**kwargs)
            ax.contour(
                X1, X2, Y, levels=[-10**6, 0, 10**6],
                colors='#EE6677', linewidths=2)
            plt.xlabel(r"$x_1$", fontsize=12)
            plt.ylabel(r"$x_2$", fontsize=12)
            plt.grid()
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(1.5)
                ax.spines[axis].set_color('k')
            if save_fig is True:
                fig.savefig(self.__class__.__name__,
                            dpi=300, bbox_inches="tight")
            plt.show()

    def prob_eval(self, num_mcs: int = 10**6,
                  seed: int = 123) -> tuple[float, float]:

        # Generate samples from the multivariate normal distribution
        search_x = self.generate_multivariate_samples(
            num_mcs=num_mcs, seed=seed)

        # Evaluate the limit state function
        search_y = self.f(search_x)

        # Calculate the probability of failure
        prob_fail = float(np.sum(search_y <= 0) / num_mcs)

        # calculate the variance of the probability of failure
        prob_fail_var = np.sqrt(prob_fail * (1 - prob_fail) / num_mcs)

        return prob_fail, prob_fail_var

    def prob_cal(self, search_x: np.ndarray) -> tuple[float, float]:

        # Evaluate the limit state function
        search_y = self.f(search_x)

        # Calculate the probability of failure
        prob_fail = float(np.sum(search_y <= 0) / len(search_y))

        # calculate the variance of the probability of failure
        prob_fail_var = np.sqrt((1 - prob_fail) / (len(search_y)*prob_fail))

        return prob_fail, prob_fail_var

    def generate_multivariate_samples(self,
                                      num_mcs: int = 10**6,
                                      seed: int = 123) -> np.ndarray:
        # set seed
        np.random.seed(seed)
        # Replicate mean and standard deviation
        Mu = np.tile(self.mu, (num_mcs, 1))
        Sigma = np.tile(self.sigma, (num_mcs, 1))

        # Generate samples from the multivariate normal distribution
        self.search_x = np.random.normal(Mu, Sigma)

        # Return the generated samples
        return self.search_x


class MultiModal(SFRFunctions):
    """_summary_

    Parameters
    ----------
    SingleFidelityFunctions : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    num_dim = 2
    mu = np.array([1.5, 2.5])
    sigma = np.array([1.0, 1.0])
    input_domain = np.array([mu-5*sigma, mu+5*sigma]).T
    design_space = {"x1": [mu[0]-5*sigma[0], mu[0]+5*sigma[0]],
                    "x2": [mu[1]-5*sigma[1], mu[1]+5*sigma[1]]}

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
        """
        x1 = x[:, 0]
        x2 = x[:, 1]
        y = ((x1**2 + 4) * (x2 - 1)) / 20 - np.sin((5 * x1) / 2) - 2
        y = np.reshape(y, (len(y), 1))
        return -y


class FourBranches(SFRFunctions):
    """_summary_

    Parameters
    ----------
    SingleFidelityFunctions : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    num_dim = 2
    mu = np.array([0.0, 0.0])
    sigma = np.array([1.0, 1.0])
    input_domain = np.array([mu-5*sigma, mu+5*sigma]).T
    design_space = {"x1": [mu[0]-5*sigma[0], mu[0]+5*sigma[0]],
                    "x2": [mu[1]-5*sigma[1], mu[1]+5*sigma[1]]}

    def f(self, x) -> np.ndarray:
        # Extract input variables x1 and x2
        x1 = x[:, 0]
        x2 = x[:, 1]

        # Four branches
        g1 = 3 + (x1 - x2)**2 / 10 - (x1 + x2) / np.sqrt(2)
        g2 = 3 + (x1 - x2)**2 / 10 + (x1 + x2) / np.sqrt(2)
        g3 = (x1 - x2) + 7 / np.sqrt(2)
        g4 = -(x1 - x2) + 7 / np.sqrt(2)

        # Combine constraints into a matrix
        g_matrix = np.column_stack((g1, g2, g3, g4))

        # Compute the minimum constraint violation for each sample
        obj = np.min(g_matrix, axis=1)
        # reshape
        obj = np.reshape(obj, (len(obj), 1))
        return obj
