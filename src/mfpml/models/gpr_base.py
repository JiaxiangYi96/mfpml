
from typing import Any

import numpy as np
from matplotlib import pyplot as plt


class SingleFidelityGP:
    """base class for Gaussian Process models"""

    def train(self, sample_x: np.ndarray, sample_y: np.ndarray) -> None:
        """traning procedure of gpr models

        Parameters
        ----------
        sample_x : np.ndarray
            sample array of sample
        sample_y : np.ndarray
            responses of the sample
        """
        # get number samples
        self.num_samples = sample_x.shape[0]
        # the original sample_x
        self.sample_x = sample_x
        self.sample_scaled_x = self.normalize_input(sample_x, self.bounds)
        # get the response
        self.sample_y = sample_y.reshape(-1, 1)
        self.sample_y_scaled = self.normalize_output(self.sample_y)

        # optimizer hyper-parameters
        self._hyper_paras_optimization()

        # update kernel matrix info with optimized hyper-parameters
        self._update_kernel_matrix()

    def update_optimizer(self, optimizer: Any) -> None:
        """Change the optimizer for optimizing hyper parameters

        Parameters
        ----------
        optimizer : any
            instance of optimizer
        """
        self.optimizer = optimizer

    def update_model(self, update_x: np.ndarray, update_y: np.ndarray) -> None:
        """update the model with new samples

        Parameters
        ----------
        update_x : np.ndarray
            update sample array
        update_y : np.ndarray
            update responses
        """

        sample_x = np.concatenate((self.sample_x, update_x))
        sample_y = np.concatenate((self.sample_y, update_y))
        # update the model
        self.train(sample_x, sample_y)

    def plot_prediction(self, fig_name: str = "gpr_pred",
                        save_fig: bool = False, **kwargs,
                        ) -> None:
        """plot model prediction

        Parameters
        ----------
        fig_name : str, optional
            figure name, by default "gpr_pred"
        save_fig : bool, optional
            save figure otr not, by default False
        """

        if self.num_dim == 1:
            x_plot = np.linspace(
                start=self.bounds[0, 0], stop=self.bounds[0, 1], num=1000
            )
            x_plot = x_plot.reshape((-1, 1))
            y_pred, y_sigma = self.predict(x_plot, return_std=True)
            # with plt.style.context(["ieee", "science"]):
            fig, ax = plt.subplots(**kwargs)
            ax.plot(self.sample_x, self.sample_y, "ro", label="samples",)
            ax.plot(x_plot, y_pred, "--", color='b', label="pred mean")
            ax.fill_between(
                x_plot.ravel(),
                (y_pred + 2 * y_sigma).ravel(),
                (y_pred - 2 * y_sigma).ravel(),
                color="g",
                alpha=0.3,
                label=r"95% confidence interval",
            )
            ax.tick_params(axis="both", which="major", labelsize=12)
            plt.legend(loc='best')
            plt.xlabel(r"$x$", fontsize=12)
            plt.ylabel(r"$y$", fontsize=12)
            plt.grid()
            if save_fig is True:
                fig.savefig(fig_name, dpi=300, bbox_inches="tight")
            plt.show()
        elif self.num_dim == 2:
            # 2d visualization
            num_plot = 200
            x1_plot = np.linspace(
                start=self.bounds[0, 0],
                stop=self.bounds[0, 1],
                num=num_plot,
            )
            x2_plot = np.linspace(
                start=self.bounds[1, 0],
                stop=self.bounds[1, 1],
                num=num_plot,
            )
            X1, X2 = np.meshgrid(x1_plot, x2_plot)
            pred = np.zeros([len(X1), len(X2)])
            # get the values of Y at each mesh grid
            for i in range(len(X1)):
                for j in range(len(X1)):
                    xy = np.array([X1[i, j], X2[i, j]])
                    xy = np.reshape(xy, (1, 2))
                    pred[i, j] = self.predict(xy)
            fig, ax = plt.subplots()
            ax.plot(self.sample_x[:, 0],
                    self.sample_x[:, 1], "ro", label="samples")
            cs = ax.contour(X1, X2, pred, 15)
            plt.colorbar(cs)
            ax.set(xlabel=r"$x_1$")
            ax.set(ylabel=r"$x_2$")
            ax.legend(loc="best")
            if save_fig:
                fig.savefig(self.__class__.__name__, dpi=300)
            plt.show()
        else:
            raise ValueError("Only support 1d and 2d visualization")

    def normalize_output(self, sample_y: np.ndarray) -> np.ndarray:
        """Normalize output to range [0, 1]

        Parameters
        ----------
        sample_y : np.ndarray
            output to scale

        Returns
        -------
        np.ndarray
            normalized output
        """
        # normalize the output
        self.y_mean = np.mean(sample_y)
        self.y_std = np.std(sample_y)

        return (sample_y - self.y_mean) / self.y_std

    @staticmethod
    def normalize_input(sample_x: np.ndarray,
                        bounds: np.ndarray) -> np.ndarray:
        """Normalize samples to range [0, 1]

        Parameters
        ----------
        sample_x : np.ndarray
            samples to scale
        bounds : np.ndarray
            bounds with shape=((num_dim, 2))

        Returns
        -------
        np.ndarray
            normalized samples
        """
        return (sample_x - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])

    @property
    def _num_samples(self) -> int:
        """Return the number of samples

        Returns
        -------
        num_samples : int
            num samples
        """
        return self.sample_x.shape[0]


class MultiFidelityGP:

    def train(self, samples: dict, responses: dict) -> None:
        """Train the hierarchical Kriging model

        Parameters
        ----------
        samples : dict
            dict with two keys, 'hf' contains np.ndarray of
            high-fidelity sample points and 'lf' contains
            low-fidelity
        responses : dict
            dict with two keys, 'hf' contains high-fidelity
            responses and 'lf' contains low-fidelity ones
        """
        # train the low-fidelity model
        self._train_lf(samples["lf"], responses["lf"])
        # train high-fidelity model, it will be trained at child-class
        self._train_hf(samples["hf"], responses["hf"])

    def update_model(self, Xnew: dict, Ynew: dict) -> None:
        """Update the multi-fidelity model with new samples

        Parameters
        ----------
        Xnew : dict
            dict with two keys, where contains the new samples
            If value is None then no sample to update
        Ynew : dict
            corresponding responses w.r.t. Xnew
        """
        XHnew = Xnew["hf"]
        YHnew = Ynew["hf"]
        XLnew = Xnew["lf"]
        YLnew = Ynew["lf"]
        if XLnew is not None and YLnew is not None:
            if XHnew is not None and YHnew is not None:
                X = {}
                Y = {}
                X["hf"] = np.concatenate((self.sample_xh, XHnew))
                Y["hf"] = np.concatenate((self.sample_yh, YHnew))
                X["lf"] = np.concatenate((self.sample_xl, XLnew))
                Y["lf"] = np.concatenate((self.sample_yl, YLnew))
                self.train(X, Y)
            else:
                X = {}
                Y = {}
                X["hf"] = self.sample_xh
                Y["hf"] = self.sample_yh
                X["lf"] = np.concatenate((self.sample_xl, XLnew))
                Y["lf"] = np.concatenate((self.sample_yl, YLnew))
                self.train(X, Y)
        else:
            if XHnew is not None and YHnew is not None:
                XH = np.concatenate((self.sample_xh, XHnew))
                YH = np.concatenate((self.sample_yh, YHnew))
                self._train_hf(XH, YH)

    def _update_optimizer_hf(self, optimizer: Any) -> None:
        """Change the optimizer for high-fidelity hyper parameters

        Parameters
        ----------
        optimizer : any
            instance of optimizer
        """
        self.optimizer = optimizer

    def _update_optimizer_lf(self, optimizer: Any) -> None:
        """Change the optimizer for low-fidelity hyper parameters

        Parameters
        ----------
        optimizer : Any
            instance of optimizer
        """
        # update the optimizer for low-fidelity model
        self.lf_model.update_optimizer(optimizer)

    @ property
    def _get_lf_model(self) -> Any:
        """Get the low-fidelity model

        Returns
        -------
        Any
            low-fidelity model instance
        """

        return self.lf_model

    @ property
    def _num_xh(self) -> int:
        """Return the number of high-fidelity samples

        Returns
        -------
        int
            #high-fidelity samples
        """
        return self.sample_xh.shape[0]

    @ property
    def _num_xl(self) -> int:
        """Return the number of low-fidelity samples

        Returns
        -------
        int
            #low-fidelity samples
        """
        return self.lf_model._num_samples

    @ property
    def _get_sample_hf(self) -> np.ndarray:
        """Return samples of high-fidelity

        Returns
        -------
        np.ndarray
            high-fidelity samples
        """
        return self.sample_xh

    @ property
    def _get_sample_lf(self) -> np.ndarray:
        """Return samples of high-fidelity

        Returns
        -------
        np.ndarray
            high-fidelity samples
        """
        return self.sample_xl

    def _train_lf(self, sample_xl: np.ndarray, sample_yl: np.ndarray) -> None:
        """Train the low-fidelity model

        Parameters
        ----------
        sample_xl : np.ndarray
            low-fidelity samples
        sample_yl : np.ndarray
            low-fidelity responses
        """
        # Kriging model will normalize the input directly
        self.lf_model.train(sample_xl, sample_yl)
        # normalize the input
        self.sample_xl = sample_xl
        self.sample_xl_scaled = self.normalize_input(sample_xl)
        self.sample_yl = sample_yl

    def predict_lf(
        self, test_xl: np.ndarray, return_std: bool = False
    ) -> np.ndarray:
        """Predict low-fidelity responses

        Parameters
        ----------
        test_xl : np.ndarray
            array of low-fidelity to be predicted
        return_std : bool, optional
            whether to return the standard deviation, by default False

        Returns
        -------
        np.ndarray
            prediction of low-fidelity
        """
        return self.lf_model.predict(test_xl, return_std)

    def _eval_corr(
        self, X: np.ndarray, Xprime: np.ndarray, fidelity="hf"
    ) -> np.ndarray:
        """Evaluate the correlation values based on current multi-
        fidelity model

        Parameters
        ----------
        X : np.ndarray
            x
        Xprime : np.ndarray
            x'
        fidelity : str, optional
            str indicating fidelity level, by default 'hf'

        Returns
        -------
        np.ndarray
            correlation matrix
        """
        X = self.normalize_input(X)
        Xprime = self.normalize_input(Xprime)
        if fidelity == "hf":
            return self.kernel.get_kernel_matrix(X, Xprime)
        elif fidelity == "lf":
            return self.lf_model.kernel.get_kernel_matrix(X, Xprime)
        else:
            ValueError("Unknown fidelity input.")

    def normalize_input(self, inputs: np.ndarray) -> np.ndarray:
        """Normalize samples to range [0, 1]

        Parameters
        ----------
        inputs : np.ndarray
            samples to scale
        Returns
        -------
        np.ndarray
            normalized samples
        """
        return (inputs - self.bounds[:, 0]) / (
            self.bounds[:, 1] - self.bounds[:, 0]
        )

    def normalize_hf_output(self, outputs: np.ndarray) -> np.ndarray:
        """Normalize output to normal distribution

        Parameters
        ----------
        outputs : np.ndarray
            output to scale

        Returns
        -------
        np.ndarray
            normalized output
        """
        self.yh_mean = np.mean(outputs)
        self.yh_std = np.std(outputs)
        return (outputs - self.yh_mean) / self.yh_std
