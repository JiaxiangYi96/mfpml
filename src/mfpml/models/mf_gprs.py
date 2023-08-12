

from typing import Any

import numpy as np


class mf_model:

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

    @property
    def _get_lf_model(self) -> Any:
        """Get the low-fidelity model

        Returns
        -------
        Any
            low-fidelity model instance
        """

        return self.lf_model

    @property
    def _num_xh(self) -> int:
        """Return the number of high-fidelity samples

        Returns
        -------
        int
            #high-fidelity samples
        """
        return self.sample_xh.shape[0]

    @property
    def _num_xl(self) -> int:
        """Return the number of low-fidelity samples

        Returns
        -------
        int
            #low-fidelity samples
        """
        return self.lf_model._num_samples

    @property
    def _get_sample_hf(self) -> np.ndarray:
        """Return samples of high-fidelity

        Returns
        -------
        np.ndarray
            high-fidelity samples
        """
        return self.sample_xh

    @property
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
