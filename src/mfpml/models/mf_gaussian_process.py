
import time
from typing import Any, Dict, List

import numpy as np

from mfpml.models.gaussian_process import GaussianProcessRegression


class _mfGaussianProcess:

    def __init__(self,
                 design_space: np.ndarray) -> None:
        self.design_space = design_space
        self.lfGP: GaussianProcessRegression = None

    def _train_hf(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Train the high-fidelity model

        Parameters
        ----------
        X : np.ndarray
            array of high-fidelity samples
        Y : np.ndarray
            array of high-fidelity responses
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def train(self,
              samples: List,
              responses: List) -> None:
        """training for multi-fidelity Gaussian process regression model, it is 
        for two-fidelity model, where the first fidelity is high-fidelity and the
        second fidelity is low-fidelity. 

        Parameters
        ----------
        samples : List
            list with two elements, where each element is a np.ndarray
            of samples. The first element is high-fidelity samples and
            the second element is low-fidelity samples.
        responses : List
            list with two elements, where each element is a np.ndarray
            of responses. The first element is high-fidelity responses and
            the second element is low-fidelity responses.
        """
        # record the execution time for training low-fidelity model
        clock_start = time.time()
        self._train_lf(samples[1], responses[1])
        clock_lf = time.time()
        # train high-fidelity model, it will be trained at child-class
        self._train_hf(samples[0], responses[0])
        clock_hf = time.time()
        # record the training time for low-fidelity and high-fidelity model
        self.lf_training_time = clock_lf - clock_start
        self.hf_training_time = clock_hf - clock_lf
        # print the training time
        print(f"Training time of low-fidelity model: {self.lf_training_time}")
        print(f"Training time of high-fidelity model: {self.hf_training_time}")

    def _train_lf(self,
                  X: np.ndarray,
                  Y: np.ndarray) -> None:
        """Train the low-fidelity model

        Parameters
        ----------
        X : np.ndarray
            low-fidelity samples
        Y : np.ndarray
            low-fidelity responses
        """
        # gaussian process regression will normalize the input directly
        self.lfGP.train(X, Y)
        # normalize the input
        self.sample_xl = X
        self.sample_xl_scaled = self.normalize_input(X)
        self.sample_yl = Y

    def predict_lf(
        self, X: np.ndarray, return_std: bool = False
    ) -> np.ndarray:
        """Predict low-fidelity responses

        Parameters
        ----------
        X : np.ndarray
            array of low-fidelity to be predicted
        return_std : bool, optional
            whether to return the standard deviation, by default False

        Returns
        -------
        np.ndarray
            prediction of low-fidelity
        """
        return self.lfGP.predict(X, return_std)

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
            return self.lfGP.kernel.get_kernel_matrix(X, Xprime)
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
        return (inputs - self.design_space[:, 0]) / (
            self.design_space[:, 1] - self.design_space[:, 0]
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

    @ property
    def _get_lfGP(self) -> Any:
        """Get the low-fidelity model

        Returns
        -------
        Any
            low-fidelity model instance
        """

        return self.lfGP

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
        return self.lfGP._num_samples

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
