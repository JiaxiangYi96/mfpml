import numpy as np 
from scipy.linalg import cholesky, solve

from .corrfunc import KRG

class Kriging: 
    """
    Kriging model 
    """
    def __init__(
        self, 
        bounds: np.ndarray, 
        mprior: int = 0) -> None: 
        """Initialize the Kriging model

        Parameters
        ----------
        bounds : np.ndarray
            array of shape=((num_dim,2)) where each row describes
            the bound for the variable on specfic dimension
        mprior : int, optional
            mean value for the prior, by default 0
        """
        self.num_dim = bounds.shape[0]
        self.kernel = KRG(theta=np.zeros((1, self.num_dim)))
        self.bounds = bounds
        self.low_bound = self.bounds[:, 0]
        self.high_bound = self.bounds[:, 1]
        self.mprior = mprior

    def train(self, X: np.ndarray, Y: np.ndarray, optimizer) -> None: 
        """Train the Kriging model

        Parameters
        ----------
        X : np.ndarray
            sample array of sample
        Y : np.ndarray
            responses of the sample
        optimizer : instance of optimizer
            optimizing the hyperparameters with the use style
            optimizer.run_optimizer(objective function, 
            number of dimension, design space of variables)
        """
        self.sample_X = X
        self.X = (X - self.low_bound) / (self.high_bound - self.low_bound)
        self.Y = Y.reshape(-1, 1)
        #optimizer hyperparameters
        self._optHyp(optimizer=optimizer)
        self.kernel.set_params(self.opt_param)
        #update parameters with optimized hyperparameters
        self.K = self.kernel.K(self.X, self.X)
        self.L = cholesky(self.K, lower=True)
        self.alpha = solve(self.L.T, solve(self.L, self.Y))
        one = np.ones((self.X.shape[0], 1))
        self.beta = solve(self.L.T, solve(self.L, one))
        self.mu = (np.dot(one.T, self.alpha) / np.dot(one.T, self.beta)).squeeze()
        self.gamma = solve(self.L.T, solve(self.L, (self.Y - self.mu)))
        self.sigma2 = np.dot((self.Y - self.mu).T, self.gamma) / self.X.shape[0]
        self.logp = (-.5 * self.X.shape[0] * np.log(self.sigma2) - np.sum(np.log(np.diag(self.L)))).ravel()

    def predict(self, Xinput: np.ndarray, return_std: bool=False): 
        """Predict responses through the Kriging model

        Parameters
        ----------
        Xinput : np.ndarray
            new sample need to predict
        return_std : bool, optional
            whether return the standard deviation
            , by default False

        Returns
        -------
        np.ndarray
            return the prediction with shape (#Xinput, 1)
        """
        Xnew = (Xinput - self.low_bound) / (self.high_bound - self.low_bound)
        Xnew = np.atleast_2d(Xnew)
        knew = self.kernel.K(self.X, Xnew) 
        fmean = self.mu + np.dot(knew.T, self.gamma)
        if not return_std: 
            return fmean.reshape(-1, 1)
        else:
            one = np.ones((self.X.shape[0], 1))
            delta = solve(self.L.T, solve(self.L, knew))
            mse = self.sigma2 * (1 - np.diag(np.dot(knew.T, delta)) + \
                np.diag((1 - knew.T.dot(delta)) ** 2 / one.T.dot(self.beta)))
            return fmean.reshape(-1, 1), np.sqrt(np.maximum(mse, 0)).reshape(-1, 1)
            
    def getkernelparams(self): 
        pass 
        
    def _optHyp(self, optimizer, grads: bool = None): 
        optRes = optimizer.run_optimizer(self._logLikelihood, num_dim=self.kernel.num_para, design_space=self.kernel.bounds)
        self.opt_param = optRes['best_x']

    def _logLikelihood(self, params): 
        out = np.zeros(params.shape[0])
        for i in range(params.shape[0]):
            param = params[i, :]
            #correlation matrix R
            K = self.kernel(self.X, self.X, param)
            L = cholesky(K, lower=True)
            #R^(-1)Y
            alpha = solve(L.T, solve(L, self.Y))
            one = np.ones((self.X.shape[0], 1))
            #R^(-1)1
            beta = solve(L.T, solve(L, one))
            #1R^(-1)Y / 1R^(-1)vector(1)
            mu = (np.dot(one.T, alpha) / np.dot(one.T, beta)).squeeze()
            gamma = solve(L.T, solve(L, (self.Y - mu))) 
            sigma2 = np.dot((self.Y - mu).T, gamma) / self.X.shape[0]
            logp = -.5 * self.X.shape[0] * np.log(sigma2) - np.sum(np.log(np.diag(L)))
            out[i] = logp.ravel()
        return (- out)
