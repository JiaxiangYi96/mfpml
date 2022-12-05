import numpy as np 
from scipy.linalg import cholesky, solve
from scipy.optimize import minimize

from .corrfunc import KRG

class Kriging: 
    """
    Kriging model 
    """
    def __init__(
        self, 
        design_space: np.ndarray, 
        optimizer: any = None, 
        kernel_bound: list = [-4., 3.], 
        mprior: int = 0) -> None: 
        """Initialize the Kriging model

        Parameters
        ----------
        design_space : np.ndarray
            array of shape=((num_dim,2)) where each row describes
            the bound for the variable on specfic dimension
        optimizer : any, optional
            instance of the optimizer used to optimize the hyperparameters
            with the use style optimizer.run_optimizer(objective function, 
            number of dimension, design space of variables), if not assigned,
            the 'L-BFGS-B' method in scipy is used
        kernel_bound : list, optional
            log bound for the Kriging kernel, by default [-4, 3]
        mprior : int, optional
            mean value for the prior, by default 0
        """
        self.num_dim = design_space.shape[0]
        self.kernel = KRG(theta=np.zeros(self.num_dim), bounds=kernel_bound)
        self.bounds = design_space
        self.optimizer = optimizer
        self.mprior = mprior

    def train(self, X: np.ndarray, Y: np.ndarray) -> None: 
        """Train the Kriging model

        Parameters
        ----------
        X : np.ndarray
            sample array of sample
        Y : np.ndarray
            responses of the sample
        """
        self.sample_X = X
        self.X = self.normalize_input(X, self.bounds)
        self.sample_Y = Y.reshape(-1, 1)
        #optimizer hyperparameters
        self._optHyp()
        self.kernel.set_params(self.opt_param)
        #update parameters with optimized hyperparameters
        self.K = self.kernel.K(self.X, self.X)
        self.L = cholesky(self.K, lower=True)
        self.alpha = solve(self.L.T, solve(self.L, self.sample_Y))
        one = np.ones((self.X.shape[0], 1))
        self.beta = solve(self.L.T, solve(self.L, one))
        self.mu = np.asscalar(np.dot(one.T, self.alpha) / np.dot(one.T, self.beta))
        self.gamma = solve(self.L.T, solve(self.L, (self.sample_Y - self.mu)))
        self.sigma2 = np.asscalar(np.dot((self.sample_Y - self.mu).T, self.gamma) / self.X.shape[0])
        self.logp = np.asscalar(-.5 * self.X.shape[0] * np.log(self.sigma2) - np.sum(np.log(np.diag(self.L))))

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
        Xnew = self.normalize_input(Xinput, self.bounds)
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

    def _optHyp(self, grads: bool = None): 
        """Optimize the hyperparameters

        Parameters
        ----------
        grads : bool, optional
            whether to use gradients, by default None
        """
        if self.optimizer is None:
            n_trials = 9
            opt_fs = float('inf')
            for trial in range(n_trials): 
                x0 = np.random.uniform(self.kernel._get_low_bound, 
                    self.kernel._get_high_bound, self.kernel._get_num_para)
                optRes = minimize(self._logLikelihood, x0=x0, method='L-BFGS-B',
                    bounds=self.kernel._get_bounds_list)
                if optRes.fun < opt_fs:
                    opt_param = optRes.x
                    opt_fs = optRes.fun
        else:
            optRes = self.optimizer.run_optimizer(self._logLikelihood, 
                num_dim=self.kernel._get_num_para, design_space=self.kernel._get_bounds)
            opt_param = optRes['best_x']
        self.opt_param = opt_param

    def _logLikelihood(self, params: np.ndarray) -> np.ndarray:
        """Compute the concentrated ln-likelihood

        Parameters
        ----------
        params : np.ndarray
            parameters of the kernel

        Returns
        -------
        np.ndarray
            log likelihood
        """
        params = np.atleast_2d(params)
        out = np.zeros(params.shape[0])
        for i in range(params.shape[0]):
            param = params[i, :]
            #correlation matrix R
            K = self.kernel(self.X, self.X, param)
            L = cholesky(K, lower=True)
            #R^(-1)Y
            alpha = solve(L.T, solve(L, self.sample_Y))
            one = np.ones((self.X.shape[0], 1))
            #R^(-1)1
            beta = solve(L.T, solve(L, one))
            #1R^(-1)Y / 1R^(-1)vector(1)
            mu = (np.dot(one.T, alpha) / np.dot(one.T, beta)).squeeze()
            gamma = solve(L.T, solve(L, (self.sample_Y - mu))) 
            sigma2 = np.dot((self.sample_Y - mu).T, gamma) / self.X.shape[0]
            logp = -.5 * self.X.shape[0] * np.log(sigma2) - np.sum(np.log(np.diag(L)))
            out[i] = logp.ravel()
        return (- out)

    def _update_optimizer(self, optimizer: any) -> None: 
        """Change the optimizer for optimizing hyperparameters

        Parameters
        ----------
        optimizer : any
            instance of optimizer
        """
        self.optimizer = optimizer

    @staticmethod
    def normalize_input(X: np.ndarray, bounds: np.ndarray) -> np.ndarray: 
        """Normalize samples to range [0, 1]

        Parameters
        ----------
        X : np.ndarray
            samples to scale
        bounds : np.ndarray
            bounds with shape=((num_dim, 2))

        Returns
        -------
        np.ndarray
            normalized samples
        """
        return (X - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
                
    def getkernelparams(self): 
        pass 
        
    @property
    def _num_X(self) -> int: 
        """Return the number of samples

        Returns
        -------
        int
            #samples
        """
        return self.sample_X.shape[0]