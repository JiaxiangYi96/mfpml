
import numpy as np 
from collections import OrderedDict
from scipy.linalg import cholesky, solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import pyswarms as ps

from mfpml.models.corrfunc import KRG

class Kriging: 
    """
    Kriging model 
    """
    def __init__(self, bounds: list, mprior: int = 0): 
        """Initialize the Kriging model

        Parameters
        ----------
        bounds : list
            a list of size #dimension where each entry is a list 
            describe the bound for the variable on specfic dimension
        mprior : int, optional
            _description_, by default 0
        """
        self.num_dim = len(bounds)
        self.kernel = KRG(theta=np.zeros((1, self.num_dim)))
        self.bounds = np.array(bounds)
        self.low_bound = self.bounds[:, 0]
        self.high_bound = self.bounds[:, 1]
        self.mprior = mprior

    def getkernelparams(self): 
        pass 

    def train(self, X: np.ndarray, Y: np.ndarray, n_iter: int = 500) -> None: 
        """
        """
        self.sample_X = X
        self.X = (X - self.low_bound) / (self.high_bound - self.low_bound)
        self.Y = Y.reshape(-1, 1)
        self.optHyp(n_iter=n_iter)
        self.kernel.set_params(self.opt_param)

        self.K = self.kernel.K(self.X, self.X) 
        self.L = cholesky(self.K, lower=True)
        self.alpha = solve(self.L.T, solve(self.L, self.Y)) 
        one = np.ones((self.X.shape[0], 1))
        self.beta = solve(self.L.T, solve(self.L, one)) 
        self.mu = (np.dot(one.T, self.alpha) / np.dot(one.T, self.beta)).squeeze()
        self.gamma = solve(self.L.T, solve(self.L, (self.Y - self.mu))) 
        self.sigma2 = np.dot((self.Y - self.mu).T, self.gamma) / self.X.shape[0]
        self.logp = (-.5 * self.X.shape[0] * np.log(self.sigma2) - np.sum(np.log(np.diag(self.L)))).ravel()
        
    def optHyp(self, n_iter, grads=None): 
        options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
        bounds = (np.array(self.low_bound), np.array(self.high_bound))
        optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=self.num_dim, options=options, bounds=bounds)
        opt_fs, opt_param = optimizer.optimize(self._logLikelihood, iters=n_iter, verbose=False)
        self.opt_param = opt_param

    def predict(self, Xinput: np.ndarray, return_std: bool=False): 
        """_summary_

        Parameters
        ----------
        Xnew : _type_
            _description_
        return_std : bool, optional
            _description_, by default False

        Returns
        -------
        _type_
            _description_
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

    def _logLikelihood(self, params): 
        out = np.zeros(params.shape[0])
        for i in range(params.shape[0]):
            param = params[i, :]
            K = self.kernel(self.X, self.X, param) 
            L = cholesky(K, lower=True)
            alpha = solve(L.T, solve(L, self.Y)) 
            one = np.ones((self.X.shape[0], 1))
            beta = solve(L.T, solve(L, one)) 
            mu = (np.dot(one.T, alpha) / np.dot(one.T, beta)).squeeze()
            gamma = solve(L.T, solve(L, (self.Y - mu))) 
            sigma2 = np.dot((self.Y - mu).T, gamma) / self.X.shape[0]
            logp = -.5 * self.X.shape[0] * np.log(sigma2) - np.sum(np.log(np.diag(L)))
            out[i] = logp.ravel()
        return (- out)


class mf_model: 

    def update(self, Xnew: dict, Ynew: dict) -> None: 
        XHnew = Xnew['hf']
        YHnew = Ynew['hf']
        XLnew = Xnew['lf']
        YLnew = Ynew['lf']
        if XLnew is not None and YLnew is not None: 
            if XHnew is not None and YHnew is not None: 
                X = {}
                Y = {}
                X['hf'] = np.concatenate((self.sample_XH, XHnew))
                Y['hf'] = np.concatenate((self.YH, YHnew))
                X['lf'] = np.concatenate((self.model_lf.sample_X, XLnew))
                Y['lf'] = np.concatenate((self.model_lf.Y, YLnew))
                self.train(X, Y)
            else:
                X = {}
                Y = {}
                X['hf'] = self.sample_XH
                Y['hf'] = self.YH
                X['lf'] = np.concatenate((self.model_lf.sample_X, XLnew))
                Y['lf'] = np.concatenate((self.model_lf.Y, YLnew))
                self.train(X, Y)
        else: 
            if XHnew is not None and YHnew is not None: 
                XH = np.concatenate((self.sample_XH, XHnew)) 
                YH = np.concatenate((self.YH, YHnew))
                self.train_hf(XH, YH)


class HierarchicalKriging(mf_model): 
    def __init__(self, bounds: list) -> None: 
        """_summary_

        Parameters
        ----------
        kernel_mode : str
            _description_
        num_dim : int
            _description_
        """
        self.num_dim = len(bounds)
        self.kernel = KRG(theta=np.zeros((1, self.num_dim)), bounds=[-2,3])
        self.model_lf = Kriging(bounds=bounds)
        self.bounds = np.array(bounds)
        self.low_bound = self.bounds[:, 0]
        self.high_bound = self.bounds[:, 1]

    def train(self, X: dict, Y: dict, n_iter: int = 500) -> None: 
        """_summary_

        Parameters
        ----------
        X : dict
            _description_
        Y : dict
            _description_
        """
        self.sample_XH = X['hf']
        self.XH = (self.sample_XH - self.low_bound) / (self.high_bound - self.low_bound)
        self.YH = Y['hf'].reshape(-1, 1) 
        self.model_lf.train(X['lf'], Y['lf'], n_iter=n_iter)
        self.F = self.model_lf.predict(self.sample_XH)
        self.optHyp(n_iter=n_iter)
        self.kernel.set_params(self.opt_param)
        
        self.K = self.kernel.K(self.XH, self.XH) 
        self.L = cholesky(self.K, lower=True)
        self.alpha = solve(self.L.T, solve(self.L, self.YH)) 
        self.beta = solve(self.L.T, solve(self.L, self.F)) 
        self.mu = (np.dot(self.F.T, self.alpha) / np.dot(self.F.T, self.beta)).squeeze()
        self.gamma = solve(self.L.T, solve(self.L, (self.YH - self.mu * self.F))) 
        self.sigma2 = np.dot((self.YH - self.mu * self.F).T, self.gamma).squeeze() / self.XH.shape[0]
        self.logp = (-self.XH.shape[0] * np.log(self.sigma2) - np.sum(np.log(np.diag(self.L)))).squeeze()

    def train_hf(self, XH: np.ndarray, YH: np.ndarray, n_iter: int = 500) -> None: 
        self.sample_XH = XH
        self.F = self.model_lf.predict(self.sample_XH)
        self.XH = (self.sample_XH - self.low_bound) / (self.high_bound - self.low_bound)
        self.YH = YH.reshape(-1, 1)
        self.optHyp(n_iter=n_iter)
        self.kernel.set_params(self.opt_param)
        
        self.K = self.kernel.K(self.XH, self.XH) 
        self.L = cholesky(self.K, lower=True)
        self.alpha = solve(self.L.T, solve(self.L, self.YH)) 
        self.beta = solve(self.L.T, solve(self.L, self.F)) 
        self.mu = (np.dot(self.F.T, self.alpha) / np.dot(self.F.T, self.beta)).squeeze()
        self.gamma = solve(self.L.T, solve(self.L, (self.YH - self.mu * self.F))) 
        self.sigma2 = np.dot((self.YH - self.mu * self.F).T, self.gamma).squeeze() / self.XH.shape[0]
        self.logp = (-self.XH.shape[0] * np.log(self.sigma2) - np.sum(np.log(np.diag(self.L)))).squeeze()

    def predict_lf(self, XLnew: np.ndarray, return_std: bool=False) -> np.ndarray: 

        return self.model_lf.predict(XLnew, return_std) 

    def predict(self, Xinput: np.ndarray, return_std: bool=False) -> np.ndarray: 
        XHnew = (Xinput - self.low_bound) / (self.high_bound - self.low_bound)
        XHnew = np.atleast_2d(XHnew) 
        pre_lf = self.predict_lf(Xinput)
        knew = self.kernel.K(XHnew, self.XH) 
        fmean = self.mu * pre_lf + np.dot(knew, self.gamma)
        if not return_std: 
            return fmean.reshape(-1, 1)
        else: 
            delta = solve(self.L.T, solve(self.L, knew.T))
            mse = self.sigma2 * (1 - np.diag(knew.dot(delta)) + \
                np.diag(np.dot((knew.dot(self.beta) - pre_lf), (knew.dot(self.beta) - pre_lf).T)) / self.F.T.dot(self.beta))
            return fmean.reshape(-1, 1), np.sqrt(np.maximum(mse, 0)).reshape(-1, 1)

    def optHyp(self, n_iter, grads=None): 
        options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
        bounds = (np.array(self.low_bound), np.array(self.high_bound))
        optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=self.num_dim, options=options, bounds=bounds)
        opt_fs, opt_param = optimizer.optimize(self._logLikelihood, verbose=False, iters=n_iter)
        self.opt_param = opt_param

    def _logLikelihood(self, params): 
        out = np.zeros(params.shape[0])
        for i in range(params.shape[0]):
            param = params[i, :]      
            K = self.kernel(self.XH, self.XH, param) 
            L = cholesky(K, lower=True)
            alpha = solve(L.T, solve(L, self.YH)) 
            beta = solve(L.T, solve(L, self.F))
            mu = (np.dot(self.F.T, alpha) / np.dot(self.F.T, beta)).squeeze()
            gamma = solve(L.T, solve(L, (self.YH - mu * self.F))) 
            sigma2 = np.dot((self.YH - mu * self.F).T, gamma).squeeze() / self.XH.shape[0]
            logp = (-self.XH.shape[0] * np.log(sigma2) - np.sum(np.log(np.diag(L))))
            out[i] = logp.squeeze()
        return (- out)      


class ScaledKriging(mf_model): 
    def __init__(self, bounds, rho_optimize=False, rho_bound=[1e-1, 1e1]): 
        """
        Kriging model with scaled function
        
        Parameters: 
        -----------------
        kernel_mode: Covariance function. 
        num_dim: Number of dimensionality

        Attributs: 
        -----------------


        Reference:
        [1] 
        """
        self.num_dim = len(bounds)
        self.kernel = KRG(theta=np.zeros((1, self.num_dim)))
        self.rho_optimize = rho_optimize 
        self.rho = 1.0
        self.model_lf = Kriging(bounds=bounds)
        self.model_disc = Kriging(bounds=bounds)

    def train(self, X: dict, Y: dict) -> None: 
        """
        Build the low-fidelity surrogate
        """ 
        self.sample_XH = X['hf']
        self.YH = Y['hf'].reshape(-1, 1)
        self.model_lf.train(X['lf'] , Y['lf']) 
        discrepancy = self._getDisc()
        self.model_disc.train(X['hf'], discrepancy) 

    def train_hf(self, XH, YH): 
        """
        Fit a multi-fidelity Gaussian Process Regressor
        
        Parameters: 
        -----------------
        XH: Traning samples for high fidelity
        YH: Corresponding target values for high fidelity

        """ 
        self.sample_XH = XH 
        self.YH = YH.reshape(-1, 1) 
        discrepancy = self._getDisc()
        self.model_disc.train(XH, discrepancy) 

    def predict_lf(self, Xnew, return_std=False): 
        """
        Predict at new samples and return mean and covariances for the multi-fidelity 
        Gaussian Process Regressor 

        Parameters: 
        -----------------
        Xnew: 
        return_std: 

        Returns: 
        -----------------

        """
        return self.model_lf.predict(Xnew, return_std)

    def predict(self, Xnew, return_std=False): 
        """
        Predict at new samples and return mean and covariances for the multi-fidelity 
        Gaussian Process Regressor 

        Parameters: 
        -----------------
        Xnew: 
        return_std: 

        Returns: 
        -----------------

        """
        if not return_std: 
            return self.model_disc.predict(Xnew) * self.rho + self.model_lf.predict(Xnew)
        else: 
            pre_lf, std_lf = self.model_lf.predict(Xnew, return_std) 
            pre_disc, std_disc = self.model_disc.predict(Xnew, return_std)
            mse =  self.rho ** 2 * std_lf ** 2 + std_disc ** 2
            return self.rho * pre_lf + pre_disc, np.sqrt(mse)
        
    def _getDisc(self): 

        return self.YH - self.rho * self.model_lf.predict(self.sample_XH)
