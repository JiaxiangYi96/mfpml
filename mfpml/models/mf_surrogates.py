
import numpy as np 
from collections import OrderedDict
from scipy.linalg import cholesky, solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from mfpml.models.kernel import KRG, RBF


class Kriging: 
    """Kriging model 


    """
    def __init__(self, kernel_mode, n_dim, mprior=0): 
        """_summary_

        Parameters
        ----------
        kernel_mode : _type_
            _description_
        n_dim : _type_
            _description_
        mprior : int, optional
            _description_, by default 0
        """
        if kernel_mode == 'KRG': 
            self.kernel = KRG(np.zeros(n_dim)) 
        elif kernel_mode == 'RBF':
            self.kernel = RBF(l=0., sigmaf=0.)
        
        self.n_dim = n_dim
        self.mprior = mprior

    def getkernelparams(self): 
        pass 

    def train(self, X, Y): 
        """
        """
        self.X = X 
        self.Y = Y.reshape(-1, 1) 
        self.optHyp(param_key=self.kernel.parameters, param_bounds=self.kernel.bounds)

    def optHyp(self, param_key, param_bounds, grads=None, n_trials=9): 

        opt_fs = float('inf')
        for trial in range(n_trials): 

            for param, bound in zip(param_key, param_bounds): 
                x0 = np.random.uniform(bound[0], bound[1], self.n_dim)

            if grads is None: 
                optout = minimize(self._logLikelihood, x0=x0, method='L-BFGS-B', \
                    args=(param_key), bounds=param_bounds)
            else: 
                pass
            if optout.fun < opt_fs: 
                opt_param = optout.x 
                opt_fs = optout.fun
        
        self._logLikelihood(opt_param, param_key) 

    def predict(self, Xnew, return_std=False): 
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
        Xnew = np.atleast_2d(Xnew) 
        knew = self.kernel(Xnew, self.X) 
        fmean = self.mu + np.dot(knew, self.gamma)
        if not return_std: 
            return fmean.reshape(-1, 1)
        else:
            one = np.ones((self.X.shape[0], 1))
            delta = solve(self.L.T, solve(self.L, knew.T))
            mse = self.sigma2 * (1 - np.diag(knew.dot(delta)) + \
                (1 - one.T.dot(delta)) ** 2 / one.T.dot(self.beta))
            return fmean.reshape(-1, 1), np.sqrt(np.maximum(mse, 0)).reshape(-1, 1)

    def _logLikelihood(self, param_vector, param_key): 

        k_params = {}
        for k, v in zip(param_key, param_vector): 
            k_params[k] = v 
        self.kernel.set_params(k_params) 
        self.K = self.kernel(self.X, self.X) 
        self.L = cholesky(self.K, lower=True)
        one = np.ones((self.X.shape[0], 1))
        self.alpha = solve(self.L.T, solve(self.L, self.Y)) 
        self.beta = solve(self.L.T, solve(self.L, one)) 
        self.mu = (np.dot(one.T, self.alpha) / np.dot(one.T, self.beta)).squeeze()
        self.gamma = solve(self.L.T, solve(self.L, (self.Y - self.mu))) 
        self.sigma2 = np.dot((self.Y - self.mu).T, self.gamma) / self.X.shape[0]
        self.logp = (-.5 * self.X.shape[0] * np.log(self.sigma2) - np.sum(np.log(np.abs(np.diag(self.L))))).squeeze()
        # self.logp = -.5 * np.dot(self.Y, self.alpha) - np.sum(np.log(np.diag(self.L)))\
        #     - self.X.shape[0] / 2 * np.log(2 * np.pi) 
        return (- self.logp)      


class mf_model: 

    def update(self, XHnew=None, YHnew=None, XLnew=None, YLnew=None): 

        if XLnew and YLnew is not None: 
            XL = np.concatenate((self.model_lf.X, XLnew), axis=0) 
            YL = np.concatenate((self.model_lf.Y, YLnew), axis=0) 
            self.model_lf.train(XL, YL) 
        if XHnew and YHnew is not None: 
            XH = np.concatenate((self.XH, XHnew), axis=0) 
            YH = np.concatenate((self.YH, YHnew), axis=0) 
            self.train_hf(XH, YH)


class HierarchicalKriging(mf_model): 
    def __init__(self, kernel_mode, n_dim): 
        """
        Hierarchical Kriging model class. 
        
        Parameters
        -------
        
        """
        if kernel_mode == 'KRG': 
            self.kernel = KRG(np.zeros(n_dim), bounds=[[-2,3]]) 
        elif kernel_mode == 'RBF':
            pass

        self.n_dim = n_dim 
        self.model_lf = Kriging(kernel_mode, n_dim)

    def train(self, XH, YH, XL, YL): 

        self.XH = XH 
        self.YH = YH.reshape(-1, 1) 
        self.model_lf.train(XL, YL) 
        self.optHyp(param_key=self.kernel.parameters, param_bounds=self.kernel.bounds)

    def train_hf(self, XH, YH): 

        self.XH = XH 
        self.YH = YH.reshape(-1, 1)
        self.optHyp(param_key=self.kernel.parameters, param_bounds=self.kernel.bounds)

    def predict_lf(self, XLnew, return_std=False): 

        return self.model_lf.predict(XLnew, return_std) 

    def predict(self, XHnew, return_std=False): 

        XHnew = np.atleast_2d(XHnew) 
        knew = self.kernel(XHnew, self.XH) 
        fmean = self.mu * self.predict_lf(XHnew).reshape(-1, 1) + np.dot(knew, self.gamma)
        if not return_std: 
            return fmean.reshape(-1, 1)
        else: 
            delta = solve(self.L.T, solve(self.L, knew.T))
            mse = self.sigma2 * (1 - np.diag(knew.dot(delta)).reshape(-1, 1) + \
                (knew.dot(self.beta) - self.model_lf.predict(XHnew).reshape(-1, 1)) ** 2 / self.F.T.dot(self.beta))
            return fmean.reshape(-1, 1), np.sqrt(np.maximum(mse, 0)).reshape(-1, 1)

    def optHyp(self, param_key, param_bounds, grads=None, n_trials=9): 

        opt_fs = float('inf')
        for trial in range(n_trials): 

            for param, bound in zip(param_key, param_bounds): 
                x0 = np.random.uniform(bound[0], bound[1], self.n_dim)

            if grads is None: 
                optout = minimize(self._logLikelihood, x0=x0, method='L-BFGS-B', \
                    args=(param_key), bounds=param_bounds)
            else: 
                pass
            if optout.fun < opt_fs: 
                opt_param = optout.x 
                opt_fs = optout.fun

        self._logLikelihood(opt_param, param_key) 

    def _logLikelihood(self, param_vector, param_key): 

        k_params = {}
        for k, v in zip(param_key, param_vector): 
            k_params[k] = v 
        self.kernel.set_params(k_params) 
        self.K = self.kernel(self.XH, self.XH) 
        self.L = cholesky(self.K, lower=True)
        self.F = self.model_lf.predict(self.XH).reshape(-1, 1)
        self.alpha = solve(self.L.T, solve(self.L, self.YH)) 
        self.beta = solve(self.L.T, solve(self.L, self.F)) 
        self.mu = (np.dot(self.F.T, self.alpha) / np.dot(self.F.T, self.beta)).squeeze()
        self.gamma = solve(self.L.T, solve(self.L, (self.YH - self.mu * self.F))) 
        self.sigma2 = np.dot((self.YH - self.mu * self.F).T, self.gamma) / self.XH.shape[0]
        self.logp = (-self.XH.shape[0] * np.log(self.sigma2) - np.sum(np.log(np.abs(np.diag(self.L))))).squeeze()

        return (- self.logp)      


class ScaledKriging(mf_model): 
    def __init__(self, kernel_mode, n_dim, rho_optimize=False, rho_bound=[1e-1, 1e1]): 
        """
        Kriging model with scaled function
        
        Parameters: 
        -----------------
        kernel_mode: Covariance function. 
        n_dim: Number of dimensionality

        Attributs: 
        -----------------


        Reference:
        [1] 
        """
        if kernel_mode == 'KRG': 
            self.kernel = KRG(np.zeros(n_dim)) 
        elif kernel_mode == 'RBF': 
            pass
        self.n_dim = n_dim 
        self.rho_optimize = rho_optimize 
        self.rho = 1.0
        self.model_lf = Kriging(kernel_mode, n_dim)
        self.model_disc = Kriging(kernel_mode, n_dim)

    def train(self, XH, YH, XL, YL): 
        """
        Build the low-fidelity surrogate
        """ 
        self.XH = XH 
        self.YH = YH.reshape(-1, 1)  
        self.model_lf.train(XL, YL) 
        self._getDisc()
        self.model_disc.train(XH, self.disc) 

    def train_hf(self, XH, YH): 
        """
        Fit a multi-fidelity Gaussian Process Regressor
        
        Parameters: 
        -----------------
        XH: Traning samples for high fidelity
        YH: Corresponding target values for high fidelity

        """ 
        self.XH = XH 
        self.YH = YH.reshape(-1, 1) 
        self._getDisc()
        self.model_disc.train(XH, self.disc) 

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
            return self.rho * pre_lf + pre_disc, np.sqrt((self.rho * std_lf) ** 2 + std_disc)
        
    def _getDisc(self): 

        a=self.model_lf.predict(self.XH)
        self.disc = self.YH - self.rho * self.model_lf.predict(self.XH)
        a=self.model_lf.predict(self.XH)
