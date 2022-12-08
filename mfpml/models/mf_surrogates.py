
import numpy as np 
from collections import OrderedDict
from scipy.linalg import cholesky, solve
from scipy.optimize import minimize

from mfpml.models.corrfunc import KRG
from .kriging import Kriging
from ..base.model import model


class mf_model(model): 

    def __init__(
        self
        ) -> None:
        pass

    def train(self, X: dict, Y: dict) -> None: 
        """Train the hierarchical Kriging model

        Parameters
        ----------
        X : dict
            dict with two keys, 'hf' contains np.ndarray of 
            high-fidelity sample points and 'lf' contains 
            low-fidelity
        Y : dict
            dict with two keys, 'hf' contains high-fidelity
            responses and 'lf' contains low-fidelity ones
        """
        #train the low-fidelity model
        self._train_lf(X['lf'], Y['lf'])
        #train high-fidelity model
        self._train_hf(X['hf'], Y['hf'])

    def plotMfmodels_1D(self):
        pass

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
        XHnew = Xnew['hf']
        YHnew = Ynew['hf']
        XLnew = Xnew['lf']
        YLnew = Ynew['lf']
        if XLnew is not None and YLnew is not None: 
            if XHnew is not None and YHnew is not None: 
                X = {}
                Y = {}
                X['hf'] = np.concatenate((self.sample_XH, XHnew))
                Y['hf'] = np.concatenate((self.sample_YH, YHnew))
                X['lf'] = np.concatenate((self.sample_XL, XLnew))
                Y['lf'] = np.concatenate((self.sample_YL, YLnew))
                self.train(X, Y)
            else:
                X = {}
                Y = {}
                X['hf'] = self.sample_XH
                Y['hf'] = self.sample_YH
                X['lf'] = np.concatenate((self.sample_XL, XLnew))
                Y['lf'] = np.concatenate((self.sample_YL, YLnew))
                self.train(X, Y)
        else: 
            if XHnew is not None and YHnew is not None: 
                XH = np.concatenate((self.sample_XH, XHnew)) 
                YH = np.concatenate((self.sample_YH, YHnew))
                self._train_hf(XH, YH)

    def _update_optimizer_hf(self, optimizer: any) -> None: 
        """Change the optimizer for high-fidelity hyperparameters

        Parameters
        ----------
        optimizer : any
            instance of optimizer
        """
        self.optimizer = optimizer

    def _update_optimizer_lf(self, optimizer: any) -> None: 
        """Change the optimizer for low-fidelity hyperparameters

        Parameters
        ----------
        optimizer : any
            instance of optimizer
        """
        self.lf_model._update_optimizer(optimizer)

    @property
    def _get_lf_model(self) -> any: 
        """Get the low-fidelity model

        Returns
        -------
        any
            low-fidelity model instance
        """
        return self.lf_model

    @property
    def _num_XH(self) -> int:
        """Return the number of high-fidelity samples

        Returns
        -------
        int
            #high-fidelity samples
        """
        return self.sample_XH.shape[0]

    @property
    def _num_XL(self) -> int: 
        """Return the number of low-fidelity samples

        Returns
        -------
        int
            #low-fidelity samples
        """
        return self.lf_model._num_X

    @property
    def _get_sample_hf(self) -> np.ndarray:
        """Return samples of high-fidelity

        Returns
        -------
        np.ndarray
            high-fidelity samples
        """
        return self.sample_XH

    @property
    def _get_sample_lf(self) -> np.ndarray:
        """Return samples of high-fidelity

        Returns
        -------
        np.ndarray
            high-fidelity samples
        """
        return self.sample_XL

    def _train_lf(self, X: np.ndarray, Y: np.ndarray) -> None: 
        """Train the low-fidelity model

        Parameters
        ----------
        X : np.ndarray
            low-fidelity samples
        Y : np.ndarray
            low-fidelity responses
        """
        self.lf_model.train(X, Y)
        self.sample_XL = X
        self.XL = self.normalize_input(X)
        self.sample_YL = Y
        
    def predict_lf(self, XLnew: np.ndarray, return_std: bool=False) -> np.ndarray: 
        """Predict low-fidelity responses

        Parameters
        ----------
        XLnew : np.ndarray
            array of low-fidelity to be predicted
        return_std : bool, optional
            whether to return the standard deviation, by default False

        Returns
        -------
        np.ndarray
            prediction of low-fidelity
        """
        return self.lf_model.predict(XLnew, return_std) 

    def _eval_corr(self, X: np.ndarray, Xprime: np.ndarray, fidelity='hf') -> np.ndarray:
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
        if fidelity == 'hf':
            return self.kernel.K(X, Xprime)
        elif fidelity == 'lf':
            return self.lf_model.kernel.K(X, Xprime)
        else:
            ValueError('Unknown fidelity input.')

    def normalize_input(self, X: np.ndarray) -> np.ndarray: 
        """Normalize samples to range [0, 1]

        Parameters
        ----------
        X : np.ndarray
            samples to scale
        Returns
        -------
        np.ndarray
            normalized samples
        """
        return (X - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])



class HierarchicalKriging(mf_model): 
    def __init__(
        self, 
        design_space: np.ndarray,
        optimizer: any = None, 
        kernel_bound: list = [-4., 2.]) -> None: 
        """Initialize hierarchical Kriging model

        Parameters
        ----------
        design_space: np.ndarray
            array of shape=((num_dim,2)) where each row describes
            the bound for the variable on specfic dimension.
        optimizer: any, optional
            instance of the optimizer used to optimize the hyperparameters
            with the use style optimizer.run_optimizer(objective function, 
            number of dimension, design space of variables), if not assigned,
            the 'L-BFGS-B' method in scipy is used.
        kernel_bound: list, optional
            log bound of the kernel for hierarchical Kriging model, by 
            default [-2, 3].
        """
        self.bounds = design_space
        self.optimizer = optimizer
        self.num_dim = design_space.shape[0]
        self.kernel = KRG(theta=np.zeros( self.num_dim), bounds=kernel_bound)

        self.lf_model = Kriging(design_space=design_space, optimizer=optimizer)

    def _train_hf(self, XH: np.ndarray, YH: np.ndarray) -> None:
        """Train the high-fidelity model

        Parameters
        ----------
        XH : np.ndarray
            array of high-fidelity samples
        YH : np.ndarray
            array of high-fidelity responses
        """
        XH, YH = self.drop_neighbor(XH, YH)
        self.sample_XH = XH
        self.XH = self.normalize_input(self.sample_XH)
        self.sample_YH = YH.reshape(-1, 1)
        #prediction of low-fidelity at high-fidelity locations
        self.F = self.predict_lf(self.sample_XH)
        #optimize the hyperparameters
        self._optHyp()
        self.kernel.set_params(self.opt_param)
        self._update_parameters()

    def predict(self, Xnew: np.ndarray, return_std: bool=False) -> np.ndarray: 
        """Predict high-fidelity responses

        Parameters
        ----------
        Xnew : np.ndarray
            array of high-fidelity to be predicted
        return_std : bool, optional
            whether to return std values, by default False

        Returns
        -------
        np.ndarray
            prediction of high-fidelity
        """
        pre_lf = self.predict_lf(Xnew)
        XHnew = np.atleast_2d(self.normalize_input(Xnew)) 
        knew = self.kernel.K(XHnew, self.XH) 
        fmean = self.mu * pre_lf + np.dot(knew, self.gamma)
        if not return_std: 
            return fmean.reshape(-1, 1)
        else: 
            delta = solve(self.L.T, solve(self.L, knew.T))
            mse = self.sigma2 * (1 - np.diag(knew.dot(delta)) + \
                np.diag(np.dot((knew.dot(self.beta) - pre_lf), (knew.dot(self.beta) - pre_lf).T)) / self.F.T.dot(self.beta))
            return fmean.reshape(-1, 1), np.sqrt(np.maximum(mse, 0)).reshape(-1, 1)

    def _optHyp(self, grads=None):
        """Optimize the hyperparameters

        Parameters
        ----------
        grads : bool, optional
            whether to use gradients, by default None
        """
        if self.optimizer is None:
            n_trials = 6
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

    def _logLikelihood(self, params): 
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
            K = self.kernel(self.XH, self.XH, param) 
            L = cholesky(K, lower=True)
            alpha = solve(L.T, solve(L, self.sample_YH)) 
            beta = solve(L.T, solve(L, self.F))
            mu = (np.dot(self.F.T, alpha) / np.dot(self.F.T, beta)).squeeze()
            gamma = solve(L.T, solve(L, (self.sample_YH - mu * self.F))) 
            sigma2 = (np.dot((self.sample_YH - mu * self.F).T, gamma).squeeze() / self.XH.shape[0]).squeeze()
            logp = (-self.XH.shape[0] * np.log(sigma2) - 2 * np.sum(np.log(np.diag(L))))
            out[i] = logp.squeeze()
        return (- out)      

    def _update_parameters(self) -> None: 
        """Update parameters of the model
        """
        self.K = self.kernel.K(self.XH, self.XH) 
        self.L = cholesky(self.K, lower=True)
        self.alpha = solve(self.L.T, solve(self.L, self.sample_YH)) 
        self.beta = solve(self.L.T, solve(self.L, self.F)) 
        self.mu = (np.dot(self.F.T, self.alpha) / np.dot(self.F.T, self.beta)).item()
        self.gamma = solve(self.L.T, solve(self.L, (self.sample_YH - self.mu * self.F))) 
        self.sigma2 = (np.dot((self.sample_YH - self.mu * self.F).T, self.gamma).squeeze() / self.XH.shape[0]).item()
        self.logp = (-self.XH.shape[0] * np.log(self.sigma2) - np.sum(np.log(np.diag(self.L)))).item()


class ScaledKriging(mf_model): 
    def __init__(
        self, 
        design_space: np.ndarray,
        lf_model: any = None, 
        disc_model: any = None, 
        optimizer: any = None, 
        kernel_bound: list = [-4., 3.], 
        rho_optimize: bool = False, 
        rho_method: str = 'error', 
        rho_bound: list = [1e-2, 1e1], 
        rho_optimizer: any = None) -> None: 
        """Multi-fidelity Kriging model with scaled function

        Parameters
        ----------
        design_space : np.ndarray
            array of shape=((num_dim,2)) where each row describes
            the bound for the variable on specfic dimension
        lf_model: any
            instance of low-fidelity model, the model should have the method: 
            train(x: np.ndarray, y: np.ndarray), 
            predict(x: np.ndarray, return_std: bool)
        disc_model: any, optional
            instance of discrepancy model, the model should have the method: 
            train(x: np.ndarray, y: np.ndarray), 
            predict(x: np.ndarray, return_std: bool). Default Kriging 
        optimizer: any, optional
            instance of the optimizer used to optimize the hyperparameters
            with the use style optimizer.run_optimizer(objective function, 
            number of dimension, design space of variables), if not assigned,
            the 'L-BFGS-B' method in scipy is used
        kernel_bound: list, optional
            log bound of the kernel for discrepancy Kriging model, by 
            default [-3, 2]
        rho_optimize : bool, optional
            whether to optimize the scale factor, if not the scale 
            factor is 1, by default False
        rho_method : str, optional
            method to choose rho, can choose from ['error', 'bumpiness']
        rho_bound : list, optional
            bound for the factor rho if optimizing rho, by default [1e-1, 1e1]
        rho_optimizer : any, optional
            optimizer for the parameter rho, by default 'L-BFGS-B'
        """
        self.bounds = design_space
        self.lf_model = lf_model
        self.optimizer = optimizer
        self.rho_optimize = rho_optimize 
        self.rho = 1.0
        self.rho_bound = rho_bound
        self.rho_method = rho_method
        self.rho_optimizer = rho_optimizer

        self.num_dim = design_space.shape[0]
        self.kernel = KRG(theta=np.zeros(self.num_dim), bounds=kernel_bound)
        if lf_model is None: 
            self.lf_model = Kriging(design_space=design_space, optimizer=optimizer)
        else: 
            self.lf_model = lf_model
        if disc_model is None:
            self.disc_model = Kriging(design_space=design_space, optimizer=optimizer)
        else: 
            self.disc_model = disc_model
        if optimizer is not None: 
            self._update_optimizer_hf(optimizer)
            self._update_optimizer_lf(optimizer)

    def _train_hf(self, XH: np.ndarray, YH: np.ndarray) -> None: 
        """Train the discrepancy model in mf models
        
        Parameters: 
        -----------------
        XH : np.ndarray
            array of high-fidelity samples
        YH : np.ndarray
            array of high-fidelity responses
        """ 
        XH, YH = self.drop_neighbor(XH, YH)
        self.sample_XH = XH 
        self.sample_YH = YH.reshape(-1, 1)
        if self.rho_optimize: 
            self._rho_optimize()
        self.disc_model.train(self.sample_XH, self._getDisc())

    def predict(self, Xnew: np.ndarray, return_std: bool = False) -> np.ndarray: 
        """Predict high-fidelity responses

        Parameters
        ----------
        Xnew : np.ndarray
            array of high-fidelity to be predicted
        return_std : bool, optional
            whether to return std values, by default False

        Returns
        -------
        np.ndarray
            prediction of high-fidelity
        """
        if not return_std: 
            return self.disc_model.predict(Xnew) * self.rho + self.lf_model.predict(Xnew)
        else: 
            pre_lf, std_lf = self.lf_model.predict(Xnew, return_std) 
            pre_disc, std_disc = self.disc_model.predict(Xnew, return_std)
            mse =  self.rho ** 2 * std_lf ** 2 + std_disc ** 2
            return self.rho * pre_lf + pre_disc, np.sqrt(mse)
        
    def _getDisc(self) -> np.ndarray: 
        """Compute the discrepancy between low-fidelity prediction 
        at high-fidelity samples and high-fidelity responses

        Returns
        -------
        np.ndarray
            discrepancy
        """
        return self.sample_YH - self.rho * self.predict_lf(self.sample_XH)

    def _update_optimizer_hf(self, optimizer: any) -> None: 
        """Change the optimizer for high-fidelity hyperparameters

        Parameters
        ----------
        optimizer : any
            instance of optimizer
        """
        self.disc_model.optimizer = optimizer

    def _rho_optimize(self) -> None: 
        """Optimize the rho value
        """
        if self.rho_optimizer is None:
            if self.rho_method == 'error':
                x0 = np.random.uniform(self.rho_bound[0], self.rho_bound[1], 1)
                optRes = minimize(self._eval_error, x0=x0, method='L-BFGS-B',
                    bounds=np.array([self.rho_bound]))
            elif self.rho_method == 'bumpiness':
                x0 = np.random.uniform(self.rho_bound[0], self.rho_bound[1], 1)
                optRes = minimize(self._eval_bumpiness, x0=x0, method='L-BFGS-B',
                    bounds=np.array([self.rho_bound]))
            else:
                pass
            self.rho = optRes.x
        else:
            if self.rho_method == 'error':
                optRes = self.rho_optimizer.run_optimizer(self._eval_error, 
                    num_dim=1, design_space=np.array([self.rho_bound]))
            elif self.rho_method == 'bumpiness':
                optRes = self.rho_optimizer.run_optimizer(self._eval_bumpiness, 
                    num_dim=1, design_space=np.array([self.rho_bound]))
            else:
                pass
            self.rho = optRes['best_x']
   
    def _eval_error(self, rho: np.ndarray) -> np.ndarray:
        """Evaluate the summation of squared error for high-fidelity samples

        Parameters
        ----------
        rho : np.ndarray
            array of rho

        Returns
        -------
        np.ndarray
            sum of error
        """
        rho = np.tile(rho.reshape(-1, 1), (1, self._num_XH))
        error = (rho * self.predict_lf(self.sample_XH).ravel() - self.sample_YH.ravel())
        sum_error2 = np.sum(error ** 2, axis=1)
        return sum_error2

    def _eval_bumpiness(self, rho: np.ndarray) -> np.ndarray:
        """Evaluate the bumpiness

        Parameters
        ----------
        rho : np.ndarray
            array of rho

        Returns
        -------
        np.ndarray
            measure of bumpiness
        """
        rho = rho.reshape(-1, 1)
        out = np.zeros(rho.shape[0])
        for i in range(rho.shape[0]):
            sum_error2 = self._eval_error(rho[i, :])
            self.disc_model.train(self.sample_XH, self.sample_YH - rho[i, :] * self.predict_lf(self.sample_XH))
            theta = self.disc_model.kernel._get_param
            out[i] = sum_error2 * np.linalg.norm(theta)
        return out


    
class CoKriging(mf_model): 
    def __init__(
        self, 
        design_space: np.ndarray,
        optimizer: any = None, 
        kernel_bound: list = [-4., 3.], 
        rho_bound: list = [1e-2, 1e2]) -> None:
        """Co-Kriging model

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
            log bound of the kernel for difference Gaussian process model, 
            by default [-4, 3]
        rho_bound : list, optional
            bound for scale factor, by default [1e-2, 1e2]
        """
        super().__init__()
        self.bounds = design_space
        self.optimizer = optimizer
        self.num_dim = design_space.shape[0]
        self.rho_bound = rho_bound
        #initialize kernel and low-fidelity model
        self.kernel = KRG(theta=np.zeros(self.num_dim), bounds=kernel_bound)
        self.lf_model = Kriging(design_space=design_space, optimizer=optimizer)

    def _train_hf(self, XH: np.ndarray, YH: np.ndarray) -> None:
        """Train the high-fidelity model

        Parameters
        ----------
        XH : np.ndarray
            array of high-fidelity samples
        YH : np.ndarray
            array of high-fidelity responses
        """
        XH, YH = self.drop_neighbor(XH, YH)
        self.sample_XH = XH
        self.XH = self.normalize_input(self.sample_XH)
        self.sample_YH = YH.reshape(-1, 1)
        #prediction of low-fidelity at high-fidelity locations
        self.F = self.predict_lf(self.sample_XH)
        #optimize the hyperparameters
        self._optHyp()
        self.rho = self.opt_param[0]
        self.kernel.set_params(self.opt_param[1:])
        self._update_parameters()

    def predict(self, Xnew: np.ndarray, return_std: bool = False) -> np.ndarray:
        """Predict high-fidelity responses

        Parameters
        ----------
        Xnew : np.ndarray
            array of high-fidelity to be predicted
        return_std : bool, optional
            whether to return std values, by default False

        Returns
        -------
        np.ndarray
            prediction of high-fidelity
        """
        Xnew = np.atleast_2d(self.normalize_input(Xnew))
        c = np.concatenate(
            (self.rho*self.lf_model.sigma2*self.lf_model.kernel.K(self.XL, Xnew), 
            self.rho**2*self.lf_model.sigma2*self.lf_model.kernel.K(self.XH, Xnew)+self.sigma2*self.kernel.K(self.XH, Xnew)),
            axis=0)
        fmean = self.mu + c.T.dot(solve(self.LC.T, solve(self.LC, (self.y - self.mu))))
        if not return_std: 
            return fmean.reshape(-1, 1)
        else:
            s2 = self.rho ** 2 * self.lf_model.sigma2 + self.sigma2\
                - c.T.dot(solve(self.LC.T, solve(self.LC, c)))
            std = np.sqrt(np.maximum(np.diag(s2), 0))
            return fmean.reshape(-1, 1), std.reshape(-1, 1)

    def _optHyp(self) -> None:
        """Optimize the hyperparameters
        """
        bounds = np.concatenate((np.array([self.rho_bound]), 
                                self.kernel._get_bounds), axis=0)
        if self.optimizer is None:
            n_trials = 6
            opt_fs = float('inf')
            for trial in range(n_trials): 
                x0 = np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0])
                optRes = minimize(self._logLikelihood, x0=x0, method='L-BFGS-B',
                    bounds=bounds)
                if optRes.fun < opt_fs:
                    opt_param = optRes.x
                    opt_fs = optRes.fun
        else:
            optRes = self.optimizer.run_optimizer(self._logLikelihood, 
                num_dim=bounds.shape[0], design_space=bounds)
            opt_param = optRes['best_x']
        self.opt_param = opt_param.squeeze()

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
            rho = params[i, 0]
            theta = params[i, 1:]
            #correlation matrix R
            K = self.kernel(self.XH, self.XH, theta)
            L = cholesky(K, lower=True)
            #R^(-1)Y
            d = self.sample_YH - rho * self.F
            alpha = solve(L.T, solve(L, d))
            one = np.ones((self.XH.shape[0], 1))
            #R^(-1)1
            beta = solve(L.T, solve(L, one))
            #1R^(-1)Y / 1R^(-1)vector(1)
            mu = (np.dot(one.T, alpha) / np.dot(one.T, beta)).squeeze()
            gamma = solve(L.T, solve(L, (d - mu))) 
            sigma2 = np.dot((d - mu).T, gamma) / self.XH.shape[0]
            logp = -.5 * self.XH.shape[0] * np.log(sigma2) - np.sum(np.log(np.diag(L)))
            out[i] = logp.ravel()
        return (- out)

    def _update_parameters(self) -> None:
        """Update parameters of the model
        """
        #correlation matrix R
        K = self.kernel.K(self.XH, self.XH)
        L = cholesky(K, lower=True)
        #R^(-1)Y
        self.d = self.sample_YH - self.rho * self.F
        alpha = solve(L.T, solve(L, self.d))
        one = np.ones((self.XH.shape[0], 1))
        #R^(-1)1
        beta = solve(L.T, solve(L, one))
        #1R^(-1)Y / 1R^(-1)vector(1)
        self.mu_d = (np.dot(one.T, alpha) / np.dot(one.T, beta)).item()
        gamma = solve(L.T, solve(L, (self.d - self.mu_d))) 
        self.sigma2 = (np.dot((self.d - self.mu_d).T, gamma) / self.XH.shape[0]).item()
        self.logp = (-.5 * self.XH.shape[0] * np.log(self.sigma2) - np.sum(np.log(np.diag(L)))).item()
        #cov matrix for Co-Kriging
        self.C = np.concatenate(
                (np.concatenate((self.lf_model.sigma2*self.lf_model.K, self.rho*self.lf_model.sigma2*self.lf_model.kernel.K(self.XL,self.XH)), axis=1), 
                np.concatenate((self.rho*self.lf_model.sigma2*self.lf_model.kernel.K(self.XH,self.XL), self.rho**2*self.lf_model.sigma2*self.lf_model.kernel.K(self.XH, self.XH)+self.sigma2*K), axis=1)), 
                axis=0)
        self.y = np.concatenate((self.sample_YL, self.sample_YH), axis=0)
        self.LC = cholesky(self.C, lower=True)
        oneC = np.ones((self.C.shape[0], 1))
        self.mu = oneC.T.dot(solve(self.LC.T, solve(self.LC, self.y))) / oneC.T.dot(solve(self.LC.T, solve(self.LC, oneC)))
