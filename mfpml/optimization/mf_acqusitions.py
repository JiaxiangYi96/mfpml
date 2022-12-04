import numpy as np  
from scipy.stats import norm
from scipy.optimize import differential_evolution


class mfAcqusitionFunction: 
    """
    Base class for multi-fidelity acqusition functions. 
    
    """ 
    pass

class mfSingleObjAcf(mfAcqusitionFunction): 
    """
    Base class for multi-fidelity acqusition functions for Single Objective Opti. 
    """ 
    @staticmethod
    def _initial_update(): 
        update_x = {} 
        update_x['hf'] = None
        update_x['lf'] = None 
        return update_x


    pass


class vfei(mfSingleObjAcf): 
    """
    Variable-fidelity Expected Improvement method for single objective 
    bayesian optimization. Note that the method cooperates with hierarchical 
    Kriging model. 

    Reference 
    -------
    [1] Zhang, Y., Han, Z. H., & Zhang, K. S. (2018). Variable-fidelity expected 
        improvement method for efficient global optimization of expensive functions.
        Structural and Multidisciplinary Optimization, 58(4), 1431-1451.
    
    """ 
    def __init__(
        self, 
        optimizer: any = None, 
        constraint: bool = False) -> None: 
        """Initialize the multi-fidelity acqusition

        Parameters
        ----------
        optimizer : any
            optimizer instance
        constraint : bool, optional
            whether to use for constrained optimization
        """
        self.constraint = constraint
        self.optimizer = optimizer

    def eval(self, x, fmin: float, mf_surrogate: any, fidelity: str) -> np.ndarray: 
        """
        Evaluates selected acqusition function at certain fidelity
        
        Parameters: 
        -----------------
        fmin: float
            best observed function evaluation
        mf_surrogate: any 
            multi-fidelity surrogate instance
        fidelity: str
            str indicating fidelity level
        
        Returns
        -----------------
        np.ndarray
            Acqusition function value w.r.t corresponding fidelity level. 
        """
        pre, std = mf_surrogate.predict(x, return_std=True) 
        if fidelity == 'hf': 
            s = std
        elif fidelity == 'lf': 
            _, std_lf = mf_surrogate.predict_lf(x, return_std=True) 
            s = mf_surrogate.mu * std_lf
        else: 
            ValueError('Unknown fidelity input.')
        
        z = (fmin - pre) / std
        vfei = (fmin - pre) * norm.cdf(z) + std * norm.pdf(z) 
        vfei[s<np.finfo(float).eps] = 0.
        return (- vfei).ravel()
    
    def query(self, mf_surrogate: any, params: dict) -> dict: 
        """Query the vfei acqusition function

        Parameters
        ----------
        mf_surrogate : any
            multi-fidelity surrogate instance
        params : dict
            parameters of Bayesian Optimization

        Returns
        -------
        dict
            contains two values where 'hf' is the update points 
            for high-fidelity and 'lf' for low-fidelity
        """
        update_x = self._initial_update()
        if self.optimizer is None:
            res_hf = differential_evolution(self.eval, bounds=params['design_space'], 
                    args=(params['fmin'], mf_surrogate, 'hf'), maxiter=2000, popsize=40)
            res_lf = differential_evolution(self.eval, bounds=params['design_space'], 
                    args=(params['fmin'], mf_surrogate, 'lf'), maxiter=2000, popsize=40)
            opt_hf = res_hf.fun
            x_hf = res_hf.x
            opt_lf = res_lf.fun
            x_lf = res_lf.x
        else:
            pass
        if opt_hf <= opt_lf: 
            update_x['hf'] = np.atleast_2d(x_hf)
        else: 
            update_x['lf'] = np.atleast_2d(x_lf)
        return update_x




class vflcb(mfSingleObjAcf): 
    """
    Variable-fidelity Lower Confidence Bound function for single 
    objective bayesian optimization. 

    Reference 
    -------
    [1] Jiang, P., Cheng, J., Zhou, Q., Shu, L., & Hu, J. (2019). Variable-fidelity 
        lower confidence bounding approach for engineering optimization problems with 
        expensive simulations. AIAA Journal, 57(12), 5416-5430.
    """
    def __init__(
        self, 
        optimizer: any = None, 
        kappa: list = [1., 1.96], 
        constraint: bool = False) -> None:
        """Initialize the vflcb acqusition

        Parameters
        ----------
        optimizer : any, optional
            optimizer instance, by default 'L-BFGS-B'
        kappa : list, optional
            balance factors for exploitation and exploration respectively
            , by default [1., 1.96]
        constraint : bool, optional
            use for constrained problem or not, by default False
        """
        super().__init__()
        self.optimizer = optimizer
        self.kappa = kappa
        self.constraint = constraint
    
    def eval(
        self, 
        x: np.ndarray, 
        mf_surrogate: any, 
        cost_ratio: float, 
        fidelity: str) -> np.ndarray:
        """Evaluate vflcb function values at certain fidelity

        Parameters
        ----------
        x : np.ndarray
            point to evaluate
        mf_surrogate : any
            multi-fidelity surrogate model instance
        cost_ratio : float
            ratio of high-fidelity cost to low-fidelity cost
        fidelity : str
            str indicating fidelity level

        Returns
        -------
        np.ndarray
            acqusition function values
        """
        cr = cost_ratio 
        mean_hf, std_hf = mf_surrogate.predict(x, return_std=True) 
        _, std_lf = mf_surrogate.predict_lf(x, return_std=True) 
        if fidelity == 'hf': 
            std = std_hf 
        elif fidelity == 'lf': 
            std = std_lf * cr 
        else:
            ValueError('Unknown fidelity input.')
        vflcb = self.kappa[0] * mean_hf - self.kappa[1] * std
        return vflcb.ravel()

    def query(self, mf_surrogate: any, params: dict) -> dict: 

        update_x = self._initial_update()
        if self.optimizer is None:
            res_hf = differential_evolution(self.eval, bounds=params['design_space'], 
                    args=(mf_surrogate, params['cr'], 'hf'), maxiter=2000, popsize=40)
            res_lf = differential_evolution(self.eval, bounds=params['design_space'], 
                    args=(mf_surrogate, params['cr'], 'lf'), maxiter=2000, popsize=40)
            opt_hf = res_hf.fun
            x_hf = res_hf.x
            opt_lf = res_lf.fun
            x_lf = res_lf.x
        else:
            pass
        if opt_hf <= opt_lf: 
            update_x['hf'] = np.atleast_2d(x_hf)
        else: 
            update_x['lf'] = np.atleast_2d(x_lf)
        return update_x
            