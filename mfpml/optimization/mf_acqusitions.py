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
    def initial_update(): 
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
        optimizer: any, 
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

    @staticmethod
    def eval(x, fmin: float, mf_surrogate: any, fidelity: str) -> np.ndarray: 
        """
        Evaluates selected acqusition function at certain fidelity. 
        
        Parameters: 
        -----------------
        fmin: float
            best observed function evaluation. 
        mf_surrogate: any 
            multi-fidelity surrogate instance.
        fidelity: str
            str indicating fidelity level.
        
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
        update_x = self.initial_update()
        res_hf = differential_evolution(self.eval, bounds=params['design_space'], 
                args=(params['fmin'], mf_surrogate, 'hf'), maxiter=2000, popsize=40)
        res_lf = differential_evolution(self.eval, bounds=params['design_space'], 
                args=(params['fmin'], mf_surrogate, 'lf'), maxiter=2000, popsize=40)
        if res_hf.fun <= res_lf.fun: 
            update_x['hf'] = np.atleast_2d(res_hf.x)
        else: 
            update_x['lf'] = np.atleast_2d(res_lf.x)
        return update_x




class vflcb(mfSingleObjAcf): 
    """
    Variable-fidelity Lower Confidence Bound function for single 
    objective bayesian optimization. 
    
        
    Parameters: 
    -------
    mode: str. Define the behaviour of the multi-fidelity acquisition strategy. 
        Currently supported values: 
        'VF-EI': Zhang, Y., Han, Z. H., & Zhang, K. S. (2018). Variable-fidelity expected 
                improvement method for efficient global optimization of expensive functions.
                Structural and Multidisciplinary Optimization, 58(4), 1431-1451.
        'VF-LCB': Jiang, P., Cheng, J., Zhou, Q., Shu, L., & Hu, J. (2019). Variable-fidelity 
                lower confidence bounding approach for engineering optimization problems with 
                expensive simulations. AIAA Journal, 57(12), 5416-5430.
    eps: Small floating value. 
    params: Extra parameters needed for certain acqusition functions. 
    """

    def __call__(self, x, surr_mf, cost, kappa=1.96, fidelity='high'):
        """
        Evaluate the acqusition function values

        Parameters: 
        -------
        surr_hf: instance 
            Multi-fidelity surrogate instance.
        cost: list
            Corresponding cost for 'high'(cost[0]) and 'low'(cost[1]) fidelity.
        kappa: float
            balance coefficient btw exploration and exploitation
        fidelity: str
            str indicating fidelity level.
        """
        cr = cost[0] / cost[1] 
        mean_hf, std_hf = surr_mf.predict(x, return_std=True) 
        _, std_lf = surr_mf.predict_lf(x, return_std=True) 

        if fidelity == 'high': 
            vflcb = mean_hf - kappa * std_hf 
        elif fidelity == 'low': 
            vflcb = mean_hf - kappa * std_lf * cr 
        else: 
            ValueError('Unknown fidelity input.')
        
        return vflcb 