import numpy as np  
from scipy.stats import norm


class MF_Acquisitions: 
    def __init__(self, mode, eps=1e-06, **params): 
        """
        Multi-fidelity acquisition for Bayesian Optimization. 
        
        Parameters: 
        -----------------
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
        self.eps = eps 
        self.params = params 

        mode_dict = {
            'VF-EI': self._vfei,
            'VF-LCB': self._vflcb
        }
        self.acf = mode_dict[mode]

    def eval(self, x, fmin, surr_mf, problemFunc, fidelity='high'): 
        """
        Evaluates selected acqusition function at certain fidelity. 
        
        Parameters: 
        -----------------
        fmin: float
            Best observed function evaluation. 
        surr_hf: instance 
            Multi-fidelity surrogate instance.
        problem: instance
            functions instance
        fidelity: str
            str indicating fidelity level.
        
        Returns
        -----------------
        float 
            Acqusition function value w.r.t corresponding fidelity level. 
        """
        
        return self.acf(x, fmin, surr_mf, fidelity, **self.params)

    @staticmethod
    def _vfei(self, x, fmin, surr_mf, fidelity): 
        """
        Variable-fidelity Expected Improvement
        
        """
        mean_hf, std_hf = surr_mf.predict(x, return_std=True) 
        if fidelity == 'high': 
            std = std_hf 
        elif fidelity == 'low': 
            _, std_lf = surr_mf.predict_lf(x, return_std=True) 
            beta0 = surr_mf.beta 
            std = beta0 * std_lf 
        else: 
            print("Wrong fidelity str input! \n") 
        
        z = (fmin - mean_hf) / std
        vfei = (fmin - mean_hf) * norm.cdf(z) + std * norm.pdf(z) 
        return -vfei 

    @staticmethod 
    def _vflcb(self, x, surr_mf, fidelity, cost, kappa=1.96): 
        """
        Variable-fidelity Lower Confidence Bound
        
        Parameters: 
        -----------------
        cost: list
            cost ratio for the optimized function. 
        kappa: float
            balance coefficient btw exploration and exploitation
        """
        cr = cost[0] / cost[1] 
        mean_hf, std_hf = surr_mf.predict(x, return_std=True) 
        _, std_lf = surr_mf.predict_lf(x, return_std=True) 

        if fidelity == 'high': 
            vflcb = mean_hf - kappa * std_hf 
        elif fidelity == 'low': 
            vflcb = mean_hf - kappa * std_lf * cr 
        else: 
            print("Wrong fidelity str input! \n") 
        
        return vflcb 