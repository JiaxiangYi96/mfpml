import numpy as np 
from MFAcquisitions import mfSingleObjAcf

class MFSOBO: 
    """
    Multi-fidelity single objective Bayesian optimization
    """
    def __init__(self, acf): 

        self.acf = acf
        self.iter = 0 
        self.history =[]

    def _initial_run(self, inital_sample)
    def update_model(self, update_dict): 

        XHnew = update_dict['high'] 
        XLnew = update_dict['low']
        if XHnew is not None: 
            YHnew = self.f(XHnew, fidelity='high') 
        if XLnew is not None: 
            YLnew = self.f(XLnew, fidelity='low')
        self.mf_model.update(XHnew, YHnew, XLnew, YLnew)



    def run(self, max_iter=5, equ_cost=10, resume=False): 

        if not resume: 
            self.mf_model.train

    def _printCurrent(self): 
        
        iter = self.iter
        eval_XH = 


class vfei(MFSOBO, mfSingleObjAcf): 

    
    def __init__(self, problem, mf_surrogate): 
        """
        
        Parameters 
        -------
        mf_surrogate: mf_surr instance
            multi-fidelity surrogate model.
        acqusition: mfAcqusition instance
        """
        self.f = problem 
        self.mf_model = mf_surrogate 


    def _optAcf(self, opt_method='DE'): 

          pass

    def eval_acf(self, x, fidelity='high'): 


        pre, std = self.mf_model.predict(x, return_std=True) 
        if fidelity == 'high': 
            s = std
        elif fidelity == 'low': 
            _, std_lf = self.mf_model.predict_lf(x, return_std=True) 
            s = self.mf_model.mu * std_lf
        else: 
            ValueError('Unknown fidelity input.')
        
        z = (fmin - pre) / std
        vfei = (fmin - pre) * norm.cdf(z) + std * norm.pdf(z) 
        vfei[s<np.finfo(float).eps] = 0. 

        return vfei
