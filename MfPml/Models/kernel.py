import numpy as np  
from scipy.spatial.distance import cdist

class kernel: 
    """
    Base class for kernel
    """
    default_bounds = {
        'l': [-5, 1], 
        'sigmaf': [-5, 1], 
        'sigman': [-5, 1], 
        'theta' : [-4, 3]
        }

    def set_params(self, params): 
        """
        Set the parameters of the kernel. 
        
        Returns 
        --------
        self
        """
        if not params:
            return self
        for key, value in params.items():
            setattr(self, key, 10 ** value)
        return self

    def __call__(self, X, Y, eval_grad=False): 
        """
        Return the kernel k(x, y) and optionally its gradient.
        
        Parameters 
        -------
        X, Y: np.array, shape=((n, n_dims))
        eval_gradient : bool
            Determines whether the gradient with respect to the log of the kernel 
            hyperparameter is computed.
            Only supported when Y is None.

        Returns
        -------
        K : ndarray, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)
        K_gradient : ndarray, shape (n_samples_X, n_samples_X, n_dims), optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        if eval_grad: 
            pass
        else:
            return self.K(X, Y) 

    @property
    def n_dims(self): 

        pass

    @property 
    def hyperparameters(self): 
        """
        Returns a list of all hyperparameters specifications
        """

        pass 


class RBF(kernel): 
    def __init__(self, l=0.0, sigmaf=0.0, sigman=0.0, parameters=['l', 'sigmaf'], bounds=None): 
        """
        Squared exponential kernel
        
        Parameters: 
        l: float 
            Characteristic length scale. 
        sigmaf: float
            Signal variance. Controls the overall scale of the covariance function.
        sigman: float
            Noise variance. Additive noise in output space.
        """
        self.l = 10 ** l 
        self.sigmaf = 10 ** sigmaf
        self.sigman = sigman 
        self.parameters = parameters 
        if bounds is not None: 
            self.bounds = bounds 
        else: 
            self.bounds = [] 
            for param in parameters: 
                self.bounds.append(self.default_bounds[param])

    def K(self, X, Y): 
        dist = cdist(X, Y) 
        return self.sigmaf * np.exp(-0.5 / self.l ** 2 * dist ** 2) + self.sigman * kronDelta(X, Y) 

    


class KRG(kernel): 
    def __init__(self, theta, parameters=['theta'], bounds=None): 
        """
        Kriging kernel

        Parameters: 
        theta: np.ndarray, shape=((n_dims,1))
            Measure of how active the function we are approximating is. 
        p: float
            smoothness parameter 
        """
        self.theta = 10 ** theta
        self.parameters = parameters
        if bounds is not None: 
            self.bounds = bounds 
        else: 
            self.bounds = [] 
            for param in parameters: 
                self.bounds.append(self.default_bounds[param]) 

    def K(self, X, Y): 
        dist = np.sum(X ** 2 * self.theta, 1).reshape(-1, 1) + np.sum(Y ** 2 * self.theta, 1)\
            - 2 * np.dot(np.sqrt(self.theta) * X, (np.sqrt(self.theta) * Y).T) 
        return np.exp(-dist)+ np.finfo(float).eps * (10 + X.shape[1]) * np.eye(X.shape[0], Y.shape[0])




def kronDelta(X, Y): 
    """
    Computes Kronecker delta for rows in x and y.

    Parameters
    ----------
    x, y: np.ndarray, shape=((n, nfeatures))
   
    Returns
    -------
    np.ndarray
        Kronecker delta between row pairs of `X` and `Xstar`.
    """
    return cdist(X, Y) < np.finfo(np.float32).eps