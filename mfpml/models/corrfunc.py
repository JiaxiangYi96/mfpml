import numpy as np  
from scipy.spatial.distance import cdist

class corrfunc: 
    """
    Base class for kernel
    """

    
    def __call__(self, X, Y, param, eval_grad=False): 
        """
        Return the kernel k(x, y) and optionally its gradient.
        
        Parameters 
        -------
        X, Y: np.array, shape=((n, num_dims))
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
            X = np.atleast_2d(X)
            Y = np.atleast_2d(Y)
            dist = np.sum(X ** 2 * param, 1).reshape(-1, 1) + np.sum(Y ** 2 * param, 1)\
                - 2 * np.dot(np.sqrt(param) * X, (np.sqrt(param) * Y).T) 
            return np.exp(-dist)+ np.finfo(float).eps * (10 + X.shape[1]) * np.eye(X.shape[0], Y.shape[0])

    @property
    def n_dims(self): 

        pass

    @property 
    def hyperparameters(self): 
        """
        Returns a list of all hyperparameters specifications
        """

        pass 


class KRG(corrfunc): 
    def __init__(self, theta, parameters=['theta'], bounds=[-4, 3]): 
        """
        Kriging kernel

        Parameters: 
        theta: np.ndarray, shape=((1,num_dims))
            Measure of how active the function we are approximating is. 
        p: float
            smoothness parameter 
        """
        self.param = 10 ** theta
        self.parameters = parameters
        self.bounds = [] 
        self.num_para = theta.shape[1]
        for i in range(self.num_para): 
            self.bounds.append(bounds)

    def K(self, X, Y): 
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        dist = np.sum(X ** 2 * self.param, 1).reshape(-1, 1) + np.sum(Y ** 2 * self.param, 1)\
            - 2 * np.dot(np.sqrt(self.param) * X, (np.sqrt(self.param) * Y).T) 
        return np.exp(-dist)+ np.finfo(float).eps * (10 + X.shape[1]) * np.eye(X.shape[0], Y.shape[0])

    def set_params(self, params): 
        """
        Set the parameters of the kernel. 
        
        Returns 
        """
        
        setattr(self, 'param', 10 ** params)



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