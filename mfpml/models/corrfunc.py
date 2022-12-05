import numpy as np  
from scipy.spatial.distance import cdist

class corrfunc: 
    """
    Base class for kernel
    """ 
    @property 
    def hyperparameters(self): 
        """
        Returns a list of all hyperparameters specifications
        """
        return self.param.tolist()

    @property
    def _get_num_para(self) -> int:
        """Return number of parameters of the kernel

        Returns
        -------
        int
            number of parameters
        """
        return self.num_para 

    @property
    def _get_bounds(self) -> np.ndarray: 
        """Get the parameters' bounds

        Returns
        -------
        np.ndarray
            design bounds of the parameters
        """
        return self.bounds

    @property
    def _get_bounds_list(self) -> list: 
        """Get the parameters' bounds with list

        Returns
        -------
        list
            design bounds of the parameters
        """
        return self.bounds.tolist()

    @property
    def _get_low_bound(self) -> list: 
        """Get the low bound of the parameters

        Returns
        -------
        list
            low bound of kernel's parameters
        """
        return self.bounds[:, 0].tolist()

    @property
    def _get_high_bound(self) -> list: 
        """Get the high bound of the parameters

        Returns
        -------
        list
            high bound of kernel's parameters
        """
        return self.bounds[:, 1].tolist()

class KRG(corrfunc): 
    def __init__(
        self, 
        theta: np.array, 
        parameters: list = ['theta'], 
        bounds: list = [-4, 3]) -> None: 
        """_summary_

        Parameters
        ----------
        theta : np.ndarray, shape=((1,num_dims))
            initial guess of the parameters
        parameters : list, optional
            list the parameters to fit, by default ['theta']
        bounds : list, optional
            log bounds of the parameters, by default [-4, 3]
        """
        self.param = 10 ** theta
        self.parameters = parameters
        self.bounds = [] 
        self.num_para = theta.size
        for i in range(self.num_para): 
            self.bounds.append(bounds)
        self.bounds = np.array(self.bounds)

    def K(self, X, Y): 
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        dist = np.sum(X ** 2 * self.param, 1).reshape(-1, 1) + np.sum(Y ** 2 * self.param, 1)\
            - 2 * np.dot(np.sqrt(self.param) * X, (np.sqrt(self.param) * Y).T) 
        return np.exp(-dist)+ np.finfo(float).eps * (10 + X.shape[1]) * np.eye(X.shape[0], Y.shape[0])

    def __call__(self, X: np.array, Y: np.array, param: np.array, eval_grad=False): 
        """
        Return the kernel k(x, y) and optionally its gradient.
        
        Parameters
        ----------
        X : np.array
            array of the first samples shape=((n1, num_dims))
        Y : np.array
            array of the second samples shape=((n2, num_dims))
        param : np.array
            parameters in the specific kernel, shape=((1, num_params))
        eval_grad : bool, optional
            whether the gradient with respect to the log of the kernel 
            hyperparameter is computed, by default False
            Only supported when Y is None

        Returns
        -------
        np.array
            kernel values with respect to parameters param
        """
        if eval_grad: 
            pass
        else:
            X = np.atleast_2d(X)
            Y = np.atleast_2d(Y)
            param = 10 ** param
            dist = np.sum(X ** 2 * param, 1).reshape(-1, 1) + np.sum(Y ** 2 * param, 1)\
                - 2 * np.dot(np.sqrt(param) * X, (np.sqrt(param) * Y).T) 
            return np.exp(-dist)+ np.finfo(float).eps * (10 + X.shape[1]) * np.eye(X.shape[0], Y.shape[0])

    def set_params(self, params): 
        """
        Set the parameters of the kernel. 
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