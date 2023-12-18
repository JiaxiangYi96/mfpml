from typing import Any
import numpy as np 


class mfArray: 
    """
    Base class for multi-fidelity data with ndarray
    """
    def __init__(self, 
                 data: list=None) -> None:
        self.data = data 

    def __call__(self, data: list, **kwds: Any) -> Any:
        self.data = data 

    def __getitem__(self, index): 
        return self.data[index] 
    
    @property
    def n_levels(self): 
        return len(self.data)
    
    @property 
    def hf(self) -> np.ndarray: 
        return self.data[0]
    
    @property
    def size(self): 
        return [len(self.data[i]) for i in range(self.n_levels)]