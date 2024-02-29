from typing import Any

import numpy as np


class mfArray:
    """
    Base class for multi-fidelity data with ndarray
    """
    def __init__(self,
                 data: list[np.ndarray] = None, 
                 num_fidelity: int = None, 
                 ) -> None:
        """
        Multi-fidelity data 

        Parameters
        ----------
        data : list[np.ndarray], optional
            data, by default None
        num_fidelity : int, optional
            number of fidelity levels by default None
        """
        if data is not None: 
            self.data = data 
        elif num_fidelity is None: 
            raise NotImplementedError("Please assing number of fidelities (num_fidelity)!") 
        else: 
            self.data = [None] * num_fidelity

    # def __call__(self, data: list, **kwds: Any) -> Any:
    #     self.data = data

    def __getitem__(self, index):
        return self.data[index]

    @property
    def num_fidelity(self):
        return len(self.data)

    @property
    def hf_data(self) -> np.ndarray:
        return self.data[-1]

    @property
    def size(self):
        return [len(self.data[i]) for i in range(self.num_fidelity)]

    def append(self, data, fidelity_ind: int): 
        self.data[fidelity_ind] = np.vstack(self.data[fidelity_ind], np.atleast_2d(data))

    def item(self): 
        return self.data

