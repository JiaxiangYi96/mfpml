import numpy as np 
import pytest 
 

from mfpml.core import mfArray 

def test_mfArray() -> None: 
    np.random.seed(123)
    data = mfArray([np.random.rand(4,2), np.random.rand(8,2)]) 

    assert data.n_levels == 2 
    np.random.seed(123)
    assert data[0] == np.random.rand(4,2)
