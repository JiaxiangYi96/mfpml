import numpy as np 
import pytest 
 

from mfpml import mfArray 

def test_mfArray() -> None: 
    mfarray1 = mfArray(num_fidelity=3) 
    assert mfarray1 == [None] * 3
    
    data = mfArray([np.random.rand(4,2), np.random.rand(8,2)]) 
    assert data.num_fidelity == 2 
    np.random.seed(123)
