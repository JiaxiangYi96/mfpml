import numpy as np
from scipy.spatial.distance import pdist, squareform

class model:

    def drop_neighbor(self, X: np.ndarray, Y: np.ndarray, epsilon: float = 1e-3) -> np.ndarray:
        nx = X.shape[0]
        dis_mat = squareform(pdist(X))
        del_index = []
        for i in range(nx-1):
            if np.min(dis_mat[i, i+1:]) < epsilon:
                del_index.append(i)
        return np.delete(X, del_index, axis=0), np.delete(Y, del_index, axis=0)
