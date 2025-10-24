import numpy as np
from scipy.sparse.linalg import cg
from discrLaplace import discrLaplace2D

def clone_grayscale(f: np.ndarray, g: np.ndarray, target_y: int, target_x: int) -> np.ndarray:
    N, M = g.shape
    A = discrLaplace2D(N, M).tolil()
    vec_g = g.flatten('F')
    b = A.dot(vec_g)

    boundary_indices = []
    for i in range(N):
        for j in range(M):
            if i == 0 or i == N - 1 or j == 0 or j == M - 1:
                k = i + j * N
                boundary_indices.append(k)

    f_target = f[target_y: target_y + N, target_x: target_x + M]
    vec_f = f_target.flatten('F')

    for k in boundary_indices:
        A[k, :] = 0.0
        A[k, k] = 1.0
        b[k] = vec_f[k]

    A_csc = A.tocsc()
    vec_h, info = cg(A_csc, b, x0=vec_g)
    h_region = vec_h.reshape((N, M), order='F')
    h_region = np.clip(h_region, 0.0, 1.0)

    res = f.copy()
    res[target_y: target_y + N, target_x: target_x + M] = h_region
    return res
