import numpy as np
from scipy.sparse.linalg import cg
from discrLaplace import discrLaplace2D, discrBackwardDiff1D, discrForwardDiff1D

def clone_mixed(f: np.ndarray, g: np.ndarray, target_y: int, target_x: int) -> np.ndarray:
    N, M = g.shape
    f_tar = f[target_y: target_y + N, target_x: target_x + M]
    vec_f = f_tar.flatten('F')

    forward_N = discrForwardDiff1D(N)
    forward_M = discrForwardDiff1D(M)

    grad_gy = forward_N @ g
    grad_gx = g @ forward_M.T
    grad_fy = forward_N @ f_tar
    grad_fx = f_tar @ forward_M.T

    norm_g = grad_gy ** 2 + grad_gx ** 2
    norm_f = grad_fy ** 2 + grad_fx ** 2

    mask = norm_f > norm_g
    v_y = np.where(mask, grad_fy, grad_gy)
    v_x = np.where(mask, grad_fx, grad_gx)

    backward_N = discrBackwardDiff1D(N)
    backward_M = discrBackwardDiff1D(M)

    div_v_matrix = backward_N @ v_y + v_x @ backward_M.T
    b = div_v_matrix.flatten('F')
    A = discrLaplace2D(N, M).tolil()

    boundary_indices = []
    for i in range(N):
        for j in range(M):
            if i == 0 or i == N - 1 or j == 0 or j == M - 1:
                k = i + j * N
                boundary_indices.append(k)

    for k in boundary_indices:
        A[k, :] = 0.0
        A[k, k] = 1.0
        b[k] = vec_f[k]

    A_csc = A.tocsc()
    vec_g = g.flatten('F')
    vec_h, info = cg(A_csc, b, x0=vec_f, maxiter=1000)
    h_region = vec_h.reshape((N, M), order='F')
    h_region = np.clip(h_region, 0.0, 1.0)

    res = f.copy()
    res[target_y: target_y + N, target_x: target_x + M] = h_region
    return res