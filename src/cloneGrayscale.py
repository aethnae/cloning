import numpy as np
from scipy.sparse.linalg import cg
from .discrLaplace import create_2d_laplacian

def clone_grayscale(f: np.ndarray, g: np.ndarray, target_y: int, target_x: int) -> np.ndarray:
    """
    Implements seamless cloning using the standard Poisson equation (Laplace method).
    This function solves the discrete Poisson equation Δh = Δg on the patch Ω,
    using the boundary condition h|∂Ω = f*|∂Ω.

    :param f: target grayscale image (f*) to clone into
    :param g: patch to clone
    :param target_y: y-coordinate of top-left corner of the patch
    :param target_x: x-coordinate of top-left corner of the patch
    :return: a copy of f with the patch g seamlessly cloned into
    """

    # initialize linear system Ax = b
    N, M = g.shape
    A = create_2d_laplacian(N, M).tolil()
    vec_g = g.flatten('F')
    b = A.dot(vec_g)

    # get boundary indices
    boundary_indices = []
    for i in range(N):
        for j in range(M):
            if i == 0 or i == N - 1 or j == 0 or j == M - 1:
                # column-major index
                k = i + j * N
                boundary_indices.append(k)

    # target patch location
    f_target = f[target_y: target_y + N, target_x: target_x + M]
    vec_f = f_target.flatten('F')

    # modify A and b to enforce boundary condition
    for k in boundary_indices:
        A[k, :] = 0.0
        A[k, k] = 1.0
        b[k] = vec_f[k]

    # solve sparse linear system Ax = b with conjugate gradients and round pixel values if outside of given range
    A_csc = A.tocsc()
    vec_h, info = cg(A_csc, b, x0=vec_f, maxiter=1000)
    h_region = vec_h.reshape((N, M), order='F')
    h_region = np.clip(h_region, 0.0, 1.0)

    # insert solved patch into target image
    res = f.copy()
    res[target_y: target_y + N, target_x: target_x + M] = h_region
    return res
