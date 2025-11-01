import numpy as np
from scipy.sparse.linalg import cg

from .discrLaplace import create_2d_laplacian, create_backward_diff, create_forward_diff


def clone_mixed(f: np.ndarray, g: np.ndarray, target_y: int, target_x: int) -> np.ndarray:
    """
    Implements seamless cloning using the mixed gradients method. This
    function solves the discrete Poisson equation Δh = div(v) on the patch Ω,
    where v is a mixed gradient field from f* and g, and h|∂Ω = f*|∂Ω.

    :param f: target grayscale image to clone into
    :param g: source patch
    :param target_y: y-coordinate of top-left corner of the patch
    :param target_x: x-coordinate of top-left corner of the patch
    :return: a copy of f with patch g seamlessly cloned into
    """

    # vectorizing target patch for boundary conditions
    N, M = g.shape
    f_tar = f[target_y: target_y + N, target_x: target_x + M]
    vec_f = f_tar.flatten('F')

    # forward difference operators for gradient calculation
    forward_N = create_forward_diff(N)
    forward_M = create_forward_diff(M)

    # calculate 2D gradients for source (g) and target (f) image
    grad_gy = forward_N @ g
    grad_gx = g @ forward_M.T
    grad_fy = forward_N @ f_tar
    grad_fx = f_tar @ forward_M.T

    # create mixed gradient field
    norm_g = grad_gy ** 2 + grad_gx ** 2
    norm_f = grad_fy ** 2 + grad_fx ** 2

    mask = norm_f > norm_g
    v_y = np.where(mask, grad_fy, grad_gy)
    v_x = np.where(mask, grad_fx, grad_gx)

    # backward difference operators for divergence calculation
    backward_N = create_backward_diff(N)
    backward_M = create_backward_diff(M)

    # divergence calculation and linear system setup
    div_v_matrix = backward_N @ v_y + v_x @ backward_M.T
    b = div_v_matrix.flatten('F')
    A = create_2d_laplacian(N, M).tolil()

    # enforcing boundary condition, same as in cloneGrayscale.py
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

    # solve sparse linear system Ax = b with conjugate gradients and round pixel values if outside of given range
    A_csc = A.tocsc()
    vec_h, info = cg(A_csc, b, x0=vec_f, maxiter=1000)
    h_region = vec_h.reshape((N, M), order='F')
    h_region = np.clip(h_region, 0.0, 1.0)

    # insert solved patch into target image
    res = f.copy()
    res[target_y: target_y + N, target_x: target_x + M] = h_region
    return res