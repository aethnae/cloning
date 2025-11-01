import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags, eye, kron, csc_matrix


def create_1d_laplacian(size: int) -> csc_matrix:
    """
    Helper function that creates the matrix representation D^2 of the discrete second derivative by stacking diagonals.

    :param size: matrix will be of dimension size x size
    :return: sparse matrix containing the discrete second derivative
    """
    main = -2 * np.ones(size)
    sub = np.ones(size-1)
    top = np.ones(size-1)
    result = diags([sub, main, top], [-1, 0, 1], shape=(size, size), format="csc")
    return result


def create_forward_diff(size: int) -> csc_matrix:
    """
    Helper function that creates the matrix representation D^v of the discrete first derivative by stacking diagonals.

    :param size: matrix will be of dimension size x size
    :return: sparse matrix containing the discrete first derivative
    """
    main = -1 * np.ones(size)
    top = np.ones(size - 1)

    main[-1] = -1.0

    result = diags([main, top], [0, 1], shape=(size, size), format='csc')
    return result


def create_backward_diff(size: int) -> csc_matrix:
    """
    Helper function that creates the matrix representation D^r of the backwards differences by stacking diagonals.

    :param size: matrix will be of dimension size x size
    :return: sparse matrix containing the backwards difference
    """
    main = np.ones(size)
    sub_diag = -1 * np.ones(size - 1)

    main[0] = 1.0

    result = diags([sub_diag, main], [-1, 0], shape=(size, size), format='csc')
    return result


def create_2d_laplacian(N: int, M: int) -> csc_matrix:
    """
    Computes the vectorized discrete Laplace operator Δ = (I_M ⊗ D_N^(2) + D_M^(2) ⊗ I_N).

    :param N: vertical length of the image
    :param M: horizontal length of the image
    :return: sparse matrix containing the discrete Laplace operator
    """
    I_M = eye(M, format="csc")
    I_N = eye(N, format="csc")
    discrLaplaceN = create_1d_laplacian(N)
    discrLaplaceM = create_1d_laplacian(M)
    result = kron(I_M, discrLaplaceN, format="csc") + kron(discrLaplaceM, I_N, format="csc")
    return result


def visual(N: int, M: int) -> None:
    """
    Convenience function for visualizing the discrete Laplace operator's sparsity.

    :param N: vertical length of the image
    :param M: horizontal length of the image
    :return: prints plot of the discrete Laplace operator
    """
    res = create_2d_laplacian(N, M)
    plt.figure(figsize=(10, 8))

    ax = plt.gca()
    ax.spy(res, markersize=3)
    ax.set_title(f"Sparsity Pattern of Laplace Operator for (N,M) = ({N},{M})")
    ax.set_xlabel("Column Index")
    ax.set_ylabel("Row Index")

    plt.show(block=True)
    return None

# For testing purposes, will be called separately in main.py
if __name__ == "__main__":
    visual(5, 7)