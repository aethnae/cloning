import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, eye, kron, csc_matrix

def discrLaplace1D(size: int) -> csc_matrix:
    """
    Helper function that creates the matrix representation of the discrete second derivative
    :param size: size of vector we want to differentiate
    :return: sparse matrix containing the discrete second derivative
    """
    main = -2 * np.ones(size)
    sub = np.ones(size-1)
    top = np.ones(size-1)
    result = diags([sub, main, top], [-1, 0, 1], shape=(size, size), format="csc")
    return result

def discrLaplace2D(N: int, M: int) -> csc_matrix:
    """
    Computes the vectorized discrete Laplace operator Δ = (I_M ⊗ D_N^(2) + D_M^(2) ⊗ I_N)
    :param N: vertical length of the image
    :param M: horizontal length of the image
    :return: sparse matrix containing the discrete Laplace operator
    """
    I_M = eye(M, format="csc")
    I_N = eye(N, format="csc")
    discrLaplaceN = discrLaplace1D(N)
    discrLaplaceM = discrLaplace1D(M)
    result = kron(I_M, discrLaplaceN, format="csc") + kron(discrLaplaceM, I_N, format="csc")
    return result

def visual(N: int, M: int) -> None:
    res = discrLaplace2D(N, M)
    plt.figure(figsize=(9,7))
    plt.imshow(res.toarray())
    plt.colorbar(label="Wert")
    plt.title(f"Vectorised Laplace operator Δ for image of size = ({N},{M})")
    plt.show()
    return None

print(visual(5,7))
