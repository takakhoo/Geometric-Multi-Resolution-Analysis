import numpy as np
from typing import List, Tuple, Union

def mindim(sigmas: np.ndarray,
           errortype: str,
           err: float) -> int:
    s2: float = np.sum(sigmas**2)

    tol: float = None
    if errortype.lower() == "absolute":
        tol = err**2
    else:
        tol = err*s2

    dim: int = 0
    while dim < sigmas.shape[0]-1 and s2>tol:
        s2 = s2 - sigmas[dim]**2

        dim += 1
    return dim


def rand_pca(A: np.ndarray,
             k: int,
             its: int = 2,
             l: int = None,
             shelf=None, inverse=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    U: np.ndarray = None
    s: np.ndarray = None
    V: np.ndarray = None

    if l is None:
        l = k + 2

    n, m = A.shape
    if (its*l >= m/1.25) or (its*l >= n/1.25):
        U, s, V = np.linalg.svd(A, full_matrices=inverse)

        U = U[:, :k]
        s = s[:k]
        # V = V[:, :k] #CHANGED V = V[:, 1:k]

    else:
        H: np.ndarray = None
        if n >= m:
            if shelf is None:
                H = A.dot(2*np.random.randn(m, l) - np.ones(m, l))
            else:
                shelf.rand = np.random.randn(m, l)
                H = A.dot(2*shelf.rand - (shelf.ndarray((m,l), dtype=float)*0+1))

            F: np.ndarray = None
            if shelf is None:
                F = np.zeros(n, its*l)
            else:
                F = shelf.nparray((n, its*l), dtype=float)*0
            F[:n, :l] = H

            for it in range(its):
                H = H.T.dot(A).T
                H = A.dot(H)
                F[:n, (it+1)*l:(it+2)*l] = H

            Q,_,_ = qr(F, mode="enconomic")
            U2, s, V = np.linalg.svd(Q.T.dot(A))
            U = Q.dot(U2)

            U = U[:, :k]
            s = s[:k]
            # V = V[:,:k]

        else:
            if shelf is None:
                H = (2*np.random.randn(n, l) - np.ones(n, l)).dot(A).T
            else:
                shelf.rand = np.random.randn(n,l)
                H = (2*shelf.rand - (shelf.ndarray((n,l), dtype=float)*0+1)).dot(A).T

            F: np.ndarray = None
            if shelf is None:
                F = np.zeros(m, its*l)
            else:
                F = shelf.nparray((m, its*l), dtype=float)*0
            F[:n, :l] = H
            F[:m, :l] = H

            for it in range(its):
                H = A.dot(H)
                H = H.T.dot(A).T
                F[:m, (it+1)*l:(it+2)*l] = H

            Q,_,_ = qr(F, mode="enconomic")
            U, s, V2 = np.linalg.svd(A.dot(Q))
            V = Q.dot(V2)

            U = U[:, :k]
            s = s[:k]
            # V = V[:,:k]

    return U, s, V

def easy_pca(A: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return U[:, :k], s[:k], Vt

def easy_node_function(X: np.ndarray,
                       manifold_dim: int,
                       max_dim: int,
                       is_leaf: bool,
                       errortype: str = "relative",
                       shelf=None,
                       
                       threshold: float = 0.5,
                       precision: float = 1e-2) -> Tuple[np.ndarray, int, float,
                                                            np.ndarray, np.ndarray]:
    # X have shape (n, d)
    mu: np.ndarray = np.mean(X, axis=0, keepdims=True)
    X_norm = X - mu

    radius: float = np.sqrt(np.max((X_norm**2).sum(axis=-1)))
    size: int = max(1, X.shape[0])  
    max_dim = max(X_norm.shape)
    basis, sigmas, Z = easy_pca(X_norm, min(min(X_norm.shape), max_dim))
    rem_energy: float = max(np.sum(np.sum(X_norm**2) - np.sum(sigmas**2)), 0)
    return mu, X.shape[0], radius, basis, sigmas, Z
    
def node_function(X: np.ndarray,
                  manifold_dim: int,
                  max_dim: int,
                  is_leaf: bool,
                  errortype: str = "relative",
                  shelf=None,
                  threshold: float = 0.5,
                  precision: float = 1e-2, inverse = False) -> Tuple[np.ndarray, int, float,
                                                 np.ndarray, np.ndarray]:
    if inverse:
        X = X.T # convert to (d,n)
        mu: np.ndarray = np.mean(X, axis=1, keepdims=True)
    else:
        mu: np.ndarray = np.mean(X, axis=0, keepdims=True)

    X_mean_centered: np.ndarray = X - mu
    radius: float = np.sqrt(np.max((X_mean_centered**2).sum(axis=-1)))

    size: int = max(1, X.shape[0])

    sigmas: np.ndarray = None
    basis: np.ndarray = None
    if is_leaf or manifold_dim == 0:
        V, s, Z = rand_pca(X_mean_centered, min(min(X_mean_centered.shape), max_dim))
        rem_energy: float = max(np.sum(np.sum(X_mean_centered**2) - np.sum(s**2)), 0)
        sigmas = np.hstack([s, [np.sqrt(rem_energy)]]) / np.sqrt(size)

        dim: int = None
        if not is_leaf:
            dim = min(s.shape[0], mindim(sigmas, errortype, threshold))
        else:
            dim = min(s.shape[0], mindim(sigmas, errortype, precision))
        basis = V[:, :dim]
        # print('adaptive dim:',dim, 'original dim', s.shape[0])
        # basis = V

    else:
        V, s, Z = rand_pca(X_mean_centered, min(min(X_mean_centered.shape), manifold_dim))
        sigmas = s / np.sqrt(size)
        if V.shape[-1] < manifold_dim:
            V = np.hstack([V, np.zeros((V.shape[0], manifold_dim - size))])
        basis = V[:, :min(manifold_dim, int(np.sum(sigmas > 0)))]
        # print('hard select dim:',min(manifold_dim, int(np.sum(sigmas > 0))), 'original dim', s.shape[0])

    if inverse:
        return mu, X.shape[0], radius, basis.T, sigmas, Z
    else:
        return mu, X.shape[0], radius, basis, sigmas, Z
