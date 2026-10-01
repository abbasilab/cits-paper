"""
rcit.py

Minimal Random Fourier Features approximation of HSIC for conditional
independence testing. Following Strobl, Zhang, Visweswaran 2019.

API
---
  pval = rcit_test(x, y, z=None, K=25, seed=0)

  x, y : (N,) or (N, d) arrays - the variables being tested for independence
  z    : (N, d_z) array or None - conditioning variables (None => marginal test)
  K    : number of random Fourier features per variable
  seed : RNG seed (for reproducibility)

  Returns p-value under H0: x ⊥ y | z.

Implementation
--------------
  - RFF embedding: each variable mapped to 2K features via cos/sin of K
    random projections sampled from N(0, 1/sigma²) where sigma is the median
    pairwise distance.
  - For marginal test (z=None): compute squared Frobenius norm of
    cross-covariance between feature embeddings.
  - For conditional test: residualize x and y embeddings against z embedding,
    then compute cross-covariance of residuals.
  - Null distribution: gamma approximation of the cross-covariance norm
    matched to its first two moments under H0.

Complexity
----------
  O(N * K + K^2). At N=250, K=25: ~6,250 + 625 = ~7,000 ops per test.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import gamma


def _median_bandwidth(X: np.ndarray) -> float:
    """Median pairwise distance heuristic for Gaussian kernel bandwidth."""
    if X.ndim == 1:
        X = X[:, None]
    n = X.shape[0]
    if n > 200:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=200, replace=False)
        X = X[idx]
    dists = pdist(X)
    med = np.median(dists)
    if med < 1e-12:
        med = 1.0
    return float(med)


def _rff_embed(X: np.ndarray, K: int, sigma: float, rng) -> np.ndarray:
    """Random Fourier Features approximation of Gaussian kernel.

    phi(x) = (1/sqrt(K)) * [cos(W x + b), sin(W x + b)] - returns shape (N, 2K).
    Equivalent to k(x,y) ≈ phi(x).phi(y).
    """
    if X.ndim == 1:
        X = X[:, None]
    N, d = X.shape
    W = rng.normal(scale=1.0 / sigma, size=(d, K))  # (d, K)
    b = rng.uniform(0, 2 * np.pi, size=K)
    proj = X @ W + b                                # (N, K)
    phi = np.empty((N, 2 * K))
    phi[:, :K] = np.cos(proj)
    phi[:, K:] = np.sin(proj)
    phi /= np.sqrt(K)
    return phi


def _residualize(phi: np.ndarray, phi_z: np.ndarray,
                  ridge_lambda: float = 1e-2) -> np.ndarray:
    """Kernel-ridge-regression residualize phi against phi_z in RFF space.

    Solve in closed form: B = (phi_z^T phi_z + λI)^{-1} phi_z^T phi.
    Larger ridge_lambda stabilizes the inversion when phi_z is rank-deficient
    or near-collinear; following Strobl 2019, a noticeable ridge (e.g., 1e-2)
    is essential for the conditional test to behave correctly.
    """
    K = phi_z.shape[1]
    A = phi_z.T @ phi_z + ridge_lambda * np.eye(K)
    B = np.linalg.solve(A, phi_z.T @ phi)
    return phi - phi_z @ B


def _Tstat(phi_x: np.ndarray, phi_y: np.ndarray) -> float:
    """RFF-HSIC test statistic: N * ||cross-cov||^2_F."""
    N = phi_x.shape[0]
    Sxy = (phi_x.T @ phi_y) / N
    return N * float(np.sum(Sxy ** 2))


def rcit_test(x: np.ndarray, y: np.ndarray,
              z: np.ndarray | None = None,
              K: int = 25, n_perm: int = 100,
              seed: int = 0) -> float:
    """RFF-HSIC conditional independence test using permutation null.

    For conditional test, residualizes phi_x and phi_y against phi_z first,
    then tests whether the residuals are independent (Strobl-style residual
    decomposition). Permutation of phi_y rows yields the null distribution.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    N = x.shape[0]
    if y.shape[0] != N:
        raise ValueError("x and y must have the same N")

    rng = np.random.default_rng(seed)

    sigma_x = _median_bandwidth(x)
    sigma_y = _median_bandwidth(y)
    phi_x = _rff_embed(x, K, sigma_x, rng)
    phi_y = _rff_embed(y, K, sigma_y, rng)

    phi_x = phi_x - phi_x.mean(axis=0, keepdims=True)
    phi_y = phi_y - phi_y.mean(axis=0, keepdims=True)

    if z is not None:
        z = np.asarray(z, dtype=np.float64)
        if z.ndim == 1:
            z = z[:, None]
        if z.shape[0] != N:
            raise ValueError("z must have N rows")
        sigma_z = _median_bandwidth(z)
        phi_z = _rff_embed(z, K, sigma_z, rng)
        phi_z = phi_z - phi_z.mean(axis=0, keepdims=True)
        phi_x = _residualize(phi_x, phi_z)
        phi_y = _residualize(phi_y, phi_z)

    T_obs = _Tstat(phi_x, phi_y)

    # Permutation null: shuffle rows of phi_y while keeping phi_x fixed.
    perm_count_geq = 1   # add-one smoothing (so p is never exactly 0)
    for b in range(n_perm):
        idx = rng.permutation(N)
        T_b = _Tstat(phi_x, phi_y[idx])
        if T_b >= T_obs:
            perm_count_geq += 1
    pval = perm_count_geq / (n_perm + 1)
    return pval


# Helper compatible with PC's partial-correlation pvalue interface ------------

def rcit_pvalue_for_pc(C: np.ndarray, i: int, j: int,
                        S: tuple[int, ...] | list[int], N: int,
                        K: int = 25, seed: int = 0) -> float:
    """PC-style CI test signature.

    C : (N, p) data matrix (chi-stacked observations as rows).
    i, j : variable indices into the columns of C.
    S    : tuple of conditioning column indices.
    N    : number of observations (= C.shape[0]); kept for API parity.
    K    : RFF feature count.

    Returns p-value of H0: C[:, i] ⊥ C[:, j] | C[:, S].
    """
    x = C[:, i]
    y = C[:, j]
    if len(S) == 0:
        z = None
    else:
        z = C[:, list(S)]
    return rcit_test(x, y, z=z, K=K, seed=seed)


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    N = 1000

    # Case 1: independent
    x = rng.normal(size=N); y = rng.normal(size=N)
    p1 = rcit_test(x, y)
    print(f"Independent: p = {p1:.3f}  (should be > 0.05)")

    # Case 2: linear dependence
    x = rng.normal(size=N); y = 0.5 * x + rng.normal(size=N)
    p2 = rcit_test(x, y)
    print(f"Linear dep: p = {p2:.4f}  (should be very small)")

    # Case 3: nonlinear dependence (y = sin(x))
    x = rng.normal(size=N); y = np.sin(2 * x) + 0.3 * rng.normal(size=N)
    p3 = rcit_test(x, y)
    print(f"Nonlinear dep (sin): p = {p3:.4f}  (should be very small)")

    # Case 4: conditional independence X = Z + noise, Y = Z + noise (X ⊥ Y | Z)
    z = rng.normal(size=N)
    x = z + 0.5 * rng.normal(size=N)
    y = z + 0.5 * rng.normal(size=N)
    p4_marg = rcit_test(x, y)
    p4_cond = rcit_test(x, y, z=z)
    print(f"X-Y dep but X⊥Y|Z: marg p = {p4_marg:.4f} (small), "
          f"cond p = {p4_cond:.3f} (should be > 0.05)")
