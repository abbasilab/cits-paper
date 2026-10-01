"""
gpu_cits_ci.py

Full-GPU conditional-independence test for CITS non-Gaussian regimes,
reproducing kpcalg::kernelCItest(ic.method='hsic.perm') with ZERO R:

    CI(x, y | S):
      if S empty:   marginal HSIC permutation test on (x, y)
      else:         residualize x and y on S with penalized splines,
                    then marginal HSIC permutation test on the residuals.

The marginal HSIC half is gpu_hsic.hsic_perm_gpu (validated to 0.01% vs
kpcalg). This module adds the residualization half, replicating kpcalg's
regrXonS, which fits:
      |S| = 1 : x_i ~ s(x_a)                 (univariate smooth)
      |S| = 2 : x_i ~ s(x_a, x_b)            (2-D interaction smooth)
      |S| >= 3: x_i ~ s(x_a)+s(x_b)+...      (additive smooths)
via mgcv thin-plate splines + GCV. We use penalized cubic B-splines
(P-splines) with a 2nd-order difference penalty and GCV smoothing-parameter
selection -- the same statistical operation. Parity is validated empirically
against regrXonS by residual agreement and, crucially, by agreement of the
downstream conditional-HSIC decision (see hsic_validate_conditional.py).

All linear algebra is float64 on CUDA.
"""
from __future__ import annotations
import numpy as np
import torch
from scipy.interpolate import BSpline

from gpu_hsic import hsic_perm_gpu, _rbf_centered  # noqa: F401


# ----------------------------- spline basis -------------------------------- #
def _bspline_basis(v: np.ndarray, k: int = 10, degree: int = 3) -> np.ndarray:
    """Cubic B-spline basis (n x k) for a 1-D covariate, quantile knots."""
    v = np.asarray(v, dtype=np.float64)
    lo, hi = v.min(), v.max()
    if hi <= lo:
        return np.ones((len(v), 1))
    n_int = k - degree                                  # interior+boundary segments
    qs = np.linspace(0, 1, n_int + 1)
    inner = np.quantile(v, qs)
    inner[0], inner[-1] = lo, hi
    # pad boundary knots
    t = np.concatenate([[lo] * degree, inner, [hi] * degree])
    n_basis = len(t) - degree - 1
    B = np.empty((len(v), n_basis), dtype=np.float64)
    eye = np.eye(n_basis)
    vv = np.clip(v, lo, hi)
    for j in range(n_basis):
        B[:, j] = BSpline(t, eye[j], degree, extrapolate=True)(vv)
    return B


def _diff_penalty(q: int, order: int = 2) -> np.ndarray:
    """(D^T D) 2nd-order difference penalty for q coefficients."""
    D = np.diff(np.eye(q), n=order, axis=0)
    return D.T @ D


def _center(B: np.ndarray) -> np.ndarray:
    """Sum-to-zero: subtract column means (identifiability vs global intercept)."""
    return B - B.mean(axis=0, keepdims=True)


def _build_design(S_cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (design X with leading intercept, penalty matrix P) for regrXonS.

    S_cols: (n, d) conditioning covariates. Mirrors kpcalg's rule:
      d==1 -> univariate; d==2 -> tensor-product interaction; d>=3 -> additive.
    """
    n, d = S_cols.shape
    blocks, pen_blocks = [np.ones((n, 1))], [np.zeros((1, 1))]  # unpenalized intercept

    if d == 1:
        B = _center(_bspline_basis(S_cols[:, 0], k=10))
        blocks.append(B); pen_blocks.append(_diff_penalty(B.shape[1]))

    elif d == 2:
        k = 6
        B1 = _bspline_basis(S_cols[:, 0], k=k)
        B2 = _bspline_basis(S_cols[:, 1], k=k)
        q1, q2 = B1.shape[1], B2.shape[1]
        # row-wise tensor product -> (n, q1*q2)
        T = (B1[:, :, None] * B2[:, None, :]).reshape(n, q1 * q2)
        T = _center(T)
        P1, P2 = _diff_penalty(q1), _diff_penalty(q2)
        P = np.kron(P1, np.eye(q2)) + np.kron(np.eye(q1), P2)
        blocks.append(T); pen_blocks.append(P)

    else:  # additive
        for c in range(d):
            B = _center(_bspline_basis(S_cols[:, c], k=10))
            blocks.append(B); pen_blocks.append(_diff_penalty(B.shape[1]))

    X = np.concatenate(blocks, axis=1)
    # block-diagonal penalty
    sizes = [b.shape[1] for b in blocks]
    P = np.zeros((X.shape[1], X.shape[1]))
    off = 0
    for pb, s in zip(pen_blocks, sizes):
        P[off:off + s, off:off + s] = pb
        off += s
    return X, P


# --------------------------- penalized GCV fit ----------------------------- #
def _gcv_residual(target: torch.Tensor, X: torch.Tensor, P: torch.Tensor,
                  n_lam: int = 40) -> torch.Tensor:
    """Residual of `target` after penalized-spline fit on X, GCV-selected lambda."""
    n, q = X.shape
    XtX = X.T @ X
    Xty = X.T @ target
    I = torch.eye(q, dtype=X.dtype, device=X.device)
    lams = torch.logspace(-6, 6, n_lam, dtype=X.dtype, device=X.device)
    best_gcv, best_res = torch.tensor(float('inf'), device=X.device), None
    ridge = 1e-10 * I
    for lam in lams:
        A = XtX + lam * P + ridge
        beta = torch.linalg.solve(A, Xty)
        fitted = X @ beta
        rss = torch.sum((target - fitted) ** 2)
        # effective df = tr(X A^{-1} X^T) = tr(A^{-1} XtX)
        edf = torch.trace(torch.linalg.solve(A, XtX))
        denom = (n - edf)
        gcv = n * rss / (denom * denom)
        if gcv < best_gcv:
            best_gcv, best_res = gcv, target - fitted
    return best_res


def regr_x_on_s_gpu(xy: np.ndarray, S_cols: np.ndarray,
                    device: str = 'cuda') -> tuple[torch.Tensor, torch.Tensor]:
    """GPU analog of kpcalg::regrXonS. xy: (n,2) [x,y]; S_cols: (n,d)."""
    X_np, P_np = _build_design(S_cols)
    X = torch.as_tensor(X_np, dtype=torch.float64, device=device)
    P = torch.as_tensor(P_np, dtype=torch.float64, device=device)
    tx = torch.as_tensor(xy[:, 0], dtype=torch.float64, device=device)
    ty = torch.as_tensor(xy[:, 1], dtype=torch.float64, device=device)
    resx = _gcv_residual(tx, X, P)
    resy = _gcv_residual(ty, X, P)
    return resx, resy


# ------------------------- conditional HSIC test --------------------------- #
def hsic_ci_gpu(data: np.ndarray, i: int, j: int, S: list[int],
                sig: float = 1.0, p: int = 100, device: str = 'cuda',
                generator: torch.Generator | None = None) -> float:
    """p-value of CI(i, j | S) on `data` (n x p_vars). Mirrors kernelCItest hsic.perm."""
    x = data[:, i]; y = data[:, j]
    if len(S) == 0:
        return hsic_perm_gpu(x, y, sig=sig, p=p, device=device,
                             generator=generator)['p_value']
    S_cols = data[:, list(S)]
    resx, resy = regr_x_on_s_gpu(np.column_stack([x, y]), S_cols, device=device)
    return hsic_perm_gpu(resx, resy, sig=sig, p=p, device=device,
                         generator=generator)['p_value']
