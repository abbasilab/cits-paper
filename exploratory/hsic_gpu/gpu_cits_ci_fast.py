"""
gpu_cits_ci_fast.py

Optimized full-GPU conditional-independence test for CITS, algorithmically
identical to gpu_cits_ci.hsic_ci_gpu but much faster per test:

  1. Per-seed cache of each variable's centered cubic B-spline basis, so scipy
     never runs inside the CI-test hot loop.
  2. Demmler-Reinsch eigen-form of the penalized fit: the GCV smoothing-parameter
     sweep becomes a vectorized closed-form over a lambda grid (no per-lambda
     linear solves). For univariate smooths (|S|=1) the whole eigen structure is
     cached per column and reused across every test that conditions on it.
  3. Same validated GPU HSIC permutation test (gpu_hsic.hsic_perm_gpu).

Must reproduce gpu_cits_ci's decisions (revalidated by hsic_validate_conditional.py
pointed at this module). float64 / CUDA throughout.
"""
from __future__ import annotations
import numpy as np
import torch
from scipy.interpolate import BSpline

from gpu_hsic import hsic_perm_gpu

_DT = torch.float64


# ------------------------- basis / penalty (numpy) ------------------------- #
def _bspline_basis(v: np.ndarray, k: int = 10, degree: int = 3) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        return np.ones((len(v), 1))
    qs = np.linspace(0, 1, (k - degree) + 1)
    inner = np.quantile(v, qs); inner[0], inner[-1] = lo, hi
    t = np.concatenate([[lo] * degree, inner, [hi] * degree])
    nb = len(t) - degree - 1
    eye = np.eye(nb); vv = np.clip(v, lo, hi)
    return np.stack([BSpline(t, eye[j], degree, extrapolate=True)(vv)
                     for j in range(nb)], axis=1)


def _diff_penalty(q: int, order: int = 2) -> np.ndarray:
    D = np.diff(np.eye(q), n=order, axis=0)
    return D.T @ D


class GPUCondIndTester:
    """Holds one seed's data; answers CI(i,j|S) p-values fast."""

    def __init__(self, data_samples_x_vars: np.ndarray, sig: float = 1.0,
                 p: int = 100, n_lam: int = 40, device: str = 'cuda',
                 generator: torch.Generator | None = None):
        self.data = np.asarray(data_samples_x_vars, dtype=np.float64)
        self.n, self.pvars = self.data.shape
        self.sig, self.p, self.device = sig, p, device
        self.gen = generator
        self.lams = torch.logspace(-6, 6, n_lam, dtype=_DT, device=device)
        self._col_basis: dict[int, np.ndarray] = {}          # col -> centered B (np)
        self._uni_prep: dict[int, tuple] = {}                # col -> eigen prep
        self._t = {c: torch.as_tensor(self.data[:, c], dtype=_DT, device=device)
                   for c in range(self.pvars)}

    # ---- basis cache ----
    def _basis(self, col: int, k: int = 10) -> np.ndarray:
        if col not in self._col_basis:
            B = _bspline_basis(self.data[:, col], k=k)
            self._col_basis[col] = B - B.mean(0, keepdims=True)
        return self._col_basis[col]

    # ---- eigen prep for a design (X, P) ----
    def _prep(self, X_np: np.ndarray, P_np: np.ndarray) -> tuple:
        X = torch.as_tensor(X_np, dtype=_DT, device=self.device)
        P = torch.as_tensor(P_np, dtype=_DT, device=self.device)
        M = X.T @ X + 1e-8 * torch.eye(X.shape[1], dtype=_DT, device=self.device)
        L = torch.linalg.cholesky(M)
        Linv = torch.linalg.solve_triangular(
            L, torch.eye(X.shape[1], dtype=_DT, device=self.device), upper=False)
        Pt = Linv @ P @ Linv.T
        d, U = torch.linalg.eigh(0.5 * (Pt + Pt.T))
        d = torch.clamp(d, min=0.0)
        return X, L, U, d

    def _design(self, S: list[int]) -> tuple:
        n, d = self.n, len(S)
        blocks = [np.ones((n, 1))]; pens = [np.zeros((1, 1))]
        if d == 1:
            B = self._basis(S[0]); blocks.append(B); pens.append(_diff_penalty(B.shape[1]))
        elif d == 2:
            k = 6
            B1 = _bspline_basis(self.data[:, S[0]], k=k)
            B2 = _bspline_basis(self.data[:, S[1]], k=k)
            q1, q2 = B1.shape[1], B2.shape[1]
            Tt = (B1[:, :, None] * B2[:, None, :]).reshape(n, q1 * q2)
            Tt = Tt - Tt.mean(0, keepdims=True)
            P = np.kron(_diff_penalty(q1), np.eye(q2)) + np.kron(np.eye(q1), _diff_penalty(q2))
            blocks.append(Tt); pens.append(P)
        else:
            for c in S:
                B = self._basis(c); blocks.append(B); pens.append(_diff_penalty(B.shape[1]))
        X = np.concatenate(blocks, axis=1)
        sizes = [b.shape[1] for b in blocks]
        P = np.zeros((X.shape[1], X.shape[1])); off = 0
        for pb, s in zip(pens, sizes):
            P[off:off + s, off:off + s] = pb; off += s
        return X, P

    def _prep_for(self, S: list[int]) -> tuple:
        if len(S) == 1:
            col = S[0]
            if col not in self._uni_prep:
                Xn, Pn = self._design(S)
                self._uni_prep[col] = self._prep(Xn, Pn)
            return self._uni_prep[col]
        Xn, Pn = self._design(S)
        return self._prep(Xn, Pn)

    # ---- residual via Demmler-Reinsch eigen GCV ----
    def _residual(self, y: torch.Tensor, prep: tuple) -> torch.Tensor:
        X, L, U, d = prep
        Xty = X.T @ y
        a = torch.linalg.solve_triangular(L, Xty.unsqueeze(1), upper=False).squeeze(1)
        c = U.T @ a
        yy = torch.dot(y, y)
        denom = 1.0 / (1.0 + self.lams[:, None] * d[None, :])   # (n_lam, q)
        c2 = c * c
        yf = (c2[None, :] * denom).sum(1)
        ff = (c2[None, :] * denom * denom).sum(1)
        edf = denom.sum(1)
        rss = yy - 2 * yf + ff
        gcv = self.n * rss / (self.n - edf) ** 2
        k = int(torch.argmin(gcv).item())
        dsel = denom[k]                                          # (q,)
        rhs = (U @ (dsel * c)).unsqueeze(1)
        beta = torch.linalg.solve_triangular(L.T, rhs, upper=True).squeeze(1)
        return y - X @ beta

    def pval(self, i: int, j: int, S) -> float:
        S = sorted(int(s) for s in S)
        if len(S) == 0:
            return hsic_perm_gpu(self._t[i], self._t[j], sig=self.sig, p=self.p,
                                 device=self.device, generator=self.gen)['p_value']
        prep = self._prep_for(S)
        rx = self._residual(self._t[i], prep)
        ry = self._residual(self._t[j], prep)
        return hsic_perm_gpu(rx, ry, sig=self.sig, p=self.p,
                             device=self.device, generator=self.gen)['p_value']
