"""
kernel_granger_baseline.py

Nonlinear Granger causality via kernel method (Marinazzo, Pellicoro & Stramaglia,
PRL 2008: "Kernel method for nonlinear Granger causality").

Reference:
  Marinazzo, D., Pellicoro, M., & Stramaglia, S. (2008). Kernel method for
  nonlinear Granger causality. Physical Review Letters, 100, 144103.

Method (pairwise):
  For each ordered pair (i, j):
    Fit kernel ridge:
      F1: X_j(t) = f(X_j(t-1..t-L))                       (restricted)
      F2: X_j(t) = f(X_j(t-1..t-L), X_i(t-1..t-L))        (unrestricted)
    Compute F-statistic:
      F = ((RSS1 - RSS2) / L) / (RSS2 / (N - 2L - 1))
    Test with F(L, N - 2L - 1). Reject H0 (no causality) at alpha -> A[i,j] = 1.

  Rejection means X_i's past helps predict X_j beyond X_j's own past.

Interface:
    run_kernel_granger(X, max_lag=2, alpha=0.05, gamma=None)

    Returns (p, p) int adjacency matrix; A[i,j]=1 iff X_i Granger-causes X_j.
    Diagonal always zero.
"""
from __future__ import annotations

import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from scipy.stats import f as f_dist


def _make_lagged(x: np.ndarray, lags: list[int]) -> np.ndarray:
    """Return (N, len(lags)) matrix of lagged copies of a 1D series."""
    T = len(x)
    L = max(lags)
    N = T - L
    out = np.empty((N, len(lags)))
    for i, l in enumerate(lags):
        out[:, i] = x[L - l : T - l]
    return out


def _rff_ridge_rss(X_feat: np.ndarray, y: np.ndarray,
                   gamma: float, n_components: int = 100,
                   ridge_alpha: float = 1.0,
                   seed: int = 0) -> float:
    """
    RSS of an RBF-approximated ridge regression via random Fourier features.

    Keeps the model complexity bounded (n_components basis) so the restricted
    and unrestricted fits are comparable in a large-T regime.
    """
    rff = RBFSampler(gamma=gamma, n_components=n_components,
                     random_state=seed)
    Phi = rff.fit_transform(X_feat)
    ridge = Ridge(alpha=ridge_alpha)
    ridge.fit(Phi, y)
    y_pred = ridge.predict(Phi)
    return float(np.sum((y - y_pred) ** 2))


def run_kernel_granger(
    X: np.ndarray,
    max_lag: int = 2,
    alpha: float = 0.05,
    gamma: float | None = None,
) -> np.ndarray:
    """
    Run pairwise kernel Granger on multivariate time series X (p, T).

    Parameters
    ----------
    X : (p, T) array
    max_lag : lag order L
    alpha : significance level for the F-test
    gamma : RBF kernel bandwidth. If None, uses 1/median-heuristic on X_j.

    Returns
    -------
    (p, p) int adjacency matrix.
    """
    p, T = X.shape
    L = max_lag
    N = T - L
    N_COMP = 100  # RFF basis size
    df_num = N_COMP  # unrestricted has N_COMP extra effective features
    df_den = N - 2 * N_COMP - 1
    if df_den <= 0:
        return np.zeros((p, p), dtype=int)
    f_crit = f_dist.ppf(1 - alpha, df_num, df_den)

    adj = np.zeros((p, p), dtype=int)
    lags = list(range(1, L + 1))

    for j in range(p):
        y = X[j, L:]
        X_j_lag = _make_lagged(X[j], lags)  # (N, L)

        # Auto gamma: median-heuristic on X_j past
        if gamma is None:
            n_sub = min(200, X_j_lag.shape[0])
            idx = np.linspace(0, X_j_lag.shape[0] - 1, n_sub, dtype=int)
            D = X_j_lag[idx]
            sq = np.sum(D**2, axis=1)
            pair = sq[:, None] + sq[None, :] - 2 * D @ D.T
            med = np.median(pair[pair > 0]) if np.any(pair > 0) else 1.0
            g = 1.0 / (2.0 * med)
        else:
            g = gamma

        rss1 = _rff_ridge_rss(X_j_lag, y, g, n_components=N_COMP, seed=j)

        for i in range(p):
            if i == j:
                continue
            X_i_lag = _make_lagged(X[i], lags)
            X_ij = np.hstack([X_j_lag, X_i_lag])
            rss2 = _rff_ridge_rss(X_ij, y, g, n_components=N_COMP, seed=i * p + j)
            if rss2 <= 0 or rss1 <= rss2:
                F = 0.0
            else:
                F = ((rss1 - rss2) / df_num) / (rss2 / df_den)
            if F > f_crit:
                adj[i, j] = 1

    np.fill_diagonal(adj, 0)
    return adj
