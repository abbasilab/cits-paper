"""
cits_optimal.py

CITS+ v_latest: optimal 2τ+1-window variant of CITS.

KEY IDEA (Markov-blanket-reach argument)
-----------------------------------------
CITS's default `data_transform` uses window width w = 2(τ+1) (= 4 for τ=1).
That is one time-slice more than required.

For a lag-τ edge X_i(t-τ) → X_j(t):
  - Source X_i(t-τ)'s Markov blanket reaches back to t-2τ.
  - Target X_j(t)'s Markov blanket reaches back to t-τ.
  - Combined window spans t-2τ ... t, which is 2τ+1 time slices.

For a contemporaneous edge X_i(t) — X_j(t):
  - Both blankets reach back to t-τ.
  - Window spans t-τ ... t, which is τ+1 time slices (strict subset of above).

Therefore 2τ+1 slices are sufficient and 2(τ+1) = 2τ+2 wastes one slice.

With window width w = 2τ+1:
  - Number of independent chi samples: N = T // (2τ+1)
  - For τ=1, T=1000: N = 333  vs  N = 250 for original CITS
  - Sample efficiency gain: 33%  (T/(2τ+1) vs T/(2τ+2))

Window layout for τ=1 (w=3):
  Slice index within window:  0         1           2
  Time labels:                t-2τ=t-2   t-τ=t-1    t (target)
  In chi column i:            X_{t-2}    X_{t-1}    X_t

  CITS tests edges t1 in {t-τ, ..., t-1} = {t-1} → t  (lag-1 to target).
  For τ>1 it tests t1 in {t-τ, ..., t-1} (τ lags).

API
---
data_transform_optimal(X, tau)
    Mimics cits.methods.data_transform but with stride/width = 2τ+1.
    Input:  X  (p, T) float  -- as in the upstream CITS convention.
    Output: chi  (p*(2τ+1), N)  where N = T // (2τ+1).

cits_full_optimal(X, tau, alpha=0.05)
    Full CITS algorithm on the optimal chi.
    Input:  X  (p, T) float.
    Output: B  (p, p) int  -- rolled adjacency matrix (same convention as
            cits.methods.cits_full).
    Indexing (tau=1):
      - chi feature layout: slice 0 = X_{t-2}, slice 1 = X_{t-1}, slice 2 = X_t
      - Target time index: t_target = 2*tau  (= 2 for tau=1)
      - Source time indices tested: t1 in range(tau, 2*tau)  (= {1} for tau=1)
      - Conditioning iterates over ALL p*(2τ+1) chi variables EXCEPT the
        two being tested.
"""

from __future__ import annotations

import numpy as np
from itertools import chain, combinations
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Utilities (replicated from cits.methods to avoid modifying upstream)
# ---------------------------------------------------------------------------

def _powerset(iterable):
    """powerset([1,2,3]) = (), (1,), (2,), ..., (1,2,3)"""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def _partial_corr(A_idx: int, B_idx: int, S: set, data: np.ndarray) -> float:
    """Partial correlation of data[A_idx] and data[B_idx] given data[S].

    data has shape (n_vars, n_samples).  Mirrors cits.methods.partial_corr.
    """
    from scipy import linalg

    p = data.shape[0]
    idx = np.zeros(p, dtype=bool)
    for i in S:
        idx[i] = True

    C = data
    Z = C[idx, :].T                                   # (n_samples, |S|)
    Z = np.column_stack([np.ones(Z.shape[0]), Z])      # add intercept

    try:
        beta_A = linalg.lstsq(Z, C[A_idx, :].T)[0]
        beta_B = linalg.lstsq(Z, C[B_idx, :].T)[0]
    except Exception:
        return 0.0

    res_A = C[A_idx, :].T - Z.dot(beta_A)
    res_B = C[B_idx, :].T - Z.dot(beta_B)

    if res_A.std() < 1e-12 or res_B.std() < 1e-12:
        return 0.0

    r = np.corrcoef(res_A, res_B)[0, 1]
    return float(r)


def _cond_dep_pcorr(chi: np.ndarray, i1: tuple, j1: tuple, k,
                    p: int, alpha: float = 0.05) -> int:
    """Conditional dependence test using partial correlation.

    Mirrors cits.methods.cond_dep_pcorr but uses the optimal chi layout.

    Parameters
    ----------
    chi : (p*(2τ+1), N)
    i1  : (v1, t1)  -- (variable index, time-slice index in chi)
    j1  : (v2, t2)
    k   : tuple/list of flat chi-row indices to condition on
    p   : number of neurons
    alpha : significance level

    Returns 1 if conditionally dependent, 0 otherwise.
    """
    (v1, t1) = i1
    (v2, t2) = j1
    i = t1 * p + v1
    j = t2 * p + v2

    # Materialise k as a tuple so we can measure length and pass to _partial_corr.
    k_tuple = tuple(k)
    r = _partial_corr(i, j, set(k_tuple), chi)

    if abs(r) == 1.0:
        pval = 0.0
    else:
        z = 0.5 * np.log((1.0 + r) / (1.0 - r + 1e-15))
        T_stat = np.sqrt(chi.shape[1] - len(k_tuple) - 3) * abs(z)
        pval = 2.0 * (1.0 - scipy_stats.norm.cdf(T_stat))

    return 1 if pval <= alpha else 0


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def data_transform_optimal(X: np.ndarray, tau: int) -> np.ndarray:
    """Build the optimal chi matrix with window width w = 2τ+1.

    Parameters
    ----------
    X   : (p, T) float  -- time series, p variables, T time points.
    tau : int            -- Markovian order.

    Returns
    -------
    chi : (p*(2*tau+1), N) float
        Time-windowed samples.  N = T // (2*tau+1).
        For τ=1: chi has 3*p rows and N = T // 3 samples.
        Column i corresponds to time window [w*i, w*(i+1)) where w = 2τ+1.
        Within a column, row layout is:
          slices 0, 1, ..., 2τ  stacked vertically, each of length p.
          Slice 0 = X_{t-2τ}, ..., slice 2τ = X_t (target).

    Notes
    -----
    Compared to cits.methods.data_transform (width = 2(τ+1)):
      - Width reduced from 2τ+2 to 2τ+1  (saves 1 slice per window).
      - N increased from T//(2τ+2) to T//(2τ+1)  (+33% for τ=1).
    """
    p = X.shape[0]
    T = X.shape[1]
    w = 2 * tau + 1          # optimal window width (also the stride)
    N = T // w

    chi = np.zeros((p * w, N))
    for i in range(N):
        # Flatten the p × w block starting at column w*i, in row-major
        # order so that chi[:p, i] = X[:, w*i], chi[p:2p, i] = X[:, w*i+1], ...
        chi[:, i] = X[:, w * i: w * (i + 1)].T.reshape(p * w)

    return chi


def cits_unrolled_optimal(X: np.ndarray, tau: int,
                          alpha: float = 0.05) -> np.ndarray:
    """Compute the unrolled adjacency using the optimal 2τ+1 window.

    Parameters
    ----------
    X     : (p, T) float
    tau   : int
    alpha : float

    Returns
    -------
    A : (p*(2τ+1), p*(2τ+1)) int
        Unrolled directed adjacency. Entry A[t1*p+v1, t*p+v2] = 1 means
        X_{v1}(t1) -> X_{v2}(t).  t_target = 2*tau (last slice index).

    Algorithm
    ---------
    Mirrors cits.methods.cits_unrolled but on the smaller 2τ+1-wide chi.
    The time-slice indices within the window are 0 .. 2τ.
    t_target = 2*tau  (the rightmost slice, was 2*tau+1 in the original).
    Source slices tested: t1 in range(tau, 2*tau)
                          = {tau, tau+1, ..., 2*tau-1}
                          = {1} for tau=1.
    Conditioning: all chi variables EXCEPT the pair being tested.
    """
    p = X.shape[0]
    w = 2 * tau + 1          # window width
    n_feat = p * w           # total features in chi

    chi = data_transform_optimal(X, tau)

    # Initialise acyclicity: A[j, i] = 0 if time-slice(j) >= time-slice(i)
    # (no future -> past edge).
    A = np.zeros((n_feat, n_feat), dtype=int)

    t_target = 2 * tau       # target slice index (rightmost)

    # We only care about edges t1 -> t_target for t1 in [tau, 2*tau-1].
    # But to mirror the full unrolled init, also zero out back-in-time edges.
    for v in range(p):
        for t in range(w):
            for v1 in range(p):
                for t1 in range(t, w):
                    # t1 >= t means t1 is same or later slice, so no causal
                    # edge from later/same slice back to t (causal direction).
                    A[t1 * p + v1, t * p + v] = 0

    # Initialise the edges we WILL test as present.
    for v in range(p):
        for v1 in range(p):
            for t1 in range(tau, 2 * tau):
                # t1 in {tau, ..., 2*tau-1}; target always t_target = 2*tau.
                A[t1 * p + v1, t_target * p + v] = 1

    # CI tests: for each candidate edge, search for a separating set.
    for v in range(p):
        for v1 in range(p):
            for t1 in range(tau, 2 * tau):
                i_idx = t_target * p + v      # flat index of target node
                j_idx = t1 * p + v1           # flat index of source node
                # Condition on everything except these two nodes.
                cond_candidates = [
                    k for k in range(n_feat)
                    if k != i_idx and k != j_idx
                ]
                for k in _powerset(cond_candidates):
                    if _cond_dep_pcorr(
                            chi,
                            (v,  t_target),
                            (v1, t1),
                            k,
                            p,
                            alpha) == 0:
                        A[j_idx, i_idx] = 0
                        break

    return A


def cits_rolled_optimal(A: np.ndarray, p: int, tau: int) -> np.ndarray:
    """Roll the optimal unrolled adjacency into the p×p summary graph.

    Parameters
    ----------
    A   : (p*(2τ+1), p*(2τ+1)) int
    p   : int  -- number of neurons
    tau : int

    Returns
    -------
    B : (p, p) int  -- same convention as cits.methods.cits_rolled.
    """
    B = np.zeros((p, p), dtype=int)
    t_target = 2 * tau       # rightmost slice in the optimal window

    for v1 in range(p):
        for v2 in range(p):
            for t1 in range(tau, 2 * tau):   # source slices: τ to 2τ-1
                if A[t1 * p + v1, t_target * p + v2] != 0:
                    B[v1, v2] = 1
    return B


def cits_full_optimal(X: np.ndarray, tau: int,
                      alpha: float = 0.05) -> np.ndarray:
    """CITS algorithm with optimal 2τ+1 window.

    Parameters
    ----------
    X     : (p, T) float  -- time series (p variables, T time points).
    tau   : int            -- Markovian order.
    alpha : float          -- significance level for CI tests.

    Returns
    -------
    B : (p, p) int
        Rolled adjacency matrix.  B[i, j] = 1 means lag-τ edge i → j.
        Same shape and convention as cits.methods.cits_full.

    Notes
    -----
    For τ=1, T=1000:
      - Original CITS uses N = (T-4)/4 = 249 chi samples.
      - cits_full_optimal uses N = T//3 = 333 chi samples.
      - 33% more effective independent observations.
    """
    A = cits_unrolled_optimal(X, tau, alpha)
    B = cits_rolled_optimal(A, X.shape[0], tau)
    return B
