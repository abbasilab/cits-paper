"""
cits_lag_rcit.py

Lag-side discovery using RCIT in place of Fisher-z partial correlation.
Mirrors the windowing structure of cits_optimal.cits_full_optimal:

  - Chi-stack width w = 2*tau + 1 (= 3 for tau=1).
  - Target slice index t_target = 2*tau (rightmost), source slices t1 in
    {tau, ..., 2*tau - 1}.
  - For each candidate lag edge X_{v1}(t1) -> X_{v2}(t_target), search for a
    separating set S in the remaining chi features. Cap |S| at max_cond_size
    (default 5) for tractability with RCIT.

Returns a "rolled" (p, p) integer adjacency where B[i, j] = 1 if any tested
lag-edge i -> j was retained.

CPU-only. RCIT runs on raw chi-stacked data (no GPU).
"""

from __future__ import annotations
import os
import sys
import itertools
import numpy as np

_ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, _ANALYSIS_DIR)

from rcit import rcit_test
from cits_optimal import data_transform_optimal


def cits_lag_rcit(X: np.ndarray,
                  alpha: float = 0.05,
                  tau: int = 1,
                  K: int = 25,
                  n_perm: int = 100,
                  max_cond_size: int = 5,
                  seed: int = 0,
                  verbose: bool = False) -> np.ndarray:
    """RCIT-based lag-side discovery on the optimal 2*tau+1 chi.

    Parameters
    ----------
    X : (p, T) float
    alpha : float
    tau : int
    K, n_perm : RCIT hyperparams
    max_cond_size : cap on conditioning-set size (RCIT is expensive)
    seed : RNG seed
    verbose : bool

    Returns
    -------
    B : (p, p) int -- rolled adjacency (B[i, j] = 1 if any lag t1 -> t_target
        edge from i to j survived all RCIT tests).
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be (p, T); got {X.shape}")
    p, T = X.shape

    # Build the optimal chi: (p*(2*tau+1), N) where N = T // (2*tau+1)
    chi = data_transform_optimal(X, tau)  # (n_feat, N)
    n_feat, N_obs = chi.shape
    w = 2 * tau + 1
    assert n_feat == p * w

    # Switch to (N, n_feat) for RCIT (observations as rows)
    C = chi.T  # (N, n_feat)

    if verbose:
        print(f"[cits_lag_rcit] chi shape={chi.shape}; n_feat={n_feat}, "
              f"N={N_obs}", flush=True)

    t_target = 2 * tau          # rightmost slice
    t_source_range = list(range(tau, 2 * tau))   # {1} for tau=1

    # Cache (i, j, S) -> p-value to avoid repeating identical tests for
    # symmetric (i,j) order or repeated S permutations.
    pval_cache: dict = {}

    def _ci_pvalue(i: int, j: int, S: tuple) -> float:
        a, b = (i, j) if i < j else (j, i)
        S_sorted = tuple(sorted(S))
        key = (a, b, S_sorted)
        if key in pval_cache:
            return pval_cache[key]
        local_seed = (seed * 1_000_003
                      + a * 9973
                      + b * 97
                      + sum(S_sorted) * 7
                      + len(S_sorted)) & 0x7FFFFFFF
        x = C[:, i]
        y = C[:, j]
        z = C[:, list(S_sorted)] if S_sorted else None
        pval = rcit_test(x, y, z=z, K=K, n_perm=n_perm, seed=local_seed)
        pval_cache[key] = pval
        return pval

    B = np.zeros((p, p), dtype=int)

    # For each candidate lag edge X_{v1}(t1) -> X_{v2}(t_target),
    # incrementally enlarge the conditioning set drawn from "the other chi
    # variables" until either we find a separator (drop edge) or we hit
    # max_cond_size (keep edge).
    for v2 in range(p):
        for v1 in range(p):
            # No self-edges in the rolled adjacency? CITS allows self loops
            # at lag-tau. Keep self-test as well (i != j in chi space because
            # different time slices). We'll just allow v1 == v2 here.
            for t1 in t_source_range:
                i_idx = t1 * p + v1           # source chi index
                j_idx = t_target * p + v2     # target chi index
                if i_idx == j_idx:
                    continue

                # Candidate conditioners: everything in chi except {i, j}.
                cand = [k for k in range(n_feat) if k != i_idx and k != j_idx]

                # First: marginal test (l = 0)
                pval = _ci_pvalue(i_idx, j_idx, ())
                if pval > alpha:
                    # marginally independent -> no edge
                    continue

                separated = False
                for l in range(1, max_cond_size + 1):
                    if l > len(cand):
                        break
                    for S_list in itertools.combinations(cand, l):
                        pval = _ci_pvalue(i_idx, j_idx, S_list)
                        if pval > alpha:
                            separated = True
                            break
                    if separated:
                        break

                if not separated:
                    B[v1, v2] = 1
                    # No early-exit across t1 because the rolled cell is
                    # already set; subsequent (v1, t1, v2) tests are skipped
                    # by the loop structure when t_source_range is a singleton.

    if verbose:
        n_edges = int(B.sum())
        print(f"[cits_lag_rcit] retained lag edges = {n_edges}", flush=True)

    return B


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    p, T = 4, 1000
    X = rng.normal(size=(p, T))
    # Inject a lag-1 edge 0 -> 1
    X[1, 1:] += 0.6 * X[0, :-1]
    B = cits_lag_rcit(X, alpha=0.05, tau=1, K=25, n_perm=50, verbose=True)
    print("B (rolled lag adj) =\n", B)
