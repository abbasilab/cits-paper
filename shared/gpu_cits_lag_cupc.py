"""
gpu_cits_lag_cupc.py

============================ DEPRECATED -- DO NOT USE ============================
Superseded by gpu_cits_lag_cupc_faithful.py. Kept for provenance, not deleted.

This early driver does NOT faithfully reproduce CITS: it builds OVERLAPPING
windows of width 2*tau+1 (CITS uses NON-overlapping width 2*(tau+1)) and runs a
full cuPC PC skeleton over all pairs before extracting lag edges. The claim in
the original description below -- "the same CITS-lag algorithm ... No algorithmic
change" -- is INCORRECT. Use gpu_cits_lag_cupc_faithful.py, which matches exact
CITS on windowing, target parent set, conditioning pool, and time orientation
(parity-validated: >=98% edge agreement, matched Combined Scores), differing only
by the sound powerset -> neighbour-restricted PC conditioning reduction.
=================================================================================

--- original (inaccurate) description, retained for record ---
CITS-lag with cuPC-accelerated Fisher-z partial-correlation CI tests.

This is the *same* CITS-lag algorithm as :func:`cits.methods.cits_full` with
``cond_dep='cond_dep_pcorr'`` (partial-correlation Fisher-z test), just wired
to run its many CI tests in parallel on GPU via cuPC. No algorithmic change.

Algorithm (unchanged from CITS paper):
  1. Build chi-stacked time-windowed samples of width w = 2*tau + 1.
     For each of the (T - w + 1) starting positions we obtain one sample of
     the "unrolled" random vector with w * p coordinates.
  2. On the unrolled representation, run the PC skeleton (Fisher-z partial
     correlation) with iterative-by-size conditioning-set search. This is
     what cuPC does in parallel on GPU.
  3. Extract lag edges: for each (v1, v2) and each source time index
     t1 in {tau, ..., 2*tau-1}, an edge X_{v1}(t1) -> X_{v2}(2*tau) survives
     iff the corresponding pair in the unrolled skeleton is retained.
  4. Aggregate to the rolled adjacency: an edge v1 -> v2 exists iff any of
     the tested lags is retained.

The rolled adjacency is exactly what CITS-lag returns. The only differences
from cits.methods.cits_full with pcorr are:
  - CI tests run in parallel on GPU (cuPC's contribution)
  - Conditioning-set search is capped at cuPC's compile-time limit (ML = 14)
  - Time complexity per CI test is O(1) in cuPC's batched implementation
    rather than the O(2^k) in the naive powerset traversal
"""
from __future__ import annotations
import os
import sys
import numpy as np

_ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, _ANALYSIS_DIR)

from _cupc_wrapper import pc_skeleton_cupc


def _build_chi_stack(X: np.ndarray, tau: int) -> np.ndarray:
    """Build the chi-stacked unrolled sample matrix for CITS-lag.

    Parameters
    ----------
    X : (p, T) time series (benchmark convention).
    tau : Markov order.

    Returns
    -------
    U : (N, p * (2*tau + 1)) matrix.
        Row k gathers X_{v, k+t} for v = 0..p-1, t = 0..2*tau.
        Column index c = v * (2*tau + 1) + t.
    """
    p, T = X.shape
    w = 2 * tau + 1
    if T < w:
        raise ValueError(f"T={T} too small for tau={tau} (need T >= {w})")
    N = T - w + 1
    U = np.empty((N, p * w), dtype=np.float64)
    for v in range(p):
        for t in range(w):
            U[:, v * w + t] = X[v, t : t + N]
    return U


def gpu_cits_lag_cupc(X: np.ndarray, alpha: float = 0.05, tau: int = 1,
                     max_level: int = 14, verbose: bool = False) -> np.ndarray:
    """Run CITS-lag with cuPC-accelerated partial-correlation CI tests.

    Parameters
    ----------
    X : (p, T) time series.
    alpha : significance level.
    tau : Markov order.
    max_level : conditioning-set-size cap (cuPC compile-time ML = 14).
    verbose : print progress.

    Returns
    -------
    B : (p, p) int adjacency (rolled). B[i, j] = 1 iff some tested lag edge
        X_i(t1) -> X_j(t_target) is retained by the skeleton.
    """
    p, T = X.shape
    w = 2 * tau + 1
    U = _build_chi_stack(X, tau)  # (N, p * w)

    G_unrolled, _sep_sets, _inactive, _lvl = pc_skeleton_cupc(
        U, alpha=alpha, max_level=max_level, verbose=verbose)

    # Extract lag-edges into rolled adjacency.
    # Convention: source time indices t1 in {tau, ..., 2*tau - 1} (i.e. one
    # step earlier for tau=1); target index t_target = 2*tau.
    t_target = 2 * tau
    B = np.zeros((p, p), dtype=int)
    for v1 in range(p):
        for v2 in range(p):
            if v1 == v2:
                continue
            for t1 in range(tau, 2 * tau):  # excludes concurrent (t1 == t_target)
                c1 = v1 * w + t1
                c2 = v2 * w + t_target
                if G_unrolled[c1, c2] != 0:
                    B[v1, v2] = 1
                    break
    return B
