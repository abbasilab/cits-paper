"""
cits_plus_v_latest.py

CITS+ v_latest driver: union of CITS-optimal (2τ+1 window) lagged skeleton
and PC-contemp (2τ+1 chi, tau+1 stacked) skeleton, with v-structure orientation
(no Meek) and LSCM refit.

Mirror of the v2 pipeline (_pc_raw_v2.py + run_cits_plus_v2 in the benchmark),
but both the lagged and contemporaneous steps use the 2τ+1-window optimal chi.

Pipeline per trial:
  1. Run cits_full_optimal(X, tau, alpha) -> lagged rolled adjacency B_lag.
  2. Run pc_skeleton_raw_v_latest(X_T, alpha, tau, backend) -> A_contemp, sep_sets.
  3. Symmetrise sep_sets to lag-0 only (drop lagged conditioning variables)
     for the v-structure orientation phase.
  4. Orient v-structures on A_contemp via orient_v_structures(A_contemp, sep_sets_lag0).
  5. Build union skeleton: CITS-lagged OR PC-contemp (undirected union).
  6. Optionally refit LSCM via _lscm_refit (currently not invoked in the
     benchmark version -- skeleton-level comparison is the primary metric).

Usage (benchmark):
  from cits_plus_v_latest import run_cits_plus_v_latest
  A_union = run_cits_plus_v_latest(X, alpha=0.05)

  X has shape (p, T) -- the benchmark convention (transposed from (T, p)).
"""

from __future__ import annotations
import os
import sys
import warnings
import numpy as np

_ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, _ANALYSIS_DIR)

from cits_optimal import cits_full_optimal
from _pc_raw_v_latest import pc_skeleton_raw_v_latest
from _pc_orientation import orient_v_structures


def run_cits_plus_v_latest(X: np.ndarray, alpha: float = 0.05,
                            tau: int = 1, backend: str = 'python') -> np.ndarray:
    """CITS+ v_latest: union skeleton from 2τ+1-optimal CITS-lag + PC-contemp.

    Parameters
    ----------
    X       : (p, T) float  -- time series, benchmark convention.
    alpha   : float          -- significance level for both CITS and PC steps.
    tau     : int            -- Markovian order (default 1).
    backend : str            -- 'python' (CPU, default for sims) or 'cupc' (GPU).

    Returns
    -------
    union_skel : (p, p) int
        Partially directed union adjacency.
        Lag edges are symmetric (both directions set). Contemporaneous edges
        preserve v-structure orientation: union_skel[i, j] = 1 means i->j
        (directed) or i--j (undirected, both entries set); presence is
        CITS-lagged OR PC-contemporaneous.
    """
    p, T = X.shape

    # Step 1: CITS-lagged with optimal 2τ+1 window.
    try:
        cits_adj = cits_full_optimal(X, tau, alpha)
    except Exception as e:
        warnings.warn(f"cits_full_optimal failed: {e}")
        cits_adj = np.zeros((p, p), dtype=int)
    cits_bin = (np.asarray(cits_adj) != 0).astype(int)
    np.fill_diagonal(cits_bin, 0)

    # Step 2: PC-contemp with optimal 2τ+1 chi (tau+1 stacked window).
    X_T = X.T  # (T, p)
    try:
        A_contemp, r_mat, sep_sets_vl, inactive = pc_skeleton_raw_v_latest(
            X_T, alpha=alpha, tau=tau, backend=backend, verbose=False)
    except Exception as e:
        warnings.warn(f"pc_skeleton_raw_v_latest failed: {e}")
        A_contemp = np.zeros((p, p), dtype=int)
        sep_sets_vl = {}

    # Step 3: Drop lagged conditioning variables from sep_sets for orientation.
    # (Lagged vars are encoded as -(k+1); keep only k >= 0.)
    sep_sets_lag0_only = {}
    for (i, j), S in sep_sets_vl.items():
        sep_sets_lag0_only[(i, j)] = tuple(k for k in S if k >= 0)

    # Step 4: Orient v-structures on the contemp skeleton.
    try:
        G_contemp = orient_v_structures(A_contemp, sep_sets_lag0_only)
    except Exception as e:
        warnings.warn(f"orient_v_structures failed: {e}")
        G_contemp = A_contemp.copy()

    # Step 5: Build union skeleton preserving contemp direction.
    union_skel = np.zeros((p, p), dtype=int)
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            if cits_bin[i, j]:
                union_skel[i, j] = 1
            if G_contemp[i, j]:
                union_skel[i, j] = 1
    np.fill_diagonal(union_skel, 0)

    return union_skel
