"""
pc_skeleton_rcit.py

PC skeleton phase using RCIT (Randomized Conditional Independence Test) as the
CI test, in place of Fisher-z. Mirrors the chi-stacked windowing convention of
_pc_raw_v2.pc_skeleton_raw_v2 (stack the τ+1 = 2-slice window [X_{t-1}, X_t]
when tau=1) so that contemporaneous edges are inferred while conditioning on
the immediate past.

Compared to the Fisher-z version:
  - CI tests are kernel-based (HSIC via Random Fourier Features). This is
    sensitive to non-linear and non-Gaussian dependencies.
  - Empty-set ("marginal") tests use rcit_test(x, y).
  - Conditional tests use rcit_test(x, y, z=...).
  - Significantly slower than Fisher-z per test (permutation null with
    n_perm=100 by default).

CPU-only. Uses the inline CPU PC skeleton routine; no cuPC backend.

API
---
pc_skeleton_rcit(X_T, alpha=0.05, tau=1, K=25, n_perm=100,
                 max_cond_size=5, verbose=False, seed=0)

Inputs:
  X_T   : (T, p) float64 raw calcium trace (caller transposed) for one trial.
  alpha : float -- significance level for RCIT.
  tau   : int   -- CITS Markovian order. Window depth = tau+1.
  K     : int   -- RFF feature count per variable.
  n_perm: int   -- permutation count for RCIT null.
  max_cond_size : int -- cap on conditioning-set size (defensive: PC at
                         small p can otherwise enumerate large powersets).
  verbose : bool
  seed    : int -- RNG seed for RCIT (will be perturbed per CI test).

Outputs:
  A_contemp   : (p, p) int symmetric adjacency. Only X_t -- X_t edges.
  sep_sets    : dict[(i, j) -> tuple[int]]. Separating sets, indices in the
                ORIGINAL p-variable space. Lagged conditioning variables are
                represented as -(k+1) (matches _pc_raw_v2 convention).
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
from _pc_raw_v2 import _build_stacked_chi


def pc_skeleton_rcit(X_T: np.ndarray,
                     alpha: float = 0.05,
                     tau: int = 1,
                     K: int = 25,
                     n_perm: int = 100,
                     max_cond_size: int = 5,
                     verbose: bool = False,
                     seed: int = 0,
                     zero_var_tol: float = 1e-12,
                     restrict_self_past_contemp: bool = False):
    """RCIT-based PC skeleton on chi-stacked [X_{t-1}, X_t] window.

    Parameters
    ----------
    restrict_self_past_contemp : bool, default False
        If True, when testing a contemporaneous edge X_i(t) -- X_j(t)
        (both stacked indices >= p), exclude {X_i(t-k), X_j(t-k) for k>=1}
        from the conditioning-sepset candidates. Other neurons' past variables
        remain eligible. This fixes CTRNN-style multicollinearity (a neuron's
        own immediate past is nearly collinear with itself at t and forces
        spurious removal of contemporaneous edges) while preserving
        diamond-DAG blocking (other neurons' pasts can still block common
        parents). The exclusion applies ONLY to contemp-endpoint pairs;
        lag-side tests (one or both endpoints at < p) are untouched.

    Returns
    -------
    A_contemp : (p, p) int symmetric
    sep_sets  : dict {(i, j) -> tuple in original-p space using v2 convention}
    """
    X_T = np.asarray(X_T, dtype=np.float64)
    if X_T.ndim != 2:
        raise ValueError(f"X_T must be 2D (T, p); got shape {X_T.shape}")
    T, p = X_T.shape

    # Zero-variance guard (mirrors _pc_raw_v2)
    col_var = X_T.var(axis=0)
    inactive_mask = col_var < zero_var_tol
    inactive_neurons = np.flatnonzero(inactive_mask)
    if inactive_neurons.size > 0:
        rng = np.random.default_rng(seed)
        X_T = X_T.copy()
        for j in inactive_neurons:
            X_T[:, j] = rng.standard_normal(T) * 1e-10

    # Build stacked chi window [X_{t-1}, X_t] -> shape (2p, N)
    stacked = _build_stacked_chi(X_T, tau, verbose=verbose)
    two_p, N = stacked.shape
    if two_p != 2 * p:
        raise RuntimeError(f"Expected stacked rows = 2p = {2*p}, got {two_p}")

    # PC expects observations as ROWS, variables as COLUMNS.
    C = stacked.T  # (N, 2p)

    if verbose:
        print(f"[pc_skeleton_rcit] stacked C shape={C.shape}; running RCIT PC",
              flush=True)

    A_full, sep_sets_full = _pc_skeleton_cpu_rcit(
        C, alpha=alpha, K=K, n_perm=n_perm,
        max_cond_size=max_cond_size, seed=seed, verbose=verbose,
        p_contemp=p if restrict_self_past_contemp else None)

    # Extract contemp block: rows/cols p..2p-1
    A_contemp = A_full[p:2*p, p:2*p].copy()
    np.fill_diagonal(A_contemp, 0)
    # Symmetrize defensively
    A_contemp = ((A_contemp + A_contemp.T) > 0).astype(int)
    np.fill_diagonal(A_contemp, 0)

    # Zero out edges incident on inactive neurons
    if inactive_neurons.size > 0:
        for j in inactive_neurons:
            A_contemp[j, :] = 0
            A_contemp[:, j] = 0

    # Remap sep sets: stacked indices 0..p-1 are X_{t-1} (lagged),
    # p..2p-1 are X_t. Use v2 convention (lagged -> -(k+1), contemp -> k-p).
    sep_sets_v2 = {}
    for (si, sj), S in sep_sets_full.items():
        if si < p or sj < p:
            continue  # only keep contemp-block pairs
        orig_i = si - p
        orig_j = sj - p
        remapped_S = tuple(
            (k - p) if k >= p else -(k + 1)
            for k in S
        )
        sep_sets_v2[(orig_i, orig_j)] = remapped_S
        sep_sets_v2[(orig_j, orig_i)] = remapped_S

    if verbose:
        n_edges = int(A_contemp.sum() // 2)
        print(f"[pc_skeleton_rcit] contemp edges = {n_edges}", flush=True)

    return A_contemp, sep_sets_v2


# ============================================================================
#  CPU PC skeleton with RCIT as the CI test
# ============================================================================

def _pc_skeleton_cpu_rcit(C: np.ndarray,
                           alpha: float = 0.05,
                           K: int = 25,
                           n_perm: int = 100,
                           max_cond_size: int = 5,
                           seed: int = 0,
                           verbose: bool = False,
                           p_contemp: int | None = None):
    """Plain CPU PC skeleton phase using RCIT.

    Parameters
    ----------
    C : (N, V) float
        Observations as rows, variables as columns.
    alpha : float
        Significance level for RCIT.
    K, n_perm : int
        RCIT hyperparameters.
    max_cond_size : int
        Hard cap on conditioning set size (defensive against runaway PC at
        small V with high alpha).
    p_contemp : int or None
        If not None, V is interpreted as stacked layout 2*p_contemp with
        rows 0..p-1 = X_{t-1} (past) and rows p..2p-1 = X_t (contemp). When
        an edge being tested has BOTH endpoints in the contemp block
        (i, j >= p_contemp), the sepset candidate set is filtered to
        exclude the self-past variables {i - p_contemp, j - p_contemp}.
        Lag-edge tests (at least one endpoint in past block) are untouched.

    Returns
    -------
    A : (V, V) int
        Symmetric undirected skeleton.
    sep_sets : dict[(i, j) -> tuple[int]]
    """
    N, V = C.shape
    A = np.ones((V, V), dtype=int)
    np.fill_diagonal(A, 0)
    sep_sets = {}

    def _filter_nbrs_for_contemp(i: int, j: int, nbrs: list) -> list:
        """If both i and j are contemp endpoints, drop their self-past."""
        if p_contemp is None:
            return nbrs
        if i < p_contemp or j < p_contemp:
            return nbrs  # lag-edge test; no filtering
        # Both endpoints in contemp block.
        # Self-past index for stacked-index s is (s - p_contemp).
        forbid = {i - p_contemp, j - p_contemp}
        return [k for k in nbrs if k not in forbid]

    # Cache rcit p-values so we don't repeat identical tests when scanning
    # both endpoints of an edge in the same depth-l pass.
    pval_cache: dict = {}

    def _ci_pvalue(i: int, j: int, S: tuple) -> float:
        # Canonicalize key
        if i > j:
            i, j = j, i
        S_sorted = tuple(sorted(S))
        key = (i, j, S_sorted)
        if key in pval_cache:
            return pval_cache[key]
        # Deterministic per-test seed from (i, j, S, depth, master seed).
        # Hash collisions are fine because RCIT only needs reproducibility.
        local_seed = (seed * 1_000_003
                      + i * 9973
                      + j * 97
                      + sum(S_sorted) * 7
                      + len(S_sorted)) & 0x7FFFFFFF
        x = C[:, i]
        y = C[:, j]
        z = C[:, list(S_sorted)] if S_sorted else None
        pval = rcit_test(x, y, z=z, K=K, n_perm=n_perm, seed=local_seed)
        pval_cache[key] = pval
        return pval

    l = 0
    while True:
        # Build list of (i, j) edges with i < j that still exist.
        edges = []
        for i in range(V):
            for j in range(i + 1, V):
                if A[i, j] == 0:
                    continue
                # Need at least l other neighbors of i OR j to form a cond set
                nbrs_i = [k for k in range(V) if k != j and A[i, k] != 0]
                nbrs_j = [k for k in range(V) if k != i and A[j, k] != 0]
                # Apply self-past contemp filter (no-op when p_contemp is None
                # or this is a lag-edge test).
                nbrs_i_f = _filter_nbrs_for_contemp(i, j, nbrs_i)
                nbrs_j_f = _filter_nbrs_for_contemp(i, j, nbrs_j)
                if max(len(nbrs_i_f), len(nbrs_j_f)) >= l:
                    edges.append((i, j, nbrs_i_f, nbrs_j_f))

        if not edges:
            break

        removed_this_pass = 0
        for (i, j, _nbrs_i_stale, _nbrs_j_stale) in edges:
            if A[i, j] == 0:
                continue
            # Recompute neighbors in case earlier deletions changed them.
            nbrs_i = [k for k in range(V) if k != j and A[i, k] != 0]
            nbrs_j = [k for k in range(V) if k != i and A[j, k] != 0]
            # Reapply contemp self-past filter.
            nbrs_i = _filter_nbrs_for_contemp(i, j, nbrs_i)
            nbrs_j = _filter_nbrs_for_contemp(i, j, nbrs_j)

            removed = False
            # Try conditioning on subsets of adj(i) \ {j}
            if len(nbrs_i) >= l:
                for S_list in itertools.combinations(nbrs_i, l):
                    S = tuple(S_list)
                    pval = _ci_pvalue(i, j, S)
                    if pval > alpha:
                        A[i, j] = 0
                        A[j, i] = 0
                        sep_sets[(i, j)] = S
                        sep_sets[(j, i)] = S
                        removed_this_pass += 1
                        removed = True
                        break
            if removed:
                continue

            # Try conditioning on subsets of adj(j) \ {i}
            if len(nbrs_j) >= l:
                for S_list in itertools.combinations(nbrs_j, l):
                    S = tuple(S_list)
                    pval = _ci_pvalue(i, j, S)
                    if pval > alpha:
                        A[i, j] = 0
                        A[j, i] = 0
                        sep_sets[(i, j)] = S
                        sep_sets[(j, i)] = S
                        removed_this_pass += 1
                        removed = True
                        break

        if verbose:
            print(f"[pc_rcit] depth l={l}: removed {removed_this_pass} edges "
                  f"({int(A.sum()//2)} remain)", flush=True)

        l += 1
        if l > max_cond_size:
            break
        if l >= V - 1:
            break

    return A, sep_sets


if __name__ == '__main__':
    # Smoke test: 4-node toy graph, contemp edge 0->1, others spurious.
    rng = np.random.default_rng(0)
    T = 1000
    p = 4
    X = rng.normal(size=(T, p))
    X[:, 1] += 0.6 * X[:, 0]  # contemp 0-1
    A, sep = pc_skeleton_rcit(X, alpha=0.05, tau=1, K=25, n_perm=50,
                              verbose=True)
    print("A_contemp =\n", A)
    print("sep_sets (subset) =", dict(list(sep.items())[:4]))
