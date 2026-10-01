"""
_pc_raw_v_latest.py

PC-contemporaneous step for CITS+ v_latest, using the optimal 2τ+1 chi.

DIFFERENCE FROM _pc_raw_v2.py
------------------------------
v2 calls cits.methods.data_transform (window width 2(τ+1) = 4 for τ=1) and
extracts slices {2*tau, 2*tau+1} (= {2, 3}) to build the stacked [X_{t-1}, X_t]
matrix.

v_latest calls cits_optimal.data_transform_optimal (window width 2τ+1 = 3 for
τ=1) and extracts slices {tau, 2*tau} (= {1, 2}) for the same stacked window.
The chi has 3p rows instead of 4p rows, but the two slices extracted are
semantically identical: [X_{t-τ}, ..., X_{t-1}] past slice + X_t present slice.

For τ=1 (the standard case):
  Optimal chi layout (3p rows):
    slice 0 (rows 0   :   p)  =  X_{t-2}   <- NOT used by PC-contemp
    slice 1 (rows p   : 2p)   =  X_{t-1}   <- past slice (used as conditioning)
    slice 2 (rows 2p  : 3p)   =  X_t        <- present slice (edge candidates)

  We stack slices {1, 2} -> (2p, N) matrix, same as v2 stacked from the larger chi.

  PC then runs on this (2p, N) matrix.  Edge candidates are X_t -- X_t pairs
  (bottom-right p×p block).  Conditioning can include X_{t-1} variables
  (top-left p rows), which gives the Markov-blanket-reach guarantees for
  contemporaneous independence.

Sample efficiency:
  N_optimal = T // (2τ+1)  =  T // 3  for τ=1.
  N_v2      = (T - 2*(τ+1)) // (2*(τ+1))  ≈  T // 4  for τ=1.
  Ratio: N_optimal / N_v2 ≈ 4/3  (+33%).

Inputs:
  X        : (T, p) float64 raw calcium trace for one trial.
  alpha    : significance level for Fisher-z conditional-independence test.
  tau      : CITS Markovian order (default 1).
  backend  : 'cupc' (GPU) or 'python' (CPU, default for sim/test).
  verbose  : print diagnostics.

Outputs (identical schema to _pc_raw_v2.pc_skeleton_raw_v2):
  A_contemp    : (p, p) int symmetric adjacency. X_t -- X_t edges only.
  r_mat        : (p, p) float. |Pearson r| at lag-0 for retained edges.
  sep_sets_vl  : dict[(i, j) -> tuple].  Separating sets in original p-space.
                 Lagged variables (from X_{t-1} block) encoded as -(k+1).
  inactive_neurons : np.ndarray.  Zero-variance neuron indices.
"""

from __future__ import annotations
import os
import sys
import numpy as np

_ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, _ANALYSIS_DIR)


def pc_skeleton_raw_v_latest(X, alpha: float = 0.05, tau: int = 1,
                              backend: str = 'python', use_gpu: bool = True,
                              verbose: bool = False,
                              zero_var_tol: float = 1e-12):
    """PC skeleton using the 2τ+1 optimal chi for the contemporaneous step.

    Parameters
    ----------
    X : array_like, shape (T, p)
        Calcium trace for one trial.
    alpha : float
        Significance level for Fisher-z conditional-independence test.
    tau : int
        CITS Markovian order. Default 1.
    backend : str
        'cupc' for GPU, 'python' for CPU (default for sims/tests).
    verbose : bool
    zero_var_tol : float

    Returns
    -------
    A_contemp : np.ndarray, shape (p, p), int
        Symmetric contemporaneous skeleton (X_t -- X_t edges only).
    r_mat : np.ndarray, shape (p, p), float
        |Pearson r| of contemporaneous slice for retained edges.
    sep_sets_vl : dict[(i, j) -> tuple]
        Separating sets remapped to p-variable space.
        Lagged variables are encoded as -(k+1) to distinguish from lag-0 vars.
    inactive_neurons : np.ndarray
        Indices of zero-variance neurons.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (T, p); got shape {X.shape}")
    T, p = X.shape

    # Zero-variance guard
    col_var = X.var(axis=0)
    inactive_mask = col_var < zero_var_tol
    inactive_neurons = np.flatnonzero(inactive_mask)
    if inactive_neurons.size > 0 and verbose:
        print(f"[pc_skeleton_raw_v_latest] {inactive_neurons.size} inactive neurons "
              f"(var < {zero_var_tol}): {inactive_neurons.tolist()}", flush=True)
    if inactive_neurons.size > 0:
        rng = np.random.default_rng(0)
        X = X.copy()
        for j in inactive_neurons:
            X[:, j] = rng.standard_normal(T) * 1e-10

    # Build stacked [X_{t-1}, X_t] from the optimal 2τ+1 chi.
    stacked = _build_stacked_chi_optimal(X, tau, verbose=verbose)  # (2p, N_opt)

    two_p, N = stacked.shape
    assert two_p == 2 * p, f"Expected 2p={2*p} rows, got {two_p}"

    if verbose:
        print(f"[pc_skeleton_raw_v_latest] stacked shape={stacked.shape} "
              f"(2p={2*p}, N={N})", flush=True)

    # Run PC skeleton on the full (2p, N) stacked matrix.
    A_full, sep_sets_full = _run_pc_on_stacked(stacked, alpha, backend, verbose)

    # Extract bottom-right p x p contemp block.
    A_contemp = A_full[p:2*p, p:2*p].copy()
    np.fill_diagonal(A_contemp, 0)

    # Zero out edges incident on inactive neurons.
    if inactive_neurons.size > 0:
        for j in inactive_neurons:
            A_contemp[j, :] = 0
            A_contemp[:, j] = 0

    # Correlation matrix of X_t slice (for r_mat output).
    X_t_slice = stacked[p:2*p, :]   # (p, N)
    R_t = np.corrcoef(X_t_slice)    # (p, p)
    r_mat = np.where(A_contemp != 0, np.abs(R_t), 0.0)
    np.fill_diagonal(r_mat, 0.0)

    # Remap sep_sets from 2p-space to p-space.
    # Stacked layout: rows 0..p-1 = X_{t-1} (lagged), rows p..2p-1 = X_t.
    # Lagged conditioning vars k < p  -> encoded as -(k+1).
    # Present vars k >= p             -> encoded as k-p.
    sep_sets_vl = {}
    for (si, sj), S in sep_sets_full.items():
        if si < p or sj < p:
            continue
        orig_i = si - p
        orig_j = sj - p
        remapped_S = tuple(
            (k - p) if k >= p else -(k + 1)
            for k in S
        )
        sep_sets_vl[(orig_i, orig_j)] = remapped_S
        sep_sets_vl[(orig_j, orig_i)] = remapped_S

    if verbose:
        n_contemp_edges = int(A_contemp.sum() // 2)
        print(f"[pc_skeleton_raw_v_latest] contemp edges found: {n_contemp_edges}",
              flush=True)

    return A_contemp, r_mat, sep_sets_vl, inactive_neurons


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_stacked_chi_optimal(X: np.ndarray, tau: int,
                                verbose: bool = False) -> np.ndarray:
    """Extract [X_{t-tau}, X_t] stacked matrix from the optimal 2τ+1 chi.

    Parameters
    ----------
    X   : (T, p) float
    tau : int

    Returns
    -------
    stacked : (2*p, N_opt) float
        Row layout: rows 0..p-1 = X_{t-tau} (past), rows p..2p-1 = X_t.
        N_opt = T // (2*tau+1).
    """
    T, p = X.shape

    if tau == 0:
        stacked = np.vstack([X.T, X.T])
        return stacked

    from cits_optimal import data_transform_optimal

    # data_transform_optimal expects (p, T).
    chi_opt = data_transform_optimal(X.T, tau)
    # chi_opt shape: (p*(2*tau+1), N_opt)
    # Slice layout (tau=1): slice 0 = X_{t-2}, slice 1 = X_{t-1}, slice 2 = X_t
    # For general tau: slice tau = X_{t-tau}, slice 2*tau = X_t.

    w = 2 * tau + 1
    past_start    = tau * p
    past_end      = (tau + 1) * p
    contemp_start = 2 * tau * p
    contemp_end   = (2 * tau + 1) * p

    if contemp_end > chi_opt.shape[0]:
        raise ValueError(
            f"chi_opt has {chi_opt.shape[0]} rows; need {contemp_end} "
            f"for tau={tau}, p={p}")

    past_slice    = chi_opt[past_start:past_end, :]       # (p, N_opt)
    contemp_slice = chi_opt[contemp_start:contemp_end, :] # (p, N_opt)

    stacked = np.vstack([past_slice, contemp_slice])       # (2p, N_opt)

    if verbose:
        N_opt = chi_opt.shape[1]
        print(f"[_build_stacked_chi_optimal] chi_opt.shape={chi_opt.shape}, "
              f"past rows {past_start}:{past_end}, "
              f"contemp rows {contemp_start}:{contemp_end}, "
              f"stacked.shape={stacked.shape}, N_opt={N_opt}", flush=True)

    return stacked


def _run_pc_on_stacked(stacked: np.ndarray, alpha: float,
                       backend: str, verbose: bool):
    """Run PC skeleton on (2p, N) stacked matrix. Mirrors _pc_raw_v2._run_pc_on_stacked."""
    two_p, N = stacked.shape

    if backend == 'cupc':
        from _cupc_wrapper import pc_skeleton_cupc
        A_full, sep_sets_full, _, _ = pc_skeleton_cupc(
            stacked.T, alpha=alpha, verbose=verbose)
    elif backend == 'python':
        try:
            sys.path.insert(0, _ANALYSIS_DIR)
            from simulation_benchmark_fc_methods import pc_skeleton_cpu
            A_full, sep_sets_full = pc_skeleton_cpu(stacked, alpha=alpha)
        except ImportError:
            A_full, sep_sets_full = _pc_skeleton_cpu_inline(stacked, alpha=alpha)
    else:
        raise ValueError(f"Unknown backend {backend!r}; use 'cupc' or 'python'")

    return A_full, sep_sets_full


# ---------------------------------------------------------------------------
# Inline CPU PC skeleton (fallback)
# ---------------------------------------------------------------------------

def _fisher_r_crit(N: int, k: int, alpha: float = 0.05) -> float:
    from scipy import stats as scipy_stats
    df = N - k - 3
    if df <= 0:
        return 1.0
    t_crit = scipy_stats.t.ppf(1.0 - alpha / 2.0, df)
    r_crit = t_crit / np.sqrt(df + t_crit**2)
    return float(r_crit)


def _partial_corr(X_data: np.ndarray, i: int, j: int, cond_set: tuple) -> float:
    if len(cond_set) == 0:
        r = np.corrcoef(X_data[i], X_data[j])[0, 1]
        return float(r)
    S = np.array(list(cond_set), dtype=int)
    Z = X_data[S].T
    N_s = Z.shape[0]
    ones = np.ones((N_s, 1))
    Z_aug = np.hstack([ones, Z])
    try:
        B, _, _, _ = np.linalg.lstsq(Z_aug, X_data[i], rcond=None)
        res_i = X_data[i] - Z_aug @ B
        B2, _, _, _ = np.linalg.lstsq(Z_aug, X_data[j], rcond=None)
        res_j = X_data[j] - Z_aug @ B2
    except np.linalg.LinAlgError:
        return 0.0
    if res_i.std() < 1e-12 or res_j.std() < 1e-12:
        return 0.0
    r = np.corrcoef(res_i, res_j)[0, 1]
    return float(r)


def _pc_skeleton_cpu_inline(X_data: np.ndarray, alpha: float = 0.05):
    """Plain CPU PC skeleton on (p, N) data matrix. Fallback."""
    import itertools
    p, N = X_data.shape
    A = np.ones((p, p), dtype=int)
    np.fill_diagonal(A, 0)
    sep_sets = {}

    l = 0
    while True:
        edges_to_test = []
        for i in range(p):
            for j in range(p):
                if i == j or A[i, j] == 0:
                    continue
                nbrs_i = [k for k in range(p) if k != j and A[i, k] != 0]
                if len(nbrs_i) >= l:
                    edges_to_test.append((i, j))

        if not edges_to_test:
            break

        tested_pairs = set()
        for (i, j) in edges_to_test:
            if A[i, j] == 0:
                continue
            pair_key = (min(i, j), max(i, j))
            if pair_key in tested_pairs:
                continue
            tested_pairs.add(pair_key)

            nbrs_i = [k for k in range(p) if k != j and A[i, k] != 0]
            r_crit = _fisher_r_crit(N, l, alpha)

            found_sep = False
            for S_list in itertools.combinations(nbrs_i, l):
                S = tuple(S_list)
                r = _partial_corr(X_data, i, j, S)
                if abs(r) < r_crit:
                    A[i, j] = 0
                    A[j, i] = 0
                    sep_sets[(i, j)] = S
                    sep_sets[(j, i)] = S
                    found_sep = True
                    break

            if not found_sep:
                nbrs_j = [k for k in range(p) if k != i and A[j, k] != 0]
                for S_list in itertools.combinations(nbrs_j, l):
                    S = tuple(S_list)
                    r = _partial_corr(X_data, i, j, S)
                    if abs(r) < r_crit:
                        A[i, j] = 0
                        A[j, i] = 0
                        sep_sets[(i, j)] = S
                        sep_sets[(j, i)] = S
                        break

        l += 1
        if l >= p - 1:
            break

    return A, sep_sets
