"""
_pc_raw_v2.py

PC skeleton for the CITS+ v2 contemporaneous step with symmetric
Markov-blanket conditioning.

KEY DIFFERENCE FROM _pc_raw.py (v1):
  v1: PC runs on the lag-0 contemporaneous slice alone (X_t only, shape p x N).
      Conditioning sets are drawn only from other lag-0 variables. This fails on
      diamond DAGs where two contemporaneous variables share a common past parent
      (e.g., lingauss2: 0->1 and 0->2 both at lag-1; then 1 and 2 appear
      correlated at lag-0, and PC wrongly inserts a 1--2 contemp edge because
      it cannot condition on their common cause X_0[t-1]).

  v2: PC runs on the stacked [X_{t-1}, X_t] window (shape 2p x N). This lets
      the skeleton phase condition on past variables when testing contemporaneous
      independence. After the skeleton phase we extract only the p x p
      bottom-right block (X_t -- X_t edges) and discard the cross-block edges
      (X_{t-1} -- X_t, which are lagged edges handled by CITS).

      This is the Markov-blanket-reach requirement: contemporaneous endpoint
      blankets span tau lags back, so conditioning depth = tau+1 window.

IMPLEMENTATION:

  For tau=1 and p neurons, data_transform (from cits.methods) produces a
  chi matrix of shape ((2*tau+2)*p, N). For tau=1:
    - Rows (2*tau+1)*p : (2*tau+2)*p  == rows t_target*p : (t_target+1)*p
      are the contemporaneous slice (X_t).
    - Rows (2*tau)*p : (2*tau+1)*p    == rows (t_target-1)*p : t_target*p
      are the one-lag-back slice (X_{t-1}).

  We stack both into a (2p, N) matrix and run the PC skeleton on it.
  After the PC skeleton we read off the p x p subblock corresponding to
  X_t -- X_t edges.

  On the simulation benchmark (p=4), the stacked matrix is (8, N). PC at
  this size is fast on CPU (< 1s). For real-data use on large fields
  (p ~ 50-300), the cuPC backend should be used via _cupc_wrapper.

Inputs:
  X        : (T, p) float64 raw calcium trace for one trial.
  alpha    : significance level for Fisher-z conditional-independence test.
  tau      : CITS Markovian order (default 1). Conditioning window = tau+1.
  backend  : 'cupc' (GPU, fast for large p) or 'python' (CPU, for small p
             or testing). Default 'cupc'.
  use_gpu  : legacy flag passed to python backend (ignored for cupc).
  verbose  : print diagnostics.

Outputs:
  A_contemp   : (p, p) int symmetric adjacency. Only X_t -- X_t edges.
                Diagonal 0.
  r_mat       : (p, p) float. |Pearson r| at l=0 for retained edges
                (computed from the contemporaneous slice correlation only).
  sep_sets    : dict[(i, j) -> tuple[int]]. Separating set used to remove
                each removed edge. CAUTION: indices here are in the 2p
                variable space (offset by p for X_t vars). The caller must
                subtract p to map to neuron indices if needed.
  sep_sets_orig : dict[(i, j) -> tuple[int]]. Separating sets remapped to
                  original p-variable space: X_t indices i,j -> i-p, j-p;
                  conditioning variables are clamped to range [0, p-1] for
                  X_t vars (subtract p) and kept negative for X_{t-1} vars
                  to signal they are lagged. Actually, format is:
                  X_{t-1} vars appear as (k - p) with a 'lag1' marker.
                  For simplicity: any cond var k < p => 'past var k';
                  any cond var k >= p => 'present var k-p'.
                  Use sep_sets_orig for orientation phase in original p space.
  inactive_neurons : np.ndarray. Indices of zero-variance neurons.
"""

from __future__ import annotations
import os
import sys
import numpy as np

_ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, _ANALYSIS_DIR)


def pc_skeleton_raw_v2(X, alpha: float = 0.05, tau: int = 1,
                       backend: str = 'cupc', use_gpu: bool = True,
                       verbose: bool = False,
                       zero_var_tol: float = 1e-12):
    """Run PC skeleton on stacked [X_{t-1}, X_t] window (CITS+ v2 contemp step).

    Parameters
    ----------
    X : array_like, shape (T, p)
        Calcium trace for one trial.
    alpha : float
        Significance level for Fisher-z conditional-independence test.
    tau : int
        CITS Markovian order. Window depth = tau+1. Default 1.
    backend : str
        'cupc' for GPU (recommended for large p), 'python' for CPU (small p /
        tests). Default 'cupc'.
    use_gpu : bool
        Legacy; ignored for cupc backend.
    verbose : bool
        Print diagnostics.
    zero_var_tol : float
        Zero-variance threshold.

    Returns
    -------
    A_contemp : np.ndarray, shape (p, p), int
        Symmetric contemporaneous skeleton (X_t -- X_t edges only).
    r_mat : np.ndarray, shape (p, p), float
        |Pearson r| of contemporaneous slice for retained edges.
    sep_sets_v2 : dict[(i, j) -> tuple]
        Separating sets in original p-variable space. Variables that were
        lagged (from the X_{t-1} block) are represented as negative integers
        -(k+1) to distinguish from lag-0 variables. For example, if the
        separating set for the contemp edge (i, j) contains lagged variable
        k, the tuple contains -(k+1).
    inactive_neurons : np.ndarray
        Indices of neurons with zero variance.
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
        print(f"[pc_skeleton_raw_v2] {inactive_neurons.size} inactive neurons "
              f"(var < {zero_var_tol}): {inactive_neurons.tolist()}", flush=True)
    if inactive_neurons.size > 0:
        rng = np.random.default_rng(0)
        X = X.copy()
        for j in inactive_neurons:
            X[:, j] = rng.standard_normal(T) * 1e-10

    # Build stacked chi window: [X_{t-1}, X_t] -> shape (2p, N)
    stacked = _build_stacked_chi(X, tau, verbose=verbose)  # (2*p, N)

    two_p, N = stacked.shape
    assert two_p == 2 * p, f"Expected 2*p={2*p} rows, got {two_p}"

    if verbose:
        print(f"[pc_skeleton_raw_v2] stacked shape={stacked.shape} "
              f"(2p={2*p}, N={N})", flush=True)

    # Run PC skeleton on the full (2p, N) stacked matrix
    A_full, sep_sets_full = _run_pc_on_stacked(stacked, alpha, backend, verbose)
    # A_full is (2p, 2p). Block structure:
    #   upper-left  (0:p, 0:p) = X_{t-1} -- X_{t-1}  (past-past, ignore)
    #   upper-right (0:p, p:2p) = X_{t-1} -- X_t     (lag edges, ignore)
    #   lower-right (p:2p, p:2p) = X_t -- X_t         (contemp edges, KEEP)

    # Extract bottom-right p x p contemp block
    A_contemp = A_full[p:2*p, p:2*p].copy()
    np.fill_diagonal(A_contemp, 0)

    # Zero out edges incident on inactive neurons
    if inactive_neurons.size > 0:
        for j in inactive_neurons:
            A_contemp[j, :] = 0
            A_contemp[:, j] = 0

    # Build correlation matrix of X_t slice only (for r_mat output)
    X_t_slice = stacked[p:2*p, :]  # (p, N)
    R_t = np.corrcoef(X_t_slice)   # (p, p)
    r_mat = np.where(A_contemp != 0, np.abs(R_t), 0.0)
    np.fill_diagonal(r_mat, 0.0)

    # Remap sep_sets from 2p-variable space to p-variable space.
    # Indices 0..p-1 in stacked correspond to X_{t-1} (lagged).
    # Indices p..2p-1 correspond to X_t (contemp).
    # For the contemp skeleton (bottom-right block), edges are between
    # nodes with stacked indices p+i and p+j. Their sep sets may contain
    # any of the 2p variables. Remap:
    #   stacked index k < p   -> lagged var k,  represented as -(k+1)
    #   stacked index k >= p  -> contemp var k-p, represented as k-p (non-neg)
    sep_sets_v2 = {}
    for (si, sj), S in sep_sets_full.items():
        # Only include pairs from the contemp block (both indices in p..2p-1)
        if si < p or sj < p:
            continue
        orig_i = si - p
        orig_j = sj - p
        remapped_S = tuple(
            (k - p) if k >= p else -(k + 1)
            for k in S
        )
        sep_sets_v2[(orig_i, orig_j)] = remapped_S
        sep_sets_v2[(orig_j, orig_i)] = remapped_S

    if verbose:
        n_contemp_edges = int(A_contemp.sum() // 2)
        print(f"[pc_skeleton_raw_v2] contemp edges found: {n_contemp_edges}",
              flush=True)

    return A_contemp, r_mat, sep_sets_v2, inactive_neurons


def _build_stacked_chi(X: np.ndarray, tau: int, verbose: bool = False):
    """Build the stacked [X_{t-1}, X_t] matrix using CITS data_transform.

    Parameters
    ----------
    X : np.ndarray, shape (T, p)
        Raw calcium trace.
    tau : int
        CITS Markovian order.

    Returns
    -------
    stacked : np.ndarray, shape (2*p, N)
        Stacked matrix. Rows 0..p-1 are X_{t-1}; rows p..2p-1 are X_t.
        N = floor((T - 2*tau) / (2*tau+2)).
    """
    T, p = X.shape

    if tau == 0:
        # tau=0: no lagged variables. Stacked reduces to [X_t, X_t] (trivial
        # duplicate). Not meaningful scientifically but handles edge case.
        stacked = np.vstack([X.T, X.T])  # (2p, T)
        return stacked

    try:
        from cits import methods as cits_m
    except ImportError as e:
        raise ImportError(
            "cits package required; install or set tau=0") from e

    # data_transform expects (p, T). Our X is (T, p) so transpose.
    chi_full = cits_m.data_transform(X.T, tau)
    # chi_full shape: ((2*tau+2)*p, N) for tau=1 -> (4p, N)
    # For tau=1:
    #   t_target = 2*tau+1 = 3  -> rows 3p : 4p = X_t (contemporaneous)
    #   t_target - 1 = 2        -> rows 2p : 3p = X_{t-1}
    t_target = 2 * tau + 1
    past_start   = (t_target - 1) * p
    past_end     = t_target * p
    contemp_start = t_target * p
    contemp_end   = (t_target + 1) * p

    if contemp_end > chi_full.shape[0]:
        raise ValueError(
            f"chi_full has only {chi_full.shape[0]} rows; "
            f"need at least {contemp_end} for tau={tau}, p={p}")

    past_slice    = chi_full[past_start:past_end, :]     # (p, N) X_{t-1}
    contemp_slice = chi_full[contemp_start:contemp_end, :]  # (p, N) X_t

    stacked = np.vstack([past_slice, contemp_slice])  # (2p, N)

    if verbose:
        print(f"[_build_stacked_chi] chi_full.shape={chi_full.shape}, "
              f"past rows {past_start}:{past_end}, "
              f"contemp rows {contemp_start}:{contemp_end}, "
              f"stacked.shape={stacked.shape}", flush=True)

    return stacked


def _run_pc_on_stacked(stacked: np.ndarray, alpha: float,
                       backend: str, verbose: bool):
    """Run PC skeleton on the stacked (2p, N) matrix.

    Parameters
    ----------
    stacked : np.ndarray, shape (2p, N)
        Stacked past+present data.
    alpha : float
        Fisher-z significance level.
    backend : str
        'cupc' or 'python'.
    verbose : bool

    Returns
    -------
    A_full : np.ndarray, shape (2p, 2p), int
        Full undirected skeleton across all 2p variables.
    sep_sets_full : dict[(i, j) -> tuple[int]]
        Separating sets in 2p-variable space.
    """
    two_p, N = stacked.shape

    if backend == 'cupc':
        from _cupc_wrapper import pc_skeleton_cupc
        # cupc expects (T, p) = (N, 2p)
        A_full, sep_sets_full, _, _ = pc_skeleton_cupc(
            stacked.T, alpha=alpha, verbose=verbose)
    elif backend == 'python':
        # CPU PC skeleton from simulation_benchmark_fc_methods (self-contained)
        # Import the function from the benchmark script if available, else inline.
        try:
            sys.path.insert(0, _ANALYSIS_DIR)
            from simulation_benchmark_fc_methods import pc_skeleton_cpu
            A_full, sep_sets_full = pc_skeleton_cpu(stacked, alpha=alpha)
        except ImportError:
            A_full, sep_sets_full = _pc_skeleton_cpu_inline(stacked, alpha=alpha)
    else:
        raise ValueError(f"Unknown backend {backend!r}; use 'cupc' or 'python'")

    return A_full, sep_sets_full


# =============================================================================
#  Inline CPU PC skeleton (fallback, no external deps)
# =============================================================================

def _fisher_r_crit(N: int, k: int, alpha: float = 0.05) -> float:
    """Critical |r| for Fisher-z test with k conditioning vars."""
    from scipy import stats as scipy_stats
    df = N - k - 3
    if df <= 0:
        return 1.0
    t_crit = scipy_stats.t.ppf(1.0 - alpha / 2.0, df)
    r_crit = t_crit / np.sqrt(df + t_crit**2)
    return float(r_crit)


def _partial_corr(X_data: np.ndarray, i: int, j: int, cond_set: tuple) -> float:
    """Partial correlation of X_data[i] and X_data[j] given X_data[cond_set]."""
    import itertools
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
    """Plain CPU PC skeleton on (p, N) data matrix. Fallback for no-benchmark-import."""
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

        made_removal = False
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
                    made_removal = True
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
                        made_removal = True
                        break

        l += 1
        if l >= p - 1:
            break

    return A, sep_sets
