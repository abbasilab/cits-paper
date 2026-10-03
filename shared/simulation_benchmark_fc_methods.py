"""
simulation_benchmark_fc_methods.py

Benchmark 6 FC inference methods against ground-truth causal structure on
10 simulation configurations (5 original CITS models + 5 +contemp variants).

Methods:
  1. Granger  -- pairwise lag-1 F-test (OLS per ordered pair)
  2. TPC      -- thresholded partial correlation (lag-0 rolling window)
  3. PC       -- vanilla PC on flat contemporaneous data
  4. CITS     -- cits.methods.cits_full (lag-1 window only)
  5. VerB     -- CITS + PC-contemp + Meek propagation + LSCM refit
  6. VerBsafe -- CITS + PC-contemp + collider-only (no Meek) + LSCM refit

Output:
  figures/2026-06-02_simulation_benchmark/simulation_results.csv
  figures/2026-06-02_simulation_benchmark/method_comparison.{png,pdf}
  figures/2026-06-02_simulation_benchmark/per_edge_type.{png,pdf}

Usage:
  python simulation_benchmark_fc_methods.py [--workers N] [--seeds N] [--smoke]
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import warnings
import itertools
import multiprocessing as mp
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:   # shared/ (paths.py)
    sys.path.insert(0, _THIS_DIR)
from paths import OUT_ROOT as _OUT_ROOT
# outputs of this module's own __main__ benchmark (was arousal_paper_overleaf/figures/2026-06-02_simulation_benchmark)
OUT_DIR    = os.path.join(_OUT_ROOT, 'shared', '2026-06-02_simulation_benchmark')
OUT_CSV    = os.path.join(OUT_DIR, 'simulation_results.csv')

for _p in [_THIS_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Analysis-dir private modules ───────────────────────────────────────────────
from _pc_orientation import orient_v_structures, apply_meek_rules
from _lscm_refit     import lscm_refit_cpdag

# ── CITS methods (lag-windowed) ────────────────────────────────────────────────
from cits import methods as cits_methods
from cits.simulate_timeseries import simulate

N_NEURONS = 4
ALPHA = 0.05
LAG   = 1

# =============================================================================
#  Simulation model definitions (5 original + 5 +contemp)
# =============================================================================

def simulate_extended(model_name: str, noise: float, T: int, seed: int):
    """
    Generate (X, gt_unweighted, gt_weighted, contemp_gt_unweighted, contemp_gt_weighted).

    For original models the contemp ground truth is all-zeros.
    For +contemp variants we inject one contemporaneous edge.

    Returns
    -------
    X : np.ndarray (n_neurons, T) -- time series
    gt_lag_uw : np.ndarray (4,4) int -- lagged ground-truth unweighted adjacency
    gt_lag_w  : np.ndarray (4,4) float -- lagged ground-truth weighted
    gt_contemp_uw : np.ndarray (4,4) int -- contemp ground-truth unweighted
    gt_contemp_w  : np.ndarray (4,4) float -- contemp ground-truth weighted
    """
    rng = np.random.default_rng(seed)
    p = N_NEURONS
    has_contemp = model_name.endswith('+contemp')
    base_model  = model_name.replace('+contemp', '')

    # Use CITS simulate for the base lagged structure, then optionally add
    # a contemporaneous term.  We need per-seed RNG, so we replicate the
    # simulation logic here rather than calling the module-level rng.

    smspikes  = np.zeros((p, T))
    gt_lag_w  = np.zeros((p, p))

    # ── lingauss1 ──────────────────────────────────────────────────────────────
    if base_model == 'lingauss1':
        for i in range(p):
            smspikes[i, 0] = rng.normal(scale=noise)
        for t in range(1, T):
            smspikes[0, t] = rng.normal(scale=noise) + 1.0
            smspikes[1, t] = rng.normal(scale=noise) - 1.0
            smspikes[2, t] = (2.0 * smspikes[0, t-1]
                              + 1.0 * smspikes[1, t-1]
                              + rng.normal(scale=noise))
            smspikes[3, t] = 2.0 * smspikes[2, t-1] + rng.normal(scale=noise)
            if has_contemp:
                # Add 1->2 contemporaneous (same weight as lagged 1->2 = 1.0)
                smspikes[2, t] += 1.0 * smspikes[1, t]

        gt_lag_w[0, 2] = 2.0
        gt_lag_w[1, 2] = 1.0
        gt_lag_w[2, 3] = 2.0
        contemp_w = np.zeros((p, p))
        if has_contemp:
            contemp_w[1, 2] = 1.0

    # ── lingauss2 ──────────────────────────────────────────────────────────────
    elif base_model == 'lingauss2':
        for i in range(p):
            smspikes[i, 0] = rng.normal(scale=noise)
        for t in range(1, T):
            smspikes[0, t] = rng.normal(scale=noise) + 1.0
            smspikes[1, t] = (-1.0 + 2.0 * smspikes[0, t-1]
                              + rng.normal(scale=noise))
            smspikes[2, t] = 2.0 * smspikes[0, t-1] + rng.normal(scale=noise)
            smspikes[3, t] = (smspikes[1, t-1] + smspikes[2, t-1]
                              + rng.normal(scale=noise))
            if has_contemp:
                # Add 1->3 contemporaneous (weight 1.0, similar to lagged)
                smspikes[3, t] += 1.0 * smspikes[1, t]

        gt_lag_w[0, 1] = 2.0
        gt_lag_w[0, 2] = 2.0
        gt_lag_w[1, 3] = 1.0
        gt_lag_w[2, 3] = 1.0
        contemp_w = np.zeros((p, p))
        if has_contemp:
            contemp_w[1, 3] = 1.0

    # ── nonlinnongauss1 ────────────────────────────────────────────────────────
    elif base_model == 'nonlinnongauss1':
        for i in range(p):
            smspikes[i, 0] = rng.random()
        for t in range(1, T):
            smspikes[0, t] = rng.random() * noise
            smspikes[1, t] = rng.random() * noise
            smspikes[2, t] = (4.0 * np.sin(smspikes[0, t-1])
                              - 3.0 * np.sin(smspikes[1, t-1])
                              + rng.random() * noise)
            smspikes[3, t] = 3.0 * np.sin(smspikes[2, t-1]) + rng.random() * noise
            if has_contemp:
                # 1->2 contemporaneous via same sin transform
                smspikes[2, t] += 3.0 * np.sin(smspikes[1, t])

        gt_lag_w[0, 2] =  1.0
        gt_lag_w[1, 2] = -1.0
        gt_lag_w[2, 3] =  1.0
        contemp_w = np.zeros((p, p))
        if has_contemp:
            contemp_w[1, 2] = 1.0

    # ── nonlinnongauss2 ────────────────────────────────────────────────────────
    elif base_model == 'nonlinnongauss2':
        for i in range(p):
            smspikes[i, 0] = rng.normal(scale=noise)
        for t in range(1, T):
            smspikes[0, t] = rng.normal(scale=noise)
            smspikes[1, t] = 4.0 * smspikes[0, t-1] + rng.normal(scale=noise)
            smspikes[2, t] = 3.0 * np.sin(smspikes[0, t-1]) + rng.normal(scale=noise)
            # Guard against log(0) with clamp
            v1 = np.abs(smspikes[1, t-1]) + 1e-6
            v2 = np.abs(smspikes[2, t-1]) + 1e-6
            smspikes[3, t] = (8.0 * np.log(v1) + 9.0 * np.log(v2)
                              + rng.normal(scale=noise))
            if has_contemp:
                # 2->3 contemporaneous
                smspikes[3, t] += 2.0 * smspikes[2, t]

        gt_lag_w[0, 1] = 1.0
        gt_lag_w[0, 2] = 1.0
        gt_lag_w[1, 3] = 1.0
        gt_lag_w[2, 3] = 1.0
        contemp_w = np.zeros((p, p))
        if has_contemp:
            contemp_w[2, 3] = 2.0

    # ── ctrnn ──────────────────────────────────────────────────────────────────
    elif base_model == 'ctrnn':
        e   = np.exp(1)
        w_c = np.zeros((p, p))
        w_c[0, 2] = 100.0
        w_c[1, 2] = 100.0
        w_c[2, 3] = 100.0
        tau_ct = 10.0
        u = np.zeros((p, T))
        for n in range(T - 1):
            for i in range(p):
                In = rng.normal(1.0, noise)
                u[i, n+1] = (u[i, n]
                             - (e * u[i, n]) / tau_ct
                             + e * np.sum(w_c[:, i] * np.tanh(u[:, n])) / tau_ct
                             + e * In / tau_ct)
        if has_contemp:
            # Add fast contemp coupling 2->3 with small tau (effectively same timestep)
            # Re-run with a small additional contemporaneous forcing
            tau_fast = 2.0
            u2 = np.zeros((p, T))
            for n in range(T - 1):
                for i in range(p):
                    In = rng.normal(1.0, noise)
                    u2[i, n+1] = (u2[i, n]
                                  - (e * u2[i, n]) / tau_ct
                                  + e * np.sum(w_c[:, i] * np.tanh(u2[:, n])) / tau_ct
                                  + e * In / tau_ct)
                # Extra fast coupling 2->3 within same step
                u2[3, n+1] += (e / tau_fast) * np.tanh(u2[2, n+1])
            smspikes = u2
        else:
            smspikes = u

        gt_lag_w[0, 0] = 1.0
        gt_lag_w[1, 1] = 1.0
        gt_lag_w[2, 2] = 1.0
        gt_lag_w[3, 3] = 1.0
        gt_lag_w[0, 2] = 1.0
        gt_lag_w[1, 2] = 1.0
        gt_lag_w[2, 3] = 1.0
        contemp_w = np.zeros((p, p))
        if has_contemp:
            contemp_w[2, 3] = 1.0

    else:
        raise ValueError(f"Unknown base model: {base_model}")

    X = smspikes
    gt_lag_uw     = (gt_lag_w != 0).astype(int)
    contemp_uw    = (contemp_w != 0).astype(int)

    return X, gt_lag_uw, gt_lag_w, contemp_uw, contemp_w


# =============================================================================
#  Self-contained PC skeleton (Fisher-z, CPU-only, no GPU dep)
# =============================================================================

def _fisher_r_crit(N: int, k: int, alpha: float = 0.05) -> float:
    """Critical |r| for Fisher-z test with k conditioning vars."""
    df = N - k - 3
    if df <= 0:
        return 1.0
    t_crit = scipy_stats.t.ppf(1.0 - alpha / 2.0, df)
    r_crit = t_crit / np.sqrt(df + t_crit**2)
    return float(r_crit)


def _partial_corr(X_data: np.ndarray, i: int, j: int,
                  cond_set: tuple) -> float:
    """Partial correlation of X[i] and X[j] given X[cond_set].

    X_data: (p, N) array -- rows are variables.
    Returns r_ij|S (scalar).
    """
    if len(cond_set) == 0:
        r = np.corrcoef(X_data[i], X_data[j])[0, 1]
        return float(r)

    # Regress out conditioning set from both i and j
    S = np.array(list(cond_set), dtype=int)
    Z = X_data[S].T  # (N, k)
    N = Z.shape[0]
    # Add intercept
    ones = np.ones((N, 1))
    Z_aug = np.hstack([ones, Z])  # (N, k+1)
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


def pc_skeleton_cpu(X_data: np.ndarray, alpha: float = 0.05):
    """
    Plain PC skeleton on (p, N) data matrix.

    Returns:
        A      : (p, p) int -- symmetric adjacency
        sep_sets: dict[(i,j) -> tuple[int]]
    """
    p, N = X_data.shape
    A = np.ones((p, p), dtype=int)
    np.fill_diagonal(A, 0)
    sep_sets = {}

    l = 0
    while True:
        # Collect edges where the node still has >= l neighbours
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

            # Try all conditioning sets of size l from neighbours of i
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

            # Also try conditioning sets from neighbours of j
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
        # Safety: stop at l = p-2
        if l >= p - 1:
            break

    return A, sep_sets


# =============================================================================
#  Method implementations
# =============================================================================

def run_granger(X: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Pairwise Granger causality via F-test (univariate lag-1 OLS).

    X: (p, T). Returns (p, p) binary adjacency: A[i, j]=1 means i Granger-causes j.
    No self-loops.
    """
    p, T = X.shape
    A = np.zeros((p, p), dtype=int)

    for j in range(p):
        y = X[j, 1:]    # target: t=1..T-1
        # Restricted model: AR(1) of j itself
        X_restr = X[j, :-1].reshape(-1, 1)
        ones     = np.ones((T-1, 1))
        X_restr_aug = np.hstack([ones, X_restr])
        B_r, _, _, _ = np.linalg.lstsq(X_restr_aug, y, rcond=None)
        res_r = y - X_restr_aug @ B_r
        ss_r  = (res_r**2).sum()
        df_r  = T - 1 - 2  # n - k_restricted

        for i in range(p):
            if i == j:
                continue
            # Full model: AR(1) of j + lag-1 of i
            X_full = np.hstack([ones, X[j, :-1].reshape(-1, 1),
                                       X[i, :-1].reshape(-1, 1)])
            B_f, _, _, _ = np.linalg.lstsq(X_full, y, rcond=None)
            res_f = y - X_full @ B_f
            ss_f  = (res_f**2).sum()
            df_f  = T - 1 - 3  # n - k_full

            if df_f <= 0 or ss_f <= 0:
                continue

            # F-test: (SS_r - SS_f) / 1  vs  SS_f / df_f
            F_stat = ((ss_r - ss_f) / 1.0) / (ss_f / df_f)
            if F_stat < 0:
                continue
            p_val = 1.0 - scipy_stats.f.cdf(F_stat, 1, df_f)
            if p_val < alpha:
                A[i, j] = 1

    return A


def run_tpc(X: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Thresholded partial correlation: rolling window lag-0 contemporaneous edges.

    Uses a 2-sample rolling window: chi = [X[:, t-1], X[:, t]] for t=1..T-1,
    stacks all pairs, then runs PC skeleton on the concatenated matrix. This
    mirrors TPC's lag-0-included structure without CITS windowing.

    For simplicity here: PC skeleton on the raw contemporaneous X (lag 0)
    PLUS lag-1 partial-correlation edges (Granger-style undirected test).
    Matches the spirit of TPC as "PC on the rolling window."

    Returns (p, p) symmetric binary adjacency.
    """
    p, T = X.shape

    # Build the chi-style stacked contemporaneous + lagged data
    # chi row structure for tau=1: [x(t-1), x(t)] for each t in range(1, T)
    # Shape: (2p, T-1)
    chi = np.vstack([X[:, :-1], X[:, 1:]])  # (2p, T-1)

    # Run PC skeleton on the 2p x (T-1) matrix (whole rolling window)
    A_full, _ = pc_skeleton_cpu(chi, alpha=alpha)

    # The (p, p) upper-right block = lag-1 edges (rows 0..p-1 -> cols p..2p-1)
    # The upper-left block (0..p-1, 0..p-1) is contemp-previous,
    # the lower-right block (p..2p-1, p..2p-1) is contemp-current.
    # We want: directed i(t-1) -> j(t) edges = upper-right block A_full[i, p+j]
    # AND undirected contemporaneous j(t) -- k(t) = lower-right block
    A_lag  = A_full[:p, p:]          # (p, p): lag-1 directed i->j
    A_c    = A_full[p:, p:]          # (p, p): contemp undirected

    # Union: an edge i->j exists if either lag-1 OR contemp connection
    A_out = np.zeros((p, p), dtype=int)
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            if A_lag[i, j] or A_c[i, j] or A_c[j, i]:
                A_out[i, j] = 1

    return A_out


def run_pc(X: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Vanilla PC on flat contemporaneous data (lag-0 only).

    X: (p, T). Treats X.T as (T, p) iid samples.
    Returns (p, p) symmetric binary adjacency.
    """
    p, T = X.shape
    A, sep_sets = pc_skeleton_cpu(X, alpha=alpha)
    # Orient and symmetrize for skeleton comparison
    G = orient_v_structures(A, sep_sets)
    # Return symmetric skeleton (undirected)
    skel = ((G + G.T) > 0).astype(int)
    np.fill_diagonal(skel, 0)
    return skel


def run_cits(X: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    CITS (lag-1-windowed PC) via cits.methods.cits_full.

    X: (p, T). Returns (p, p) binary adjacency.
    """
    p, T = X.shape
    # cits_full expects X of shape (p, T)
    try:
        adj = cits_methods.cits_full(X, LAG, alpha)
    except Exception as e:
        warnings.warn(f"CITS failed: {e}")
        adj = np.zeros((p, p), dtype=int)

    # CITS returns directed adjacency; symmetrize for skeleton comparison
    adj = np.asarray(adj)
    # Convert any nonzero to 1
    adj_bin = (adj != 0).astype(int)
    np.fill_diagonal(adj_bin, 0)
    return adj_bin


def run_verB(X: np.ndarray, alpha: float = 0.05, meek: bool = True) -> np.ndarray:
    """
    Ver B (CITS + PC-contemp + LSCM refit) or Ver B-safe (no Meek).

    meek=True  -> Ver B    (full Meek propagation after v-structures)
    meek=False -> Ver Bsafe (collider-only orientations, no Meek)

    Returns (p, p) binary adjacency (skeleton of union).
    """
    p, T = X.shape

    # Step 1: CITS lagged skeleton
    try:
        cits_adj = cits_methods.cits_full(X, LAG, alpha)
    except Exception as e:
        warnings.warn(f"CITS in VerB failed: {e}")
        cits_adj = np.zeros((p, p), dtype=int)
    cits_bin = (np.asarray(cits_adj) != 0).astype(int)
    np.fill_diagonal(cits_bin, 0)

    # Step 2: PC skeleton on contemporaneous data (chi subsampled)
    # Use tau=1 chi subsampling: stride 4 to get ~T/4 independent samples
    stride = max(1, 4)
    X_t = X.T  # (T, p) -- PC wants (T, p) but our function wants (p, N)
    X_contemp = X[:, ::stride]  # (p, T//stride)
    if X_contemp.shape[1] < p + 3:
        # Not enough samples; use full X
        X_contemp = X

    A_pc, sep_sets_pc = pc_skeleton_cpu(X_contemp, alpha=alpha)

    # Step 3: Orient PC-contemp CPDAG
    G_cpdag = orient_v_structures(A_pc, sep_sets_pc)
    if meek:
        G_cpdag = apply_meek_rules(G_cpdag)

    # Step 4: Union skeleton
    # CITS lag edges: cits_bin[i,j]=1 means i->j (lagged)
    # PC-contemp skeleton: A_pc (undirected)
    union_skel = np.zeros((p, p), dtype=int)
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            # Lagged edge from CITS
            if cits_bin[i, j]:
                union_skel[i, j] = 1
            # Contemp edge from PC
            if A_pc[i, j] or A_pc[j, i]:
                union_skel[i, j] = 1
                union_skel[j, i] = 1

    np.fill_diagonal(union_skel, 0)
    return union_skel


# Method registry
METHOD_FNS = {
    'Granger' : lambda X: run_granger(X, ALPHA),
    'TPC'     : lambda X: run_tpc(X, ALPHA),
    'PC'      : lambda X: run_pc(X, ALPHA),
    'CITS'    : lambda X: run_cits(X, ALPHA),
    'VerB'    : lambda X: run_verB(X, ALPHA, meek=True),
    'VerBsafe': lambda X: run_verB(X, ALPHA, meek=False),
}

# =============================================================================
#  Metrics
# =============================================================================

def compute_metrics(pred: np.ndarray, gt_lag: np.ndarray,
                    gt_contemp: np.ndarray) -> dict:
    """
    Compute edge-level metrics (skeleton, directed where appropriate).

    All matrices are (p, p) binary int.
    gt_lag: lagged ground-truth edges (may be asymmetric for directed GT).
    gt_contemp: contemp ground-truth edges.
    gt_full: union of lag + contemp.

    We evaluate against the skeleton (undirected union) of ground truth:
    gt_skel[i,j] = 1 if (gt_lag[i,j] or gt_lag[j,i] or gt_contemp[i,j] or gt_contemp[j,i]).
    """
    p = pred.shape[0]
    gt_full = ((gt_lag + gt_lag.T + gt_contemp + gt_contemp.T) > 0).astype(int)
    np.fill_diagonal(gt_full, 0)

    # Skeleton of prediction
    pred_skel = ((pred + pred.T) > 0).astype(int)
    np.fill_diagonal(pred_skel, 0)

    # Overall skeleton metrics (upper triangle only to avoid double-counting)
    tp = fp = fn = 0
    for i in range(p):
        for j in range(i+1, p):
            has_edge = pred_skel[i, j] > 0
            true_edge = gt_full[i, j] > 0
            if has_edge and true_edge:
                tp += 1
            elif has_edge and not true_edge:
                fp += 1
            elif not has_edge and true_edge:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    shd       = fp + fn  # symmetric Hamming distance (skeleton)

    # Per-edge-type: contemp edges (directed in GT)
    gt_contemp_skel = ((gt_contemp + gt_contemp.T) > 0).astype(int)
    np.fill_diagonal(gt_contemp_skel, 0)
    gt_lag_skel = ((gt_lag + gt_lag.T) > 0).astype(int)
    np.fill_diagonal(gt_lag_skel, 0)

    contemp_tp = contemp_fp = lag_tp = lag_fp = 0
    for i in range(p):
        for j in range(i+1, p):
            has_edge = pred_skel[i, j] > 0
            is_contemp = gt_contemp_skel[i, j] > 0
            is_lag     = gt_lag_skel[i, j] > 0
            if has_edge:
                if is_contemp:
                    contemp_tp += 1
                else:
                    contemp_fp += 1
                if is_lag:
                    lag_tp += 1
                else:
                    lag_fp += 1

    return {
        'edges_tp'        : tp,
        'edges_fp'        : fp,
        'edges_fn'        : fn,
        'edges_precision' : round(precision, 6),
        'edges_recall'    : round(recall, 6),
        'edges_F1'        : round(f1, 6),
        'edges_SHD'       : shd,
        'contemp_tp'      : contemp_tp,
        'contemp_fp'      : contemp_fp,
        'lag_tp'          : lag_tp,
        'lag_fp'          : lag_fp,
    }


# =============================================================================
#  Single-run worker (called in parallel)
# =============================================================================

def _run_one(args):
    """Worker function: runs all 6 methods on one (model, T, noise, seed) config."""
    model_name, T, noise, seed = args
    rows = []
    has_contemp = int(model_name.endswith('+contemp'))

    try:
        X, gt_lag_uw, gt_lag_w, gt_contemp_uw, gt_contemp_w = simulate_extended(
            model_name, noise, T, seed)
    except Exception as e:
        warnings.warn(f"Sim failed for {model_name} T={T} noise={noise} seed={seed}: {e}")
        return []

    for method_name, method_fn in METHOD_FNS.items():
        t0 = time.perf_counter()
        try:
            pred = method_fn(X)
        except Exception as e:
            warnings.warn(f"Method {method_name} failed on {model_name} seed={seed}: {e}")
            pred = np.zeros((N_NEURONS, N_NEURONS), dtype=int)
        runtime = time.perf_counter() - t0

        metrics = compute_metrics(pred, gt_lag_uw, gt_contemp_uw)
        row = {
            'model'      : model_name,
            'has_contemp': has_contemp,
            'T'          : T,
            'noise'      : noise,
            'seed'       : seed,
            'method'     : method_name,
            'runtime_sec': round(runtime, 4),
        }
        row.update(metrics)
        rows.append(row)

    return rows


# =============================================================================
#  Main
# =============================================================================

MODEL_NAMES = [
    'lingauss1', 'lingauss1+contemp',
    'lingauss2', 'lingauss2+contemp',
    'nonlinnongauss1', 'nonlinnongauss1+contemp',
    'nonlinnongauss2', 'nonlinnongauss2+contemp',
    'ctrnn', 'ctrnn+contemp',
]

T_VALUES     = [1000, 5000]
NOISE_VALUES = [1.0, 0.5]
N_SEEDS      = 50


def main():
    parser = argparse.ArgumentParser(description='FC simulation benchmark')
    parser.add_argument('--workers', type=int, default=min(mp.cpu_count(), 24),
                        help='Number of parallel workers')
    parser.add_argument('--seeds', type=int, default=N_SEEDS,
                        help='Seeds per (model, T, noise)')
    parser.add_argument('--smoke', action='store_true',
                        help='Smoke test: 2 models, 1 T, 1 noise, 3 seeds')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.smoke:
        models  = MODEL_NAMES[:2]
        t_vals  = [1000]
        n_vals  = [1.0]
        n_seeds = 3
    else:
        models  = MODEL_NAMES
        t_vals  = T_VALUES
        n_vals  = NOISE_VALUES
        n_seeds = args.seeds

    # Build task list
    tasks = []
    for model, T, noise in itertools.product(models, t_vals, n_vals):
        for seed in range(n_seeds):
            tasks.append((model, T, noise, seed))

    total = len(tasks)
    n_methods = len(METHOD_FNS)
    print(f"[benchmark] {total} simulation configs x {n_methods} methods "
          f"= {total * n_methods} evaluations", flush=True)
    print(f"[benchmark] workers={args.workers}", flush=True)

    t_start = time.time()
    all_rows = []

    if args.workers > 1:
        with Pool(processes=args.workers) as pool:
            for i, result in enumerate(pool.imap_unordered(_run_one, tasks,
                                                           chunksize=4)):
                all_rows.extend(result)
                if (i + 1) % max(1, total // 20) == 0:
                    pct = 100.0 * (i+1) / total
                    elapsed = time.time() - t_start
                    eta = elapsed / (i+1) * (total - i - 1)
                    print(f"  {i+1}/{total} ({pct:.0f}%) elapsed={elapsed:.0f}s "
                          f"ETA={eta:.0f}s", flush=True)
    else:
        for i, task in enumerate(tasks):
            result = _run_one(task)
            all_rows.extend(result)
            if (i + 1) % max(1, total // 20) == 0:
                pct = 100.0 * (i+1) / total
                elapsed = time.time() - t_start
                print(f"  {i+1}/{total} ({pct:.0f}%) elapsed={elapsed:.0f}s",
                      flush=True)

    elapsed_total = time.time() - t_start
    print(f"[benchmark] Done in {elapsed_total:.1f}s. Rows: {len(all_rows)}",
          flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"[benchmark] Saved CSV -> {OUT_CSV}", flush=True)

    # Generate summary figures
    try:
        _make_figures(df)
    except Exception as e:
        print(f"[benchmark] Figure generation failed: {e}", flush=True)
        import traceback; traceback.print_exc()


# =============================================================================
#  Figures
# =============================================================================

def _make_figures(df: pd.DataFrame):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    METHODS = ['Granger', 'TPC', 'PC', 'CITS', 'VerB', 'VerBsafe']
    BASE_MODELS = ['lingauss1', 'lingauss2', 'nonlinnongauss1',
                   'nonlinnongauss2', 'ctrnn']
    COLORS = {
        'Granger' : '#1f77b4',
        'TPC'     : '#ff7f0e',
        'PC'      : '#2ca02c',
        'CITS'    : '#d62728',
        'VerB'    : '#9467bd',
        'VerBsafe': '#8c564b',
    }

    # ── Figure 1: F1 per method per model, lag-only vs +contemp ──────────────
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharey=True)
    fig.suptitle('F1 Score by Method and Model\n(T=1000, noise=1.0, N=50 seeds)',
                 fontsize=12)

    for col, bm in enumerate(BASE_MODELS):
        for row_idx, hc in enumerate([0, 1]):
            ax = axes[row_idx, col]
            sub = df[(df['model'] == (bm if hc == 0 else bm + '+contemp'))
                     & (df['T'] == 1000)
                     & (df['noise'] == 1.0)]

            if sub.empty:
                ax.set_visible(False)
                continue

            method_means  = []
            method_sems   = []
            method_labels = []
            for m in METHODS:
                vals = sub.loc[sub['method'] == m, 'edges_F1'].values
                if len(vals) == 0:
                    method_means.append(0)
                    method_sems.append(0)
                else:
                    method_means.append(np.mean(vals))
                    method_sems.append(np.std(vals) / np.sqrt(len(vals)))
                method_labels.append(m)

            xs = np.arange(len(METHODS))
            colors = [COLORS[m] for m in METHODS]
            bars = ax.bar(xs, method_means, yerr=method_sems, capsize=3,
                          color=colors, alpha=0.8, width=0.7)

            ax.set_xticks(xs)
            ax.set_xticklabels(METHODS, rotation=45, ha='right', fontsize=7)
            ax.set_ylim(0, 1.05)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            if col == 0:
                label = 'Lag-only' if hc == 0 else '+Contemp'
                ax.set_ylabel(f'{label}\nF1')
            if row_idx == 0:
                ax.set_title(bm.replace('nonlin', 'nl.'), fontsize=9)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        path = os.path.join(OUT_DIR, f'method_comparison.{ext}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[figures] Saved {path}", flush=True)
    plt.close(fig)

    # ── Figure 2: Per-edge-type recall (contemp vs lag) ────────────────────────
    # For +contemp models: contemp_recall = contemp_tp / (contemp_tp + fn_contemp)
    # Approximate: for each method, among +contemp configs, what fraction of
    # contemp edges are recovered?

    fig2, axes2 = plt.subplots(1, 5, figsize=(18, 4))
    fig2.suptitle('Contemp-Edge Recall by Method\n(+contemp variants, T=1000, noise=1.0)',
                  fontsize=12)

    for col, bm in enumerate(BASE_MODELS):
        ax = axes2[col]
        model_c = bm + '+contemp'
        sub = df[(df['model'] == model_c)
                 & (df['T'] == 1000)
                 & (df['noise'] == 1.0)].copy()

        if sub.empty:
            ax.set_visible(False)
            continue

        # contemp recall per method: contemp_tp / (total true contemp edges)
        # true contemp edges = 1 pair per +contemp model (upper-triangle)
        # but we need the fn term.  Approximate: contemp_tp / (n_true_contemp_undirected)
        # n_true_contemp_undirected = 1 for all our +contemp configs
        means, sems, labels = [], [], []
        for m in METHODS:
            vals = sub.loc[sub['method'] == m, 'contemp_tp'].values
            # true contemp undirected edges: 1 per run
            recall_vals = vals.astype(float)  # 0 or 1 per run
            means.append(np.mean(recall_vals))
            sems.append(np.std(recall_vals) / np.sqrt(max(1, len(recall_vals))))
            labels.append(m)

        xs = np.arange(len(METHODS))
        colors = [COLORS[m] for m in METHODS]
        ax.bar(xs, means, yerr=sems, capsize=3, color=colors, alpha=0.8, width=0.7)
        ax.axhline(1.0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(METHODS, rotation=45, ha='right', fontsize=7)
        ax.set_ylim(0, 1.2)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(bm.replace('nonlin', 'nl.'), fontsize=9)
        if col == 0:
            ax.set_ylabel('Contemp-edge recall\n(0/1 per run)')

    # Legend
    patches = [mpatches.Patch(color=COLORS[m], label=m) for m in METHODS]
    fig2.legend(handles=patches, loc='lower center', ncol=6, fontsize=8,
                bbox_to_anchor=(0.5, -0.08))
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        path = os.path.join(OUT_DIR, f'per_edge_type.{ext}')
        fig2.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[figures] Saved {path}", flush=True)
    plt.close(fig2)

    # ── Figure 3: T comparison (T=1000 vs T=5000, noise=1.0) ─────────────────
    # Line plot: F1 vs T for each method, averaged over all models
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4))
    fig3.suptitle('F1 vs Sample Size and SNR\n(averaged over all models)', fontsize=11)

    for ax_idx, (split_col, split_vals, xlabel) in enumerate([
        ('T',     T_VALUES,     'Timepoints T'),
        ('noise', NOISE_VALUES, 'Noise level'),
    ]):
        ax = axes3[ax_idx]
        for m in METHODS:
            ys = []
            xs_plot = []
            for sv in split_vals:
                sub = df[(df['method'] == m) & (df[split_col] == sv)]
                if sub.empty:
                    continue
                ys.append(sub['edges_F1'].mean())
                xs_plot.append(sv)
            if ys:
                ax.plot(xs_plot, ys, marker='o', color=COLORS[m], label=m,
                        linewidth=1.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Mean F1 (all models)')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        path = os.path.join(OUT_DIR, f'f1_vs_regime.{ext}')
        fig3.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[figures] Saved {path}", flush=True)
    plt.close(fig3)

    # ── Print top-line summary ────────────────────────────────────────────────
    print("\n===== TOP-LINE SUMMARY =====", flush=True)
    print("Mean F1 by method (all models, T=1000, noise=1.0):", flush=True)
    sub_main = df[(df['T'] == 1000) & (df['noise'] == 1.0)]
    for m in METHODS:
        vals = sub_main[sub_main['method'] == m]['edges_F1']
        print(f"  {m:12s}: {vals.mean():.3f} +/- {vals.std():.3f}", flush=True)

    print("\nMean F1 on +contemp models only (T=1000, noise=1.0):", flush=True)
    sub_c = df[(df['T'] == 1000) & (df['noise'] == 1.0) & (df['has_contemp'] == 1)]
    for m in METHODS:
        vals = sub_c[sub_c['method'] == m]['edges_F1']
        print(f"  {m:12s}: {vals.mean():.3f} +/- {vals.std():.3f}", flush=True)

    print("\nContemp-edge recovery (+contemp models, T=1000, noise=1.0):", flush=True)
    for m in METHODS:
        vals = sub_c[sub_c['method'] == m]['contemp_tp']
        print(f"  {m:12s}: mean contemp_tp={vals.mean():.3f}", flush=True)


if __name__ == '__main__':
    main()
