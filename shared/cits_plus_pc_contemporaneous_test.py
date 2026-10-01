"""
cits_plus_pc_contemporaneous_test.py

HYPOTHESIS: CITS structurally excludes contemporaneous (lag-0) edges.
Test whether adding PC-derived contemporaneous edges (using the SAME
partial-correlation + Fisher-z + first-S-wins machinery as CITS, but
applied to the contemporaneous time slice of chi) rescues CITS's
area-level Spearman r deficit vs EM directed synapse density.

Four methods compared:
  TPC              -- existing cfc_raw_2026-04-28 (includes lag-0 via rolling)
  CITS             -- existing cits_pc_raw_2026-05-13 (lag-0 structurally absent)
  COMBINED_PCCONTEMP -- CITS + PC-contemporaneous edges (where CITS has none)
  NULL_RANDOM      -- CITS + same-count random edges (control for "more edges")

Plus a reference row from the prior COMBINED_CITS_TPCLAG0 analysis (2026-05-20)
for side-by-side comparison.

PC-contemporaneous extraction (option (a) -- all-PC, same machinery as CITS):
  - For each trial, compute chi = data_transform(X, tau=1), shape (4p, N)
  - Extract contemporaneous slice: chi_contemp = chi[t_target*p:(t_target+1)*p, :]
    where t_target = 2*tau+1 = 3. Shape: (p, N).
  - Run plain PC skeleton on chi_contemp using the same partial-correlation
    conditioning (Fisher-z, alpha=0.05, first-S-wins, GPU-batched residualization
    via _level_l_batched_numba). The conditioning set at each level consists of
    other neurons at the same contemporaneous time slot (all p nodes are in chi_contemp).
  - Edge weight: |Pearson r| between chi_contemp[i,:] and chi_contemp[j,:] at l=0
    (same vectorized corrcoef as CITS's _level0_vectorized). This is the
    marginal correlation; assigned as weight for ALL surviving edges (i.e., edges
    not removed by any conditioning set at any level get the l=0 |r| as weight).
  - Output: (p, p) directed symmetric matrix (undirected skeleton; A[i,j]=A[j,i]=|r_ij|
    for surviving edges, 0 elsewhere, diagonal 0).

COMBINED merge rule:
  COMBINED[i,j] = CITS[i,j]        if CITS[i,j] != 0
                = PC_contemp[i,j]   if CITS[i,j] == 0 AND PC_contemp[i,j] != 0
                = 0                 otherwise
  i == j always 0 (no autapses).
  CITS nonzero values are NEVER overwritten.

NULL-RANDOM: for each field, count N_contemp = number of nonzero PC_contemp
off-diagonal entries not already in CITS. Sample N_contemp random positions
from the non-CITS off-diagonal entries. Assign magnitudes sampled (with
replacement) from the PC_contemp nonzero magnitude distribution.
RNG seed fixed per field (hash of field key) for reproducibility.

Persistent cache: /data1/rb1/microns/saves/cits_pc_contemporaneous_2026-05-23/
  output_{name}_session{S}_scan{SC}_field{F}.csv  -- (p, p) PC-contemp weight matrix
    Row i, col j: |r_ij| (l=0 Pearson r) if edge (i,j) survives PC, else 0.
    Matrix is symmetric. Diagonal is 0.

Two headline tests (matched_df universe = 638,546 ordered pairs, 38 fields, no autapses):
  (a) Pair-level SC fold enrichment: FC-present vs FC-absent, Wilson 95% CI, Fisher exact
  (b) Area-level Spearman r vs EM directed pair density, field-clustered bootstrap 95% CI,
      PAIRED diff vs CITS for COMBINED and NULL-RANDOM

Stats CSV: arousal_paper_overleaf/figures/2026-05-23/cits_plus_pc_contemporaneous_test.csv
Figure:    arousal_paper_overleaf/figures/2026-05-23/fig_area_level_EM_vs_FC_TPC_CITS_PCCONTEMP_NULL.{png,pdf}
"""

import os
# Thread budget: 1 per BLAS operation -- GPU handles heavy linear algebra
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '4')

import sys
import re
import glob
import time
import json
import pickle
import warnings
import traceback
from multiprocessing import Pool, Value, current_process, cpu_count
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.stats import spearmanr, fisher_exact
from statsmodels.stats.proportion import proportion_confint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')

# ── Analysis dir on sys.path for CITS modules ─────────────────────────────────
_ANALYSIS_DIR = '/home/rbiswas1/microns/analysis/functional_circuitry'
if _ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, _ANALYSIS_DIR)

# ── Paths ──────────────────────────────────────────────────────────────────────
SAVES_DIR   = '/data1/rb1/microns/saves/'
TPC_DIR     = os.path.join(SAVES_DIR, 'cfc_raw_2026-04-28/')
CITS_DIR    = os.path.join(SAVES_DIR, 'cits_pc_raw_2026-05-13/')
CONTEMP_DIR = '/data1/rb1/microns/saves/cits_pc_contemporaneous_2026-05-23/'

MATCHED_DF_PATH  = os.path.join(SAVES_DIR, 'matched_df_v1718.pkl')
SYN_CACHE_PATH   = os.path.join(SAVES_DIR, 'synapses_matcheddf_frompre_v1718.pkl')

OVERLEAF    = '/home/rbiswas1/microns/arousal_paper_overleaf'
EM_JSON     = os.path.join(OVERLEAF,
    'figures/2026-05-09/without_autapses/fig1_sc/'
    'em_within_area_density_matcheddf_only_results_v1718.json')
EM_FIELD_KEYS_NPZ = os.path.join(OVERLEAF,
    'figures/2026-04-29/fig4/bootstrap_sf_correlation_13sess.npz')
TPC_NPZ   = os.path.join(OVERLEAF,
    'figures/2026-04-29/fig2/bootstrap_perfield_means_13sess.npz')
CITS_NPZ  = os.path.join(OVERLEAF,
    'figures/2026-05-16_cits/bootstrap_perfield_means_13sess.npz')
PRIOR_COMBINED_CSV = os.path.join(OVERLEAF,
    'figures/2026-05-20/tpc_lag0_into_cits_test.csv')

OUTDIR      = os.path.join(OVERLEAF, 'figures/2026-05-23')
OUT_CSV     = os.path.join(OUTDIR, 'cits_plus_pc_contemporaneous_test.csv')
OUT_FIG_PNG = os.path.join(OUTDIR, 'fig_area_level_EM_vs_FC_TPC_CITS_PCCONTEMP_NULL.png')
OUT_FIG_PDF = os.path.join(OUTDIR, 'fig_area_level_EM_vs_FC_TPC_CITS_PCCONTEMP_NULL.pdf')

AREA_FILE   = '/home/rbiswas1/microns/all_unit_areas.csv'

# ── Parameters ─────────────────────────────────────────────────────────────────
TAU           = 1         # same as CITS
ALPHA         = 0.05      # same as CITS
N_WORKERS     = 8         # one per GPU
N_GPUS        = 8
N_BOOT        = 10000
BOOT_SEED     = 42
AREAS         = ['AL', 'LM', 'RL', 'V1']
ORDERED_PAIRS = [(a, b) for a in AREAS for b in AREAS]
WITHIN_IDX    = [i for i, (a, b) in enumerate(ORDERED_PAIRS) if a == b]
BETWEEN_IDX   = [i for i, (a, b) in enumerate(ORDERED_PAIRS) if a != b]
CAVE_VERSION  = 1718

os.makedirs(CONTEMP_DIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

# ── Spawn guard ────────────────────────────────────────────────────────────────
# Workers spawned by multiprocessing.Pool import this module. All pickled worker
# functions (_worker_init, process_contemp_trial) are defined below this block at
# module level. The analysis code (Sections 0 onward) is inside main() and only
# runs when __name__ == '__main__'. Workers exit here after importing the functions.
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: PC-contemporaneous extraction
#
# For each trial in the EM-field subset:
#   1. Load statement_dfs calcium data for (session, scan, field, trial_name).
#   2. Center per-neuron (same as CITS in regenerate_cits_pc_raw_2026-05-13.py).
#   3. Compute chi = cits_m.data_transform(X, tau=1), shape (4p, N).
#   4. Extract chi_contemp = chi[t_target*p:(t_target+1)*p, :], shape (p, N).
#   5. Run PC skeleton on chi_contemp:
#      - l=0: vectorized corrcoef threshold (same _level0_vectorized logic)
#      - l>=1: GPU-batched residualization (_level_l_batched_numba adapted
#              to the (p, N) chi_contemp)
#      - first-S-wins removal (same as CITS)
#   6. Edge weight: |Pearson r| from chi_contemp corrcoef (l=0).
#      All surviving edges get the l=0 |r| as weight.
#      Non-surviving edges get 0.
#   7. Save (p, p) symmetric matrix to CONTEMP_DIR as CSV.
#
# Bug-checker note: we use chi_contemp (p rows, p nodes) as the chi matrix
# directly in the PC machinery. The adjacency A is initialized as a full p x p
# directed graph (all i!=j). The t_target, tau, n_nodes parameters are
# irrelevant -- we use the chi-submatrix approach so we can reuse
# _level_l_batched_numba directly with the (p, p) A, chi_contemp (p, N),
# and edges_to_test = all (i, j) pairs with i != j.
#
# Specifically: we alias t_target=0, p_local=p, tau_local=0, n_nodes=p
# and edges_to_test = [(0, j, 0) for j in range(p)] but adapt the edge
# indexing. Simpler: we write a custom inline loop that directly calls
# the same residualization code via a wrapper.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70, flush=True)
print("SECTION 1: PC-contemporaneous extraction (per-trial, EM fields only)", flush=True)
print("="*70, flush=True)


def _pc_contemp_skeleton(chi_c, alpha=0.05, use_gpu=True, verbose=False,
                         return_sep_sets=False):
    """Run PC skeleton on chi_c (p, N) using partial-correlation conditional independence.

    This is a vanilla PC implementation reusing the same Fisher-z test and
    GPU-batched residualization as CITS. The contemporaneous chi slice
    chi_c = chi[t_target*p:(t_target+1)*p, :] is the SAME data used
    by CITS's t_target rows -- so this is the exact same partial-correlation
    machinery, just operating on the p x p contemporaneous block instead of
    the full (4p) x (4p) CITS graph.

    Returns:
        A: (p, p) int array, A[i,j]=1 if edge (i,j) survives PC skeleton.
           Matrix is symmetric (undirected skeleton).
           Diagonal is 0.
        r_mat: (p, p) float array, r_mat[i,j] = |Pearson r| from chi_c at l=0.
               Used as edge weights.
        sep_sets (only if return_sep_sets=True): dict[(i, j) -> tuple[int]].
            Separating set used to remove each removed edge (i, j). Required
            by v-structure orientation in _pc_orientation.
    """
    try:
        import torch as _torch
        _TORCH_OK = True
    except Exception:
        _torch = None
        _TORCH_OK = False

    try:
        from cits_pc_skeleton_numba import (
            _level_l_batched_numba, _indep_vec, _enum_combos_into_flat,
            _ncombos,
        )
        from cits_pc_skeleton_optimized import (
            _fisher_r_crit, _is_cond_indep_pcorr,
        )
    except Exception as e:
        raise ImportError(f"Failed to import CITS skeleton modules: {e}")

    p = chi_c.shape[0]
    N = chi_c.shape[1]

    # ── l=0: vectorized Pearson r threshold ──────────────────────────────────
    r_crit = _fisher_r_crit(N, 0, alpha)
    if verbose:
        print(f"  [pc-contemp l=0] p={p}, N={N}, r_crit={r_crit:.4f}", flush=True)

    # Full p x p Pearson correlation on chi_c (same formula as _level0_vectorized)
    R = np.corrcoef(chi_c)   # (p, p); R[i,i]=1 by definition
    abs_R = np.abs(R)
    # Store l=0 |r| for later weight assignment
    r_mat = abs_R.copy()
    np.fill_diagonal(r_mat, 0.0)

    # Initialize A: fully connected undirected (symmetric) p x p adjacency
    # A[i,j] = 1 means edge (i,j) is still in the skeleton
    A = np.ones((p, p), dtype=int)
    np.fill_diagonal(A, 0)  # no self-loops

    sep_sets = {}

    # Remove edges where |r| < r_crit at l=0
    removed_l0 = 0
    for i in range(p):
        for j in range(p):
            if i == j or A[i, j] == 0:
                continue
            if abs_R[i, j] < r_crit:
                A[i, j] = 0
                A[j, i] = 0  # symmetric
                sep_sets[(i, j)] = ()
                sep_sets[(j, i)] = ()
                removed_l0 += 1
    if verbose:
        print(f"  [pc-contemp l=0] removed {removed_l0} directed edges, "
              f"remaining {int((A != 0).sum())}", flush=True)

    # ── l>=1: adapted _level_l_batched_numba ─────────────────────────────────
    # We need to adapt the CITS-specific _level_l_batched_numba to work with
    # a plain (p, p) adjacency and (p, N) chi, instead of the (n_nodes, n_nodes)
    # unrolled adjacency and (n_nodes, N) chi.
    #
    # Strategy: call _level_l_batched_numba with a FAKE t_target, tau, p, n_nodes
    # such that the candidate edges are (0, j, 0) for j in [0, p), where node i
    # maps to t_target*p_param + v. We want node i (in [0,p)) to be i directly.
    # Set: t_target_param=0, p_param=p, tau_param=0, n_nodes_param=p.
    # Then edges_to_test = [(v, v1, t1) for v in range(p) for v1 in range(p) for t1 in range(1,1)]
    # which is EMPTY (tau=0 gives empty range).
    #
    # This doesn't work directly. Instead we call a custom inline PC loop that
    # mirrors _level_l_batched_numba's logic for the (p x p) case.
    # ──────────────────────────────────────────────────────────────────────────

    actual_use_gpu = bool(
        use_gpu and _TORCH_OK and _torch is not None and _torch.cuda.is_available()
    )
    device = _torch.device('cuda:0') if actual_use_gpu else None

    # Pre-upload chi_c to GPU once
    chi_dev = None
    if actual_use_gpu:
        try:
            chi_dev = _torch.from_numpy(chi_c.astype(np.float64)).to(
                device=device, dtype=_torch.float64)
        except Exception:
            actual_use_gpu = False
            chi_dev = None

    def _residualize_and_corr_gpu(Ai_grp, Bi_grp, S_idx_arr):
        """GPU-batched: partial corr of all (Ai, Bi) pairs given conditioning set S_idx_arr."""
        k = len(S_idx_arr)
        if N - k - 3 <= 0:
            return np.full(len(Ai_grp), np.nan)
        S_idx_t = _torch.from_numpy(S_idx_arr.astype(np.int64)).to(device)
        ones = _torch.ones((N, 1), dtype=_torch.float64, device=device)
        Z = _torch.cat([ones, chi_dev[S_idx_t].T], dim=1)  # (N, k+1)
        ZtZ = Z.T @ Z
        try:
            L = _torch.linalg.cholesky(ZtZ)
            use_chol = True
        except Exception:
            use_chol = False
            ZtZ_inv = _torch.linalg.pinv(ZtZ)
        all_idx_set = set(Ai_grp.tolist()) | set(Bi_grp.tolist())
        all_idx = np.array(sorted(all_idx_set), dtype=np.int64)
        idx_t = _torch.from_numpy(all_idx).to(device)
        Y = chi_dev[idx_t]   # (n_unique, N)
        ZtY = Z.T @ Y.T      # (k+1, n_unique)
        if use_chol:
            Beta = _torch.cholesky_solve(ZtY, L)
        else:
            Beta = ZtZ_inv @ ZtY
        res = Y - (Z @ Beta).T   # (n_unique, N)
        res_c = res - res.mean(dim=1, keepdim=True)
        res_n = res_c.norm(dim=1, keepdim=True).clamp_min(1e-30)
        res_unit = res_c / res_n
        pos_map = {int(ix): pi for pi, ix in enumerate(all_idx)}
        A_pos = _torch.tensor([pos_map[a] for a in Ai_grp.tolist()],
                              dtype=_torch.long, device=device)
        B_pos = _torch.tensor([pos_map[b] for b in Bi_grp.tolist()],
                              dtype=_torch.long, device=device)
        rs = (res_unit[A_pos] * res_unit[B_pos]).sum(dim=1).cpu().numpy()
        return rs

    def _residualize_and_corr_cpu(Ai_grp, Bi_grp, S_idx_arr):
        """CPU numpy: partial corr of all (Ai, Bi) pairs given conditioning set S_idx_arr."""
        k = len(S_idx_arr)
        if N - k - 3 <= 0:
            return np.full(len(Ai_grp), np.nan)
        S_idx = S_idx_arr.astype(np.int64)
        Z = np.column_stack([np.ones(N), chi_c[S_idx].T])  # (N, k+1)
        ZtZ = Z.T @ Z
        try:
            L_cpu = np.linalg.cholesky(ZtZ)
            use_chol = True
        except np.linalg.LinAlgError:
            use_chol = False
            ZtZ_inv = np.linalg.pinv(ZtZ)
        all_idx_set = set(Ai_grp.tolist()) | set(Bi_grp.tolist())
        all_idx = np.array(sorted(all_idx_set), dtype=np.int64)
        Y = chi_c[all_idx]   # (n_unique, N)
        ZtY = Z.T @ Y.T
        if use_chol:
            tmp = np.linalg.solve(L_cpu, ZtY)
            Beta = np.linalg.solve(L_cpu.T, tmp)
        else:
            Beta = ZtZ_inv @ ZtY
        res = Y - (Z @ Beta).T
        res_c = res - res.mean(axis=1, keepdims=True)
        res_n = np.linalg.norm(res_c, axis=1, keepdims=True)
        res_n = np.maximum(res_n, 1e-30)
        res_unit = res_c / res_n
        pos_map = {int(ix): pi for pi, ix in enumerate(all_idx)}
        A_pos = np.array([pos_map[a] for a in Ai_grp.tolist()], dtype=np.int64)
        B_pos = np.array([pos_map[b] for b in Bi_grp.tolist()], dtype=np.int64)
        rs = (res_unit[A_pos] * res_unit[B_pos]).sum(axis=1)
        return rs

    residualize_and_corr = (
        _residualize_and_corr_gpu if actual_use_gpu else _residualize_and_corr_cpu
    )

    def _fisher_indep(r, k):
        """Fisher-z independence test: same as _is_cond_indep_pcorr."""
        df = N - k - 3
        if df <= 0:
            return False
        r_c = max(min(r, 0.999999999), -0.999999999)
        z = 0.5 * np.log((1.0 + r_c) / (1.0 - r_c))
        T = np.sqrt(float(df)) * abs(z)
        z_crit = scipy_stats.norm.ppf(1.0 - alpha / 2.0)
        return bool(T < z_crit)

    def _fisher_indep_vec(rs, k):
        """Vectorised Fisher-z independence test."""
        df = N - k - 3
        if df <= 0:
            return np.zeros(len(rs), dtype=bool)
        rs_c = np.clip(rs, -0.999999999, 0.999999999)
        at_one = np.abs(rs) >= 1.0
        z = 0.5 * np.log((1.0 + rs_c) / (1.0 - rs_c))
        T = np.sqrt(float(df)) * np.abs(z)
        z_crit = scipy_stats.norm.ppf(1.0 - alpha / 2.0)
        indep = T < z_crit
        indep[at_one] = False
        return indep

    from itertools import combinations

    l = 1
    while True:
        A_frozen = A.copy()
        A_or = (A_frozen != 0) | (A_frozen.T != 0)

        # Build all (i, j) candidate edges still in A
        # edges_to_test: list of (i, j) with i!=j and A[i,j] != 0
        edges_to_test = [(i, j) for i in range(p) for j in range(p)
                         if i != j and A[i, j] != 0]
        if not edges_to_test:
            break

        # Group tasks by conditioning set S (same as CITS _level_l_batched)
        per_edge_S = {}    # (i, j) -> list of S tuples in enumeration order
        tasks_by_S = {}    # S -> list of (i, j)

        for (i, j) in edges_to_test:
            # Neighbor set: union of adjacency for i and j (excluding i and j)
            nbr_mask = A_or[i] | A_or[j]
            nbr_mask[i] = False
            nbr_mask[j] = False
            nbrs_arr = np.flatnonzero(nbr_mask)
            if nbrs_arr.size < l:
                continue
            nbrs_sorted = nbrs_arr.tolist()
            S_list = list(combinations(nbrs_sorted, l))
            per_edge_S[(i, j)] = S_list
            for S in S_list:
                tasks_by_S.setdefault(S, []).append((i, j))

        if not per_edge_S:
            break

        if verbose:
            n_tasks = sum(len(v) for v in tasks_by_S.values())
            print(f"  [pc-contemp l={l}] edges={len(per_edge_S)} "
                  f"unique_S={len(tasks_by_S)} total_tests={n_tasks}", flush=True)

        # For each unique S, run batched residualization
        per_edge_result = {ek: {} for ek in per_edge_S}

        for S, edge_list in tasks_by_S.items():
            k = len(S)
            S_idx_arr = np.array(list(S), dtype=np.int32)
            Ai_grp = np.array([e[0] for e in edge_list], dtype=np.int32)
            Bi_grp = np.array([e[1] for e in edge_list], dtype=np.int32)
            rs = residualize_and_corr(Ai_grp, Bi_grp, S_idx_arr)
            indep_flags = _fisher_indep_vec(
                np.where(np.isnan(rs), 0.0, rs), k)
            for idx_e, (ei, ej) in enumerate(edge_list):
                if np.isnan(rs[idx_e]):
                    per_edge_result[(ei, ej)][S] = (np.nan, False)
                else:
                    per_edge_result[(ei, ej)][S] = (
                        float(rs[idx_e]), bool(indep_flags[idx_e]))

        # First-S-wins removal scan
        any_removed = False
        for (i, j), S_list in per_edge_S.items():
            if A[i, j] == 0:
                continue
            for S in S_list:
                r_val, indep = per_edge_result.get((i, j), {}).get(S, (np.nan, False))
                if indep:
                    A[i, j] = 0
                    # Also remove the symmetric edge (undirected skeleton)
                    A[j, i] = 0
                    sep_sets[(i, j)] = tuple(S)
                    sep_sets[(j, i)] = tuple(S)
                    any_removed = True
                    break

        if verbose:
            print(f"  [pc-contemp l={l}] done: edges remaining {int((A != 0).sum())}, "
                  f"any_removed={any_removed}", flush=True)

        if not any_removed:
            break
        l += 1

    # Apply l=0 |r| as weight for surviving edges
    weight_mat = np.where(A != 0, r_mat, 0.0)
    np.fill_diagonal(weight_mat, 0.0)

    if return_sep_sets:
        return A, weight_mat, sep_sets
    return A, weight_mat


# ── Worker initializer (GPU pin, same as CITS driver) ─────────────────────────

def _worker_init(gpu_counter):
    """Pin worker to one GPU, same approach as CITS driver."""
    with gpu_counter.get_lock():
        wid = gpu_counter.value
        gpu_counter.value += 1
    gpu_id = wid % N_GPUS
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    try:
        import torch  # noqa
    except Exception:
        pass
    current_process()._contemp_wid = wid
    current_process()._contemp_gpu = gpu_id


# Per-worker session cache (LRU=1)
_sess_cache = {'key': None, 'names': None, 'dfs': None}


def _ensure_session(session, scan_idx):
    key = (session, scan_idx)
    if _sess_cache['key'] == key:
        return _sess_cache['names'], _sess_cache['dfs']
    pkl_path = os.path.join(SAVES_DIR,
        f'statement_dfs_session{session}_scan{scan_idx}.pkl')
    with open(pkl_path, 'rb') as fh:
        names, dfs = pickle.load(fh)
    _sess_cache['key'] = key
    _sess_cache['names'] = names
    _sess_cache['dfs'] = dfs
    return names, dfs


def process_contemp_trial(args):
    """Worker: extract PC-contemporaneous matrix for one (session, scan, field, name).

    Preprocessing (centering): same as CITS (regenerate_cits_pc_raw_2026-05-13.py):
      f_centered = f_raw - f_raw.mean(axis=1, keepdims=True)

    chi computation: cits_m.data_transform(f_centered, tau=1), same as CITS.
    chi_contemp: chi[t_target*p:(t_target+1)*p, :] where t_target = 2*tau+1 = 3.

    Output: (p, p) symmetric weight matrix, |r| at l=0 for surviving edges, 0 elsewhere.
    Saved as CSV (same format as CITS: no header, no index).
    """
    session, scan_idx, field, name, out_dir = args
    task_info = f"S{session}Sc{scan_idx}F{field}_{name}"
    out_path = os.path.join(out_dir,
        f'output_{name}_session{session}_scan{scan_idx}_field{field}.csv')

    if os.path.exists(out_path):
        return ('skipped_exists', task_info, None, out_path, 0.0, None)

    t_start = time.time()
    try:
        # Lazy import so CUDA_VISIBLE_DEVICES is respected
        from cits import methods as cits_m

        names, dfs = _ensure_session(session, scan_idx)
        if name not in names:
            return ('fail_no_trial', task_info, f"trial '{name}' not in session",
                    out_path, 0.0, None)

        df = dfs[names.index(name)]
        if not isinstance(df, pd.DataFrame) or df.empty:
            return ('fail_empty_trial', task_info, "empty df", out_path, 0.0, None)

        ca_cols = [c for c in df.columns if c.startswith('calcium')]
        if not ca_cols:
            return ('fail_no_calcium', task_info, "no calcium columns",
                    out_path, 0.0, None)

        df_field = df[df['field'] == field].sort_values(by='ID')
        if df_field.empty:
            return ('fail_no_field', task_info, f"no data for field {field}",
                    out_path, 0.0, None)

        f_raw = df_field[ca_cols].values.astype(np.float64)
        n_neurons, n_obs = f_raw.shape

        if n_neurons < 2:
            return ('fail_few_neurons', task_info, f"only {n_neurons} neuron(s)",
                    out_path, 0.0, n_neurons)
        if n_obs < 100:
            return ('fail_few_obs', task_info, f"only {n_obs} timepoints",
                    out_path, 0.0, n_neurons)

        # Center per-neuron (same as CITS)
        f_centered = f_raw - f_raw.mean(axis=1, keepdims=True)

        # Compute chi with tau=1 (same as CITS)
        chi_full = cits_m.data_transform(f_centered, TAU)
        # chi_full shape: (4p, N) where N = floor((n_obs - 4) / 4)
        t_target = 2 * TAU + 1  # = 3 for tau=1
        p = n_neurons
        chi_contemp = chi_full[t_target * p:(t_target + 1) * p, :]
        # chi_contemp shape: (p, N) -- contemporaneous time slice

        N_contemp = chi_contemp.shape[1]
        if N_contemp < 10:
            return ('fail_few_windows', task_info,
                    f"only {N_contemp} time windows after transform",
                    out_path, 0.0, n_neurons)

        # Run PC skeleton on chi_contemp
        _A, weight_mat = _pc_contemp_skeleton(
            chi_contemp, alpha=ALPHA, use_gpu=True, verbose=False)

        # Sanity: diagonal must be 0, matrix must be symmetric
        np.fill_diagonal(weight_mat, 0.0)

        # Atomic write
        tmp_path = out_path + '.tmp'
        pd.DataFrame(weight_mat).to_csv(tmp_path, index=False, header=False)
        os.replace(tmp_path, out_path)

        elapsed = time.time() - t_start
        n_edges = int((_A != 0).sum())
        return ('ok', task_info, None, out_path, elapsed, n_neurons)

    except MemoryError as e:
        return ('fail_oom', task_info, f"MemoryError: {e}",
                out_path, time.time() - t_start, None)
    except Exception as e:
        tb = traceback.format_exc()[-400:]
        return ('fail_other', task_info, f"{type(e).__name__}: {e}\n{tb}",
                out_path, time.time() - t_start, None)


if __name__ == '__main__':
    import multiprocessing as _mp
    try:
        _mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # ── SECTION 0: Load EM field keys ──────────────────────────────────────────
    print("\n" + "="*70, flush=True)
    print("SECTION 0: Load EM field keys", flush=True)
    print("="*70, flush=True)
    d_sc = np.load(EM_FIELD_KEYS_NPZ, allow_pickle=True)
    em_field_keys = [tuple(int(x) for x in k) for k in d_sc['field_keys']]
    em_field_set  = set(em_field_keys)
    print(f"  EM fields: {len(em_field_keys)}", flush=True)
    for _k in em_field_keys:
        print(f"    {_k}", flush=True)

    print(f"PC-contemp cache:  {CONTEMP_DIR}", flush=True)
    print(f"Output directory:  {OUTDIR}", flush=True)
    print(f"CPUs available:    {cpu_count()}", flush=True)

    # ── Build task list (only EM-field subset) ─────────────────────────────────

    print("Building task list for EM-field subset ...", flush=True)

    # Collect tasks by matching (name, session, scan, field) to the EM field set
    # Load statement_dfs to enumerate available trials
    all_tasks = []
    sdf_files = sorted(
        f for f in glob.glob(os.path.join(SAVES_DIR, 'statement_dfs_session*_scan*.pkl'))
        if 'OLD_BROKEN' not in os.path.basename(f)
        and 'fixed' not in os.path.basename(f)
    )

    for sdf_path in sdf_files:
        m = re.search(r'session(\d+)_scan(\d+)', os.path.basename(sdf_path))
        if not m:
            continue
        session, scan_idx = int(m.group(1)), int(m.group(2))

        # Find which EM fields belong to this session/scan
        local_fields = [f for (s, sc, f) in em_field_keys if s == session and sc == scan_idx]
        if not local_fields:
            continue

        try:
            with open(sdf_path, 'rb') as fh:
                names, dfs = pickle.load(fh)
        except Exception as e:
            print(f"  WARN: could not load {sdf_path}: {e}", flush=True)
            continue

        for name, df in zip(names, dfs):
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            for field in local_fields:
                if (session, scan_idx, field) not in em_field_set:
                    continue
                all_tasks.append((session, scan_idx, field, name, CONTEMP_DIR))

    print(f"  Total tasks: {len(all_tasks)}", flush=True)
    already_done = sum(
        1 for (s, sc, f, n, od) in all_tasks
        if os.path.exists(os.path.join(od, f'output_{n}_session{s}_scan{sc}_field{f}.csv'))
    )
    print(f"  Already cached: {already_done}/{len(all_tasks)}", flush=True)

    # Sort heaviest tasks first (approximate: no p info, use field as proxy)
    all_tasks.sort(key=lambda t: (t[0], t[1], t[2]))

    # Time estimate
    to_run = len(all_tasks) - already_done
    per_trial_est_s = 300   # ~5 min/trial (PC-contemp is faster than CITS due to smaller chi)
    serial_est_h = to_run * per_trial_est_s / 3600
    parallel_est_h = serial_est_h / N_WORKERS
    print(f"\nTime estimate (est {per_trial_est_s}s/trial):", flush=True)
    print(f"  Serial:   {serial_est_h:.1f} h", flush=True)
    print(f"  Parallel ({N_WORKERS} workers): {parallel_est_h:.2f} h", flush=True)

    if to_run == 0:
        print("All tasks already cached. Skipping pool run.", flush=True)
    else:
        import multiprocessing as mp
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        gpu_counter = Value('i', 0)
        t0 = time.time()
        print(f"\nStarting Pool({N_WORKERS}) for PC-contemporaneous extraction ...", flush=True)

        MAX_RESTARTS = 3
        attempt = 0
        results = []
        remaining = list(all_tasks)

        while remaining and attempt <= MAX_RESTARTS:
            if attempt > 0:
                print(f"[POOL RESTART {attempt}/{MAX_RESTARTS}] "
                      f"{len(remaining)} tasks remaining.", flush=True)
            try:
                with Pool(N_WORKERS, initializer=_worker_init,
                          initargs=(gpu_counter,)) as pool:
                    it = pool.imap_unordered(process_contemp_trial, remaining, chunksize=1)
                    for r in it:
                        results.append(r)
                        status, info, msg, _out, dt, n = r
                        if status.startswith('fail'):
                            short = (msg or '')[:200].replace('\n', ' | ')
                            print(f"  FAIL [{status}] {info} dt={dt:.1f}s :: {short}",
                                  flush=True)
                        i = len(results)
                        if i % 5 == 0 or i == len(all_tasks):
                            n_ok = sum(1 for x in results if x[0] == 'ok')
                            n_sk = sum(1 for x in results if x[0] == 'skipped_exists')
                            n_fa = sum(1 for x in results if x[0].startswith('fail'))
                            elapsed = time.time() - t0
                            rate = i / max(elapsed, 1e-9)
                            eta = (len(all_tasks) - i) / max(rate, 1e-9)
                            print(f"  [{i}/{len(all_tasks)}] elapsed={elapsed:.0f}s "
                                  f"ok={n_ok} skip={n_sk} fail={n_fa} eta={eta:.0f}s",
                                  flush=True)
                    remaining = []
            except (BrokenPipeError, EOFError, RuntimeError, OSError) as e:
                print(f"[POOL CRASH] {type(e).__name__}: {e}", flush=True)
                done_set = set()
                for fp in glob.glob(os.path.join(CONTEMP_DIR, 'output_*.csv')):
                    mx = re.search(
                        r'output_([a-z]+_pupil\d+)_session(\d+)_scan(\d+)_field(\d+)\.csv',
                        os.path.basename(fp))
                    if mx:
                        done_set.add((mx.group(1), int(mx.group(2)),
                                      int(mx.group(3)), int(mx.group(4))))
                remaining = [t for t in all_tasks
                             if (t[3], t[0], t[1], t[2]) not in done_set]
                attempt += 1
                if attempt > MAX_RESTARTS:
                    print("Exhausted restarts; proceeding with partial cache.", flush=True)
                    break
                time.sleep(5)

        print(f"\nPool done in {time.time()-t0:.1f}s", flush=True)
        from collections import Counter
        sc = Counter(r[0] for r in results)
        for st, cnt in sorted(sc.items()):
            print(f"  {st:<25} {cnt:>4}", flush=True)


    # ══════════════════════════════════════════════════════════════════════════════
    # SECTION 2: Build per-field mean(|.|) matrices for PC-contemp
    #
    # For each EM field: load all trial CSVs from CONTEMP_DIR, stack and mean.
    # Result: field_contemp_mean[(sess, scan, field)] = (p, p) mean |.| matrix
    # ══════════════════════════════════════════════════════════════════════════════

    print("\n" + "="*70, flush=True)
    print("SECTION 2: Build per-field PC-contemp mean matrices", flush=True)
    print("="*70, flush=True)


    def load_trial_mats(field_dir, sess, scan, field, state_prefix=None):
        """Load all trial CSVs for (sess, scan, field) from field_dir.

        If state_prefix is None, loads all matching files.
        Returns mean(|.|) across trials, or None if no files found.
        """
        patterns = []
        if state_prefix is None:
            patterns += glob.glob(os.path.join(
                field_dir,
                f'output_active_pupil*_session{sess}_scan{scan}_field{field}.csv'))
            patterns += glob.glob(os.path.join(
                field_dir,
                f'output_inactive_pupil*_session{sess}_scan{scan}_field{field}.csv'))
        else:
            patterns = glob.glob(os.path.join(
                field_dir,
                f'output_{state_prefix}_pupil*_session{sess}_scan{scan}_field{field}.csv'))
        if not patterns:
            return None
        mats = []
        for fp in sorted(patterns):
            try:
                m = np.nan_to_num(
                    pd.read_csv(fp, header=None).to_numpy().astype(np.float64))
                mats.append(m)
            except Exception:
                continue
        if not mats:
            return None
        shape0 = mats[0].shape
        mats_ok = [m for m in mats if m.shape == shape0]
        if not mats_ok:
            return None
        return np.mean(np.abs(np.stack(mats_ok, axis=0)), axis=0)


    field_contemp_mean = {}   # (sess, scan, field) -> (p, p) mean |contemp| matrix
    field_cits_mean    = {}   # (sess, scan, field) -> (p, p) mean |CITS|
    field_tpc_mean     = {}   # (sess, scan, field) -> (p, p) mean |TPC|

    for key in em_field_keys:
        sess, scan, field = key
        tkey = tuple(key)

        contemp_mat = load_trial_mats(CONTEMP_DIR, sess, scan, field)
        cits_mat    = load_trial_mats(CITS_DIR, sess, scan, field)
        tpc_mat     = load_trial_mats(TPC_DIR, sess, scan, field)

        if contemp_mat is not None:
            np.fill_diagonal(contemp_mat, 0.0)
            field_contemp_mean[tkey] = contemp_mat

        if cits_mat is not None:
            np.fill_diagonal(cits_mat, 0.0)
            field_cits_mean[tkey] = cits_mat

        if tpc_mat is not None:
            np.fill_diagonal(tpc_mat, 0.0)
            field_tpc_mean[tkey] = tpc_mat

        n_contemp = int((contemp_mat != 0).sum()) if contemp_mat is not None else -1
        n_cits    = int((cits_mat != 0).sum())    if cits_mat is not None else -1
        print(f"  {tkey}: contemp_nnz={n_contemp}, cits_nnz={n_cits}", flush=True)

    print(f"\nFields with PC-contemp: {len(field_contemp_mean)}", flush=True)
    print(f"Fields with CITS:       {len(field_cits_mean)}", flush=True)
    print(f"Fields with TPC:        {len(field_tpc_mean)}", flush=True)


    # ══════════════════════════════════════════════════════════════════════════════
    # SECTION 3: Build COMBINED_PCCONTEMP and NULL_RANDOM per-field matrices
    #
    # COMBINED merge rule:
    #   COMBINED[i,j] = CITS[i,j]       if CITS[i,j] != 0
    #                = PC_contemp[i,j]  if CITS[i,j] == 0 AND PC_contemp[i,j] != 0
    #                = 0                otherwise
    #   Diagonal always 0. CITS nonzero values never overwritten.
    #
    # NULL_RANDOM: same count of random edges as the PC_contemp-added set,
    #   magnitudes from PC_contemp empirical distribution.
    #   5 random draws per field; we report median across draws.
    # ══════════════════════════════════════════════════════════════════════════════

    print("\n" + "="*70, flush=True)
    print("SECTION 3: Build COMBINED and NULL_RANDOM matrices", flush=True)
    print("="*70, flush=True)

    field_combined_mean   = {}   # (sess, scan, field) -> COMBINED matrix
    field_null_random_mats = {}  # (sess, scan, field) -> list of 5 NULL matrices

    N_NULL_DRAWS = 5

    for key in em_field_keys:
        tkey = tuple(key)
        sess, scan, field = key

        if tkey not in field_cits_mean or tkey not in field_contemp_mean:
            continue

        cits = field_cits_mean[tkey]
        contemp = field_contemp_mean[tkey]

        if cits.shape != contemp.shape:
            print(f"  SKIP {tkey}: shape mismatch cits={cits.shape} contemp={contemp.shape}",
                  flush=True)
            continue

        p = cits.shape[0]

        # ── COMBINED ──────────────────────────────────────────────────────────────
        combined = np.where(cits != 0, cits, contemp)
        np.fill_diagonal(combined, 0.0)
        field_combined_mean[tkey] = combined

        # ── NULL_RANDOM ───────────────────────────────────────────────────────────
        # Count how many PC_contemp edges are being injected (not already in CITS)
        cits_zero_offdiag  = (cits == 0) & (~np.eye(p, dtype=bool))
        contemp_inject     = (contemp != 0) & cits_zero_offdiag
        N_inject = int(contemp_inject.sum())

        cand_mask = cits_zero_offdiag & (~contemp_inject)
        cand_positions = np.argwhere(cand_mask)
        M_cand = len(cand_positions)

        # Magnitudes from PC_contemp nonzero injection set
        contemp_magnitudes = contemp[contemp_inject]  # nonzero values

        null_mats = []
        for draw in range(N_NULL_DRAWS):
            null_mat = cits.copy()
            np.fill_diagonal(null_mat, 0.0)
            if N_inject > 0 and M_cand > 0 and len(contemp_magnitudes) > 0:
                rng_null = np.random.default_rng(abs(hash(tkey)) % (2**31) + draw * 997)
                n_inject = min(N_inject, M_cand)
                chosen_idx = rng_null.choice(M_cand, size=n_inject, replace=False)
                chosen_pos = cand_positions[chosen_idx]
                mags = rng_null.choice(contemp_magnitudes, size=n_inject, replace=True)
                for (i, j), mag in zip(chosen_pos, mags):
                    null_mat[i, j] = mag
            null_mats.append(null_mat)
        field_null_random_mats[tkey] = null_mats

        cits_nnz    = int((cits != 0).sum()) - p
        comb_nnz    = int((combined != 0).sum()) - p
        print(f"  {tkey}: p={p}, cits_edges={cits_nnz}, "
              f"contemp_inject={N_inject}, combined_edges={comb_nnz}", flush=True)

    print(f"\nFields with COMBINED: {len(field_combined_mean)}", flush=True)
    print(f"Fields with NULL_RANDOM: {len(field_null_random_mats)}", flush=True)


    # ══════════════════════════════════════════════════════════════════════════════
    # SECTION 4: Pair-level SC fold enrichment
    #
    # Universe: all matched_df v1718 unit pairs within the same field,
    # no autapses, pair-once-per-field. 638,546 ordered pairs, 38 fields.
    #
    # FC-present: mean(|.|) > 0 in the per-field pooled matrix.
    # SC+: pair (pre_root, post_root) in the global synaptic pair set (no autapses).
    #
    # Methods: TPC, CITS, COMBINED_PCCONTEMP, NULL_RANDOM (median of 5 draws)
    #
    # Statistics per method:
    #   N_fc_present, N_fc_absent
    #   N_fcpres_syn, N_fcabs_syn
    #   frac_present, frac_absent
    #   Wilson 95% CI on frac_present
    #   fold enrichment = frac_present / frac_absent
    #   Fisher exact OR and p-value
    # ══════════════════════════════════════════════════════════════════════════════

    print("\n" + "="*70, flush=True)
    print("SECTION 4: Pair-level SC fold enrichment", flush=True)
    print("="*70, flush=True)

    # Load matched_df
    print("Loading matched_df_v1718 ...", flush=True)
    matched_df_raw = pickle.load(open(MATCHED_DF_PATH, 'rb'))
    mv = matched_df_raw.dropna(subset=['pt_root_id_v1718']).copy()
    mv = mv[mv['pt_root_id_v1718'] > 0]
    unit_to_pt = {
        (int(row.session), int(row.scan_idx), int(row.unit_id)): int(row.pt_root_id_v1718)
        for row in mv.itertuples()
    }
    print(f"  unit->root mappings: {len(unit_to_pt):,}", flush=True)

    # Load synapses cache
    print("Loading synapses cache ...", flush=True)
    syn_cache = pickle.load(open(SYN_CACHE_PATH, 'rb'))
    sc_pos_set = set()
    for pre_root, df_syn in syn_cache.items():
        if (not hasattr(df_syn, 'columns') or 'post_pt_root_id' not in df_syn.columns
                or df_syn.empty):
            continue
        pre_r = int(pre_root)
        for post_r in df_syn['post_pt_root_id'].values:
            pr2 = int(post_r)
            if pre_r != pr2:
                sc_pos_set.add((pre_r, pr2))
    print(f"  SC+ ordered pairs (no autapses): {len(sc_pos_set):,}", flush=True)

    # Load area info for field unit_ids
    area_df = pd.read_csv(AREA_FILE)
    area_map = {}
    for _, row in area_df.iterrows():
        area_map[(int(row.session), int(row.scan_idx), int(row.unit_id))] = row.brain_area

    # Build unit_ids per field from statement_dfs
    field_unit_ids = {}   # (sess, scan, field) -> list of unit_ids in sorted order
    for sdf_path in sorted(glob.glob(os.path.join(SAVES_DIR,
            'statement_dfs_session*_scan*.pkl'))):
        if 'OLD_BROKEN' in sdf_path or 'fixed' in sdf_path:
            continue
        m = re.search(r'session(\d+)_scan(\d+)', os.path.basename(sdf_path))
        if not m:
            continue
        sess_l, scan_l = int(m.group(1)), int(m.group(2))
        local_fields = [f for (s, sc, f) in em_field_keys if s == sess_l and sc == scan_l]
        if not local_fields:
            continue
        try:
            with open(sdf_path, 'rb') as fh:
                names_l, dfs_l = pickle.load(fh)
        except Exception:
            continue
        # Use first non-empty df to get unit_ids per field
        for df_l in dfs_l:
            if isinstance(df_l, pd.DataFrame) and not df_l.empty and 'ID' in df_l.columns:
                for field_l in local_fields:
                    tkey_l = (sess_l, scan_l, field_l)
                    if tkey_l not in field_unit_ids:
                        df_f = df_l[df_l['field'] == field_l].sort_values('ID')
                        if not df_f.empty:
                            field_unit_ids[tkey_l] = list(df_f['ID'].values)

    print(f"  Field unit_ids loaded for: {len(field_unit_ids)} fields", flush=True)

    # Accumulate pair-level SC enrichment counts for each method
    # We need per-field FC-present sets, then aggregate
    N_fc_pres  = {'TPC': 0, 'CITS': 0, 'COMBINED': 0, 'NULL_RANDOM': [0]*N_NULL_DRAWS}
    N_fc_abs   = {'TPC': 0, 'CITS': 0, 'COMBINED': 0, 'NULL_RANDOM': [0]*N_NULL_DRAWS}
    N_fps_syn  = {'TPC': 0, 'CITS': 0, 'COMBINED': 0, 'NULL_RANDOM': [0]*N_NULL_DRAWS}
    N_fas_syn  = {'TPC': 0, 'CITS': 0, 'COMBINED': 0, 'NULL_RANDOM': [0]*N_NULL_DRAWS}
    N_universe = 0
    fields_in_enrichment = []

    for tkey in em_field_keys:
        sess, scan, field = tkey

        if tkey not in field_unit_ids:
            print(f"  SKIP {tkey}: no unit_ids", flush=True)
            continue

        unit_ids = field_unit_ids[tkey]
        p = len(unit_ids)

        # Root IDs for SC lookup
        root_ids = [unit_to_pt.get((sess, scan, int(uid)), None) for uid in unit_ids]

        # Load the four method matrices
        tpc_mat    = field_tpc_mean.get(tkey)
        cits_mat   = field_cits_mean.get(tkey)
        comb_mat   = field_combined_mean.get(tkey)
        null_mats  = field_null_random_mats.get(tkey, [])

        # Check shapes
        def _shape_ok(mat):
            return mat is not None and mat.shape == (p, p)

        if not _shape_ok(tpc_mat) and not _shape_ok(cits_mat):
            print(f"  SKIP {tkey}: no TPC or CITS matrix with matching shape", flush=True)
            continue

        fields_in_enrichment.append(tkey)

        # Iterate over all ordered pairs (i, j) with i != j
        for i in range(p):
            for j in range(p):
                if i == j:
                    continue
                root_i = root_ids[i]
                root_j = root_ids[j]
                if root_i is None or root_j is None:
                    continue

                # No autapses in the SC lookup (already filtered in sc_pos_set)
                if root_i == root_j:
                    continue

                sc_flag = (root_i, root_j) in sc_pos_set
                N_universe += 1

                for method, mat in [('TPC', tpc_mat), ('CITS', cits_mat), ('COMBINED', comb_mat)]:
                    if not _shape_ok(mat):
                        continue
                    fc_pres = bool(mat[i, j] != 0)
                    if fc_pres:
                        N_fc_pres[method] += 1
                        if sc_flag:
                            N_fps_syn[method] += 1
                    else:
                        N_fc_abs[method] += 1
                        if sc_flag:
                            N_fas_syn[method] += 1

                for draw_idx, null_mat in enumerate(null_mats):
                    if null_mat is None or null_mat.shape != (p, p):
                        continue
                    fc_pres = bool(null_mat[i, j] != 0)
                    if fc_pres:
                        N_fc_pres['NULL_RANDOM'][draw_idx] += 1
                        if sc_flag:
                            N_fps_syn['NULL_RANDOM'][draw_idx] += 1
                    else:
                        N_fc_abs['NULL_RANDOM'][draw_idx] += 1
                        if sc_flag:
                            N_fas_syn['NULL_RANDOM'][draw_idx] += 1

    print(f"\nUniverse size: {N_universe:,}", flush=True)
    print(f"Fields in enrichment: {len(fields_in_enrichment)}", flush=True)

    # Compute NULL_RANDOM summary (median across 5 draws)
    null_frac_pres_all  = []
    null_frac_abs_all   = []
    null_fold_all       = []
    null_fps_syn_all    = []
    null_fabs_syn_all   = []

    for draw_idx in range(N_NULL_DRAWS):
        fps = N_fps_syn['NULL_RANDOM'][draw_idx]
        fas = N_fas_syn['NULL_RANDOM'][draw_idx]
        fpres = N_fc_pres['NULL_RANDOM'][draw_idx]
        fabs  = N_fc_abs['NULL_RANDOM'][draw_idx]
        if fpres > 0:
            fp_val = fps / fpres
        else:
            fp_val = np.nan
        if fabs > 0:
            fa_val = fas / fabs
        else:
            fa_val = np.nan
        null_frac_pres_all.append(fp_val)
        null_frac_abs_all.append(fa_val)
        null_fold_all.append(fp_val / fa_val if fa_val > 0 else np.nan)
        null_fps_syn_all.append(fps)
        null_fabs_syn_all.append(fas)

    # Use draw 0 for Fisher test (representative)
    null_fps_syn_med  = int(np.median(null_fps_syn_all))
    null_fabs_syn_med = int(np.median(null_fabs_syn_all))
    null_fpres_med    = int(np.median([N_fc_pres['NULL_RANDOM'][d] for d in range(N_NULL_DRAWS)]))
    null_fabs_med     = int(np.median([N_fc_abs['NULL_RANDOM'][d] for d in range(N_NULL_DRAWS)]))

    # Build enrichment results table
    def compute_enrichment(n_fps, n_fas, n_fp, n_fa, n_fields):
        """Compute SC enrichment stats for one method."""
        if n_fp == 0 or n_fa == 0:
            return {}
        frac_pres = n_fps / n_fp
        frac_abs  = n_fas / n_fa
        fold = frac_pres / frac_abs if frac_abs > 0 else np.nan
        ci_lo_pres, ci_hi_pres = proportion_confint(n_fps, n_fp, alpha=0.05, method='wilson')
        ci_lo_abs,  ci_hi_abs  = proportion_confint(n_fas, n_fa, alpha=0.05, method='wilson')
        ct = np.array([[n_fps, n_fp - n_fps], [n_fas, n_fa - n_fas]])
        fisher_OR, fisher_p = fisher_exact(ct, alternative='two-sided')
        return {
            'N_fc_present':    n_fp,
            'N_fc_absent':     n_fa,
            'N_fcpres_syn':    n_fps,
            'N_fcabs_syn':     n_fas,
            'frac_present':    frac_pres,
            'ci_lo_present':   ci_lo_pres,
            'ci_hi_present':   ci_hi_pres,
            'frac_absent':     frac_abs,
            'ci_lo_absent':    ci_lo_abs,
            'ci_hi_absent':    ci_hi_abs,
            'fold_enrichment': fold,
            'fisher_OR':       fisher_OR,
            'fisher_p':        fisher_p,
            'n_fields':        n_fields,
            'universe':        N_universe,
        }

    enrichment = {}
    n_flds = len(fields_in_enrichment)
    enrichment['TPC']      = compute_enrichment(
        N_fps_syn['TPC'], N_fas_syn['TPC'],
        N_fc_pres['TPC'], N_fc_abs['TPC'], n_flds)
    enrichment['CITS']     = compute_enrichment(
        N_fps_syn['CITS'], N_fas_syn['CITS'],
        N_fc_pres['CITS'], N_fc_abs['CITS'], n_flds)
    enrichment['COMBINED'] = compute_enrichment(
        N_fps_syn['COMBINED'], N_fas_syn['COMBINED'],
        N_fc_pres['COMBINED'], N_fc_abs['COMBINED'], n_flds)
    enrichment['NULL_RANDOM'] = compute_enrichment(
        null_fps_syn_med, null_fabs_syn_med,
        null_fpres_med, null_fabs_med, n_flds)

    print("\n=== Pair-level SC enrichment results ===", flush=True)
    for method in ['TPC', 'CITS', 'COMBINED', 'NULL_RANDOM']:
        e = enrichment.get(method, {})
        if not e:
            print(f"  {method}: (no data)", flush=True)
            continue
        print(f"  {method}:", flush=True)
        print(f"    FC-present: {e['N_fc_present']:,}  |  FC-absent: {e['N_fc_absent']:,}", flush=True)
        print(f"    SC+ | FC-pres: {e['N_fcpres_syn']}  |  SC+ | FC-abs: {e['N_fcabs_syn']:,}", flush=True)
        print(f"    frac_pres={e['frac_present']:.6f} [{e['ci_lo_present']:.6f}, {e['ci_hi_present']:.6f}]", flush=True)
        print(f"    fold={e['fold_enrichment']:.4f}  OR={e['fisher_OR']:.4f}  p={e['fisher_p']:.3e}", flush=True)


    # ══════════════════════════════════════════════════════════════════════════════
    # SECTION 5: Area-level Spearman r vs EM directed pair density
    #
    # Use the same NPZ-based per-field approach as em_vs_fc_area_correlation.py
    # for TPC and CITS (authoritative point estimates), and build a parallel
    # per-field structure for COMBINED and NULL_RANDOM using the on-disk matrices.
    #
    # For COMBINED and NULL_RANDOM: build per-field (16,) density and strength vectors
    # (one value per ordered area pair). Density = fraction of matched-df within-field
    # pairs with FC-present. Strength = mean(|.|) over all pairs in the area pair.
    #
    # Field-clustered bootstrap: resample fields with replacement (n=38), compute
    # nanmean per pair, then Spearman r vs fixed EM vector. TPC and CITS use the
    # same approach via their NPZ matrices for cross-method consistency.
    # ══════════════════════════════════════════════════════════════════════════════

    print("\n" + "="*70, flush=True)
    print("SECTION 5: Area-level Spearman r vs EM density (field-bootstrap)", flush=True)
    print("="*70, flush=True)

    # Load EM density vector
    with open(EM_JSON, 'r') as f:
        em_data = json.load(f)

    em_dens = {}
    for area, r in em_data['within_directed_B'].items():
        em_dens[(area, area)] = r['density']
    for key_em, r in em_data['between_directed_B'].items():
        src, tgt = key_em.split('->')
        em_dens[(src, tgt)] = r['density']
    em_vec = np.array([em_dens.get((a, b), np.nan) for a, b in ORDERED_PAIRS])
    print(f"  EM density range: [{np.nanmin(em_vec):.6f}, {np.nanmax(em_vec):.6f}]", flush=True)

    # Load TPC and CITS per-field data from authoritative NPZ files
    def load_fc_perfield_npz(npz_path):
        """Load per-field dens and str from bootstrap_perfield_means NPZ."""
        d = np.load(npz_path, allow_pickle=True)
        out = {}
        for a, b in ORDERED_PAIRS:
            try:
                dens_low  = d[f'{a}_{b}_dens_low'].astype(float)
                dens_high = d[f'{a}_{b}_dens_high'].astype(float)
                str_low   = d[f'{a}_{b}_str_low'].astype(float)
                str_high  = d[f'{a}_{b}_str_high'].astype(float)
                dens_pool = np.where(
                    np.isnan(dens_low) & np.isnan(dens_high), np.nan,
                    np.nanmean(np.stack([dens_low, dens_high], axis=1), axis=1))
                str_pool = np.where(
                    np.isnan(str_low) & np.isnan(str_high), np.nan,
                    np.nanmean(np.stack([str_low, str_high], axis=1), axis=1))
                out[(a, b)] = {'dens': dens_pool, 'str': str_pool}
            except KeyError:
                out[(a, b)] = {'dens': np.full(38, np.nan), 'str': np.full(38, np.nan)}
        return out

    tpc_pf  = load_fc_perfield_npz(TPC_NPZ)
    cits_pf = load_fc_perfield_npz(CITS_NPZ)
    n_fields_npz = next(iter(tpc_pf.values()))['dens'].shape[0]
    print(f"  NPZ fields: {n_fields_npz}", flush=True)

    # Build per-field area-pair vectors for COMBINED and NULL_RANDOM
    # using the direct on-disk matrices and matched_df unit_ids

    # Build (16, n_em_fields) matrices: dens_mat and str_mat for COMBINED / NULL_RANDOM
    def compute_perfield_area_vectors(field_matrices_dict, field_unit_ids, em_field_keys,
                                      area_map, unit_to_pt, is_null_list=False):
        """Compute per-field (16,) dens and str vectors for each ordered area pair.

        field_matrices_dict: (sess, scan, field) -> matrix OR list of matrices (for NULL)
        Returns: dens_mat (16, n_em_fields), str_mat (16, n_em_fields)
        """
        n_fk = len(em_field_keys)
        dens_mat = np.full((16, n_fk), np.nan)
        str_mat  = np.full((16, n_fk), np.nan)

        for fi, tkey in enumerate(em_field_keys):
            sess, scan, field = tkey
            mat_entry = field_matrices_dict.get(tkey)
            if mat_entry is None:
                continue
            if is_null_list:
                # Use median across draws
                draw_mats = mat_entry  # list of (p,p) matrices
                if not draw_mats:
                    continue
                # Stack and median
                valid = [m for m in draw_mats if m is not None]
                if not valid:
                    continue
                stacked = np.stack(valid, axis=0)
                mat = np.median(stacked, axis=0)
            else:
                mat = mat_entry

            uid_list = field_unit_ids.get(tkey)
            if uid_list is None or len(uid_list) != mat.shape[0]:
                continue

            p = len(uid_list)
            unit_areas = [area_map.get((sess, scan, int(uid)), None) for uid in uid_list]

            # For each ordered area pair (a, b), collect (i->j) pairs
            pair_fc_vals = {pair: [] for pair in ORDERED_PAIRS}

            for i in range(p):
                ai = unit_areas[i]
                if ai not in AREAS:
                    continue
                for j in range(p):
                    if i == j:
                        continue
                    aj = unit_areas[j]
                    if aj not in AREAS:
                        continue
                    fc_val = float(mat[i, j])
                    pair_fc_vals[(ai, aj)].append(fc_val)

            for pi_ord, (a, b) in enumerate(ORDERED_PAIRS):
                vals = pair_fc_vals[(a, b)]
                if not vals:
                    continue
                vals_arr = np.array(vals)
                # Density: fraction with FC-present (nonzero)
                dens_mat[pi_ord, fi] = float(np.mean(vals_arr != 0))
                # Strength: mean |.| (includes zeros)
                str_mat[pi_ord, fi]  = float(np.mean(np.abs(vals_arr)))

        return dens_mat, str_mat

    print("  Computing per-field area vectors for COMBINED ...", flush=True)
    comb_dens_mat, comb_str_mat = compute_perfield_area_vectors(
        field_combined_mean, field_unit_ids, em_field_keys, area_map, unit_to_pt)

    print("  Computing per-field area vectors for NULL_RANDOM ...", flush=True)
    null_dens_mat, null_str_mat = compute_perfield_area_vectors(
        field_null_random_mats, field_unit_ids, em_field_keys, area_map, unit_to_pt,
        is_null_list=True)

    # For TPC and CITS, use the NPZ matrices (authoritative from the full 38-field analysis)
    # The NPZ dens_mat indexing is (16 pairs, 38 fields). Reuse directly.
    def npz_pf_to_mats(pf_dict):
        n_f = next(iter(pf_dict.values()))['dens'].shape[0]
        d = np.full((16, n_f), np.nan)
        s = np.full((16, n_f), np.nan)
        for pi_ord, (a, b) in enumerate(ORDERED_PAIRS):
            d[pi_ord, :] = pf_dict[(a, b)]['dens']
            s[pi_ord, :] = pf_dict[(a, b)]['str']
        return d, s

    tpc_dens_mat,  tpc_str_mat  = npz_pf_to_mats(tpc_pf)
    cits_dens_mat, cits_str_mat = npz_pf_to_mats(cits_pf)

    print(f"  TPC dens_mat  shape: {tpc_dens_mat.shape}", flush=True)
    print(f"  CITS dens_mat shape: {cits_dens_mat.shape}", flush=True)
    print(f"  COMB dens_mat shape: {comb_dens_mat.shape} (em-field subset)", flush=True)
    print(f"  NULL dens_mat shape: {null_dens_mat.shape} (em-field subset)", flush=True)

    # Point estimates: Spearman r vs EM for each metric
    def spearman_subset(fc_vec, em_v, idx):
        fc_sub = fc_vec[idx]
        em_sub = em_v[idx]
        valid = ~(np.isnan(fc_sub) | np.isnan(em_sub))
        if valid.sum() < 3:
            return np.nan, np.nan
        return spearmanr(fc_sub[valid], em_sub[valid])

    all_idx    = list(range(16))
    within_idx = WITHIN_IDX
    between_idx = BETWEEN_IDX

    methods_mats = {
        'TPC':      (tpc_dens_mat,  tpc_str_mat),
        'CITS':     (cits_dens_mat, cits_str_mat),
        'COMBINED': (comb_dens_mat, comb_str_mat),
        'NULL_RANDOM': (null_dens_mat, null_str_mat),
    }

    point_est = {}
    for method, (dm, sm) in methods_mats.items():
        fc_dens = np.nanmean(dm, axis=1)
        fc_str  = np.nanmean(sm, axis=1)
        r_dens_all, _ = spearman_subset(fc_dens, em_vec, all_idx)
        r_dens_w,   _ = spearman_subset(fc_dens, em_vec, within_idx)
        r_dens_b,   _ = spearman_subset(fc_dens, em_vec, between_idx)
        r_str_all,  _ = spearman_subset(fc_str,  em_vec, all_idx)
        point_est[method] = {
            'dens_all16':    r_dens_all,
            'dens_within4':  r_dens_w,
            'dens_between12': r_dens_b,
            'str_all16':     r_str_all,
        }
        print(f"  {method}: dens_all={r_dens_all:.4f} dens_within={r_dens_w:.4f} "
              f"dens_between={r_dens_b:.4f} str_all={r_str_all:.4f}", flush=True)

    # Field-clustered bootstrap for CI and paired difference vs CITS
    print(f"\n  Running field-clustered bootstrap (N={N_BOOT}) ...", flush=True)

    # TPC/CITS use n_fields_npz (e.g., 100), COMB/NULL use n_em_fields (38)
    # For comparability, run all methods bootstrapping the em_field_keys set
    # (38 fields) with the COMBINED/NULL fields as the natural bootstrap unit.
    # For TPC and CITS we use cits_dens_mat/tpc_dens_mat columns 0..37 (the first 38 fields).
    # NOTE: the NPZ perfield index ordering may differ from em_field_keys ordering.
    # To be safe, we use the direct on-disk computed comb/null and also compute
    # per-field vectors for TPC/CITS using the same on-disk approach.

    # Recompute TPC/CITS per-field vectors using on-disk matrices
    # (to have aligned 38-field indexing with COMBINED/NULL)
    print("  Computing per-field area vectors for TPC from on-disk matrices ...", flush=True)
    tpc_dens_mat38, tpc_str_mat38 = compute_perfield_area_vectors(
        field_tpc_mean, field_unit_ids, em_field_keys, area_map, unit_to_pt)

    print("  Computing per-field area vectors for CITS from on-disk matrices ...", flush=True)
    cits_dens_mat38, cits_str_mat38 = compute_perfield_area_vectors(
        field_cits_mean, field_unit_ids, em_field_keys, area_map, unit_to_pt)

    # Bootstrap function (field-clustered, 38 fields, paired across methods)
    rng = np.random.default_rng(BOOT_SEED)

    def boot_spearman_quad(dm_list, em_v, subset_idx, n_boot, rng):
        """Bootstrap Spearman r for multiple methods simultaneously (paired resampling).

        dm_list: list of (16, n_fields) dens_mat arrays, all same n_fields
        Returns: list of (n_boot,) arrays, one per method.
        """
        n_methods = len(dm_list)
        n_f = dm_list[0].shape[1]
        results = [np.full(n_boot, np.nan) for _ in range(n_methods)]
        em_sub = em_v[subset_idx]
        em_valid = ~np.isnan(em_sub)

        for bi in range(n_boot):
            idx = rng.integers(0, n_f, size=n_f)
            for mi, dm in enumerate(dm_list):
                fc_sub = np.nanmean(dm[subset_idx, :][:, idx], axis=1)
                valid = em_valid & ~np.isnan(fc_sub)
                if valid.sum() < 3:
                    continue
                r, _ = spearmanr(fc_sub[valid], em_sub[valid])
                results[mi][bi] = r
        return results

    def ci95(arr):
        v = arr[~np.isnan(arr)]
        if len(v) < 100:
            return (np.nan, np.nan)
        return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))

    def paired_diff_ci(boot_a, boot_b):
        """Paired diff a - b mean and 95% CI."""
        diff = boot_a - boot_b
        v = diff[~np.isnan(diff)]
        if len(v) < 100:
            return np.nan, (np.nan, np.nan)
        return float(np.mean(v)), (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))

    method_names = ['TPC', 'CITS', 'COMBINED', 'NULL_RANDOM']
    dm_list_38 = [tpc_dens_mat38, cits_dens_mat38, comb_dens_mat, null_dens_mat]
    sm_list_38 = [tpc_str_mat38,  cits_str_mat38,  comb_str_mat,  null_str_mat]

    metrics = {
        'dens_all16':     (all_idx,    dm_list_38),
        'dens_within4':   (within_idx, dm_list_38),
        'dens_between12': (between_idx,dm_list_38),
        'str_all16':      (all_idx,    sm_list_38),
    }

    boot_results = {}
    for metric_name, (m_idx, mats_list) in metrics.items():
        print(f"    Bootstrapping {metric_name} ...", flush=True)
        boot_arr_list = boot_spearman_quad(mats_list, em_vec, m_idx, N_BOOT, rng)
        boot_results[metric_name] = {method_names[mi]: boot_arr_list[mi]
                                     for mi in range(len(method_names))}

    # Build final results dict
    area_results = {}
    for metric_name in metrics:
        for method in method_names:
            boot_arr = boot_results[metric_name][method]
            pt = point_est[method][metric_name]
            ci_val = ci95(boot_arr)
            cits_boot = boot_results[metric_name]['CITS']
            diff_mean, diff_ci_val = paired_diff_ci(boot_arr, cits_boot)
            diff_excl_0 = (diff_ci_val[0] is not None and diff_ci_val[1] is not None and
                           not np.isnan(diff_ci_val[0]) and
                           (diff_ci_val[0] > 0 or diff_ci_val[1] < 0))
            area_results[(metric_name, method)] = {
                'r_point':          pt,
                'ci_lo':            ci_val[0],
                'ci_hi':            ci_val[1],
                'diff_vs_CITS_mean': diff_mean,
                'diff_vs_CITS_ci_lo': diff_ci_val[0],
                'diff_vs_CITS_ci_hi': diff_ci_val[1],
                'diff_excludes_0':  'YES' if diff_excl_0 else 'NO',
            }

    print("\n=== Area-level Spearman r vs EM density ===", flush=True)
    for metric_name in ['dens_all16', 'dens_within4', 'dens_between12', 'str_all16']:
        print(f"\n  {metric_name}:", flush=True)
        for method in method_names:
            r_vals = area_results[(metric_name, method)]
            print(f"    {method:<15} r={r_vals['r_point']:.4f} "
                  f"CI=[{r_vals['ci_lo']:.4f}, {r_vals['ci_hi']:.4f}] "
                  f"diff_vs_CITS={r_vals['diff_vs_CITS_mean']:.4f} "
                  f"CI=[{r_vals['diff_vs_CITS_ci_lo']:.4f}, "
                  f"{r_vals['diff_vs_CITS_ci_hi']:.4f}] "
                  f"excl0={r_vals['diff_excludes_0']}", flush=True)


    # ══════════════════════════════════════════════════════════════════════════════
    # SECTION 6: Load prior COMBINED_CITS_TPCLAG0 for comparison
    # ══════════════════════════════════════════════════════════════════════════════

    print("\n" + "="*70, flush=True)
    print("SECTION 6: Load prior COMBINED_CITS_TPCLAG0 results", flush=True)
    print("="*70, flush=True)

    prior_df = None
    try:
        prior_df = pd.read_csv(PRIOR_COMBINED_CSV)
        print(f"  Loaded: {PRIOR_COMBINED_CSV} ({len(prior_df)} rows)", flush=True)
    except Exception as e:
        print(f"  Could not load prior results: {e}", flush=True)


    # ══════════════════════════════════════════════════════════════════════════════
    # SECTION 7: Write stats CSV
    # ══════════════════════════════════════════════════════════════════════════════

    print("\n" + "="*70, flush=True)
    print("SECTION 7: Write stats CSV", flush=True)
    print("="*70, flush=True)

    rows = []

    # ── Pair-level enrichment rows ─────────────────────────────────────────────────
    for method in ['TPC', 'CITS', 'COMBINED', 'NULL_RANDOM']:
        e = enrichment.get(method, {})
        row = {
            'test':           'pair_level_SC_enrichment',
            'method':         method,
            'N_fc_present':   e.get('N_fc_present'),
            'N_fc_absent':    e.get('N_fc_absent'),
            'N_fcpres_syn':   e.get('N_fcpres_syn'),
            'N_fcabs_syn':    e.get('N_fcabs_syn'),
            'frac_present':   e.get('frac_present'),
            'ci_lo_present':  e.get('ci_lo_present'),
            'ci_hi_present':  e.get('ci_hi_present'),
            'frac_absent':    e.get('frac_absent'),
            'ci_lo_absent':   e.get('ci_lo_absent'),
            'ci_hi_absent':   e.get('ci_hi_absent'),
            'fold_enrichment': e.get('fold_enrichment'),
            'fisher_OR':      e.get('fisher_OR'),
            'fisher_p':       e.get('fisher_p'),
            'n_fields':       e.get('n_fields'),
            'universe':       e.get('universe'),
            'cave_version':   CAVE_VERSION,
            'metric':         '',
            'r_point':        '',
            'ci_lo':          '',
            'ci_hi':          '',
            'diff_vs_CITS_mean':  '',
            'diff_vs_CITS_ci_lo': '',
            'diff_vs_CITS_ci_hi': '',
            'diff_excludes_0':    '',
        }
        rows.append(row)

    # ── Area-level Spearman rows ───────────────────────────────────────────────────
    for metric_name in ['dens_all16', 'dens_within4', 'dens_between12', 'str_all16']:
        for method in method_names:
            r_vals = area_results[(metric_name, method)]
            row = {
                'test':           'area_level_spearman',
                'method':         method,
                'N_fc_present':   '',
                'N_fc_absent':    '',
                'N_fcpres_syn':   '',
                'N_fcabs_syn':    '',
                'frac_present':   '',
                'ci_lo_present':  '',
                'ci_hi_present':  '',
                'frac_absent':    '',
                'ci_lo_absent':   '',
                'ci_hi_absent':   '',
                'fold_enrichment': '',
                'fisher_OR':      '',
                'fisher_p':       '',
                'n_fields':       '',
                'universe':       '',
                'cave_version':   CAVE_VERSION,
                'metric':         metric_name,
                'r_point':        r_vals['r_point'],
                'ci_lo':          r_vals['ci_lo'],
                'ci_hi':          r_vals['ci_hi'],
                'diff_vs_CITS_mean':  r_vals['diff_vs_CITS_mean'],
                'diff_vs_CITS_ci_lo': r_vals['diff_vs_CITS_ci_lo'],
                'diff_vs_CITS_ci_hi': r_vals['diff_vs_CITS_ci_hi'],
                'diff_excludes_0':    r_vals['diff_excludes_0'],
            }
            rows.append(row)

    # ── Prior COMBINED_CITS_TPCLAG0 rows (reference only) ─────────────────────────
    if prior_df is not None:
        prior_combined_rows = prior_df[prior_df['method'] == 'COMBINED'].copy()
        for _, row_prior in prior_combined_rows.iterrows():
            row_ref = row_prior.to_dict()
            row_ref['method'] = 'COMBINED_CITS_TPCLAG0'
            row_ref['test'] = row_prior.get('test', '')
            rows.append(row_ref)

    csv_df = pd.DataFrame(rows)
    os.makedirs(OUTDIR, exist_ok=True)
    csv_df.to_csv(OUT_CSV, index=False)
    print(f"  Stats CSV saved to: {OUT_CSV}", flush=True)


    # ══════════════════════════════════════════════════════════════════════════════
    # SECTION 8: Figure -- Area-level EM scatter, four panels
    #
    # Panels: TPC | CITS | COMBINED_PCCONTEMP | NULL_RANDOM
    # X-axis: EM directed pair density (v1718)
    # Y-axis: FC density (pooled states, mean across fields)
    # Within-area: filled orange circles; Between-area: open blue squares
    # Annotation: Spearman r, 95% CI
    # ══════════════════════════════════════════════════════════════════════════════

    print("\n" + "="*70, flush=True)
    print("SECTION 8: Figure", flush=True)
    print("="*70, flush=True)

    WITHIN_COLOR  = '#E07B39'
    BETWEEN_COLOR = '#4C72B0'
    PAIR_LABELS   = [f'{a}->{b}' for a, b in ORDERED_PAIRS]

    label_offsets = {
        'AL->AL': (5, 5), 'LM->LM': (-38, 6), 'RL->RL': (5, -12), 'V1->V1': (-38, -12),
        'AL->LM': (5, 6), 'AL->RL': (-40, 6), 'AL->V1': (5, -12),
        'LM->AL': (-40, -12), 'LM->RL': (5, 6), 'LM->V1': (-40, 6),
        'RL->AL': (5, 5), 'RL->LM': (-38, -12), 'RL->V1': (5, -12),
        'V1->AL': (-38, 5), 'V1->LM': (5, 6), 'V1->RL': (5, -12),
    }

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=False)

    panel_data = [
        ('TPC',             tpc_dens_mat38,  'TPC FC density vs EM density'),
        ('CITS',            cits_dens_mat38, 'CITS FC density vs EM density'),
        ('COMBINED\n(PC-contemp)', comb_dens_mat, 'COMBINED (CITS+PC-contemp) vs EM density'),
        ('NULL_RANDOM',     null_dens_mat,   'NULL-RANDOM vs EM density'),
    ]

    for ax, (method_label, dm, title) in zip(axes, panel_data):
        fc_dens_mean = np.nanmean(dm, axis=1)
        r_val_str = f"dens_all16"
        method_key = method_label.replace('\n(PC-contemp)', '').replace('\n', '')
        if method_key == 'COMBINED':
            method_key = 'COMBINED'
        r_entry = area_results.get((r_val_str, method_key))
        if r_entry is None and 'NULL' in method_key:
            r_entry = area_results.get((r_val_str, 'NULL_RANDOM'))
        if r_entry is None:
            r_pt, ci_lo, ci_hi = np.nan, np.nan, np.nan
        else:
            r_pt  = r_entry['r_point']
            ci_lo = r_entry['ci_lo']
            ci_hi = r_entry['ci_hi']

        x_vals, y_vals, colors, markers, zorders, alphas, sizes, labels_plot = (
            [], [], [], [], [], [], [], [])
        for i, (a, b) in enumerate(ORDERED_PAIRS):
            pair_label = f'{a}->{b}'
            em_d = em_vec[i]
            fc_d = fc_dens_mean[i]
            if np.isnan(em_d) or np.isnan(fc_d):
                continue
            is_within = (a == b)
            color   = WITHIN_COLOR if is_within else BETWEEN_COLOR
            marker  = 'o' if is_within else 's'
            zo      = 4 if is_within else 3
            alpha   = 0.9 if is_within else 0.65
            ms      = 80 if is_within else 45
            x_vals.append(em_d); y_vals.append(fc_d)
            colors.append(color); markers.append(marker)
            zorders.append(zo); alphas.append(alpha); sizes.append(ms)
            labels_plot.append(pair_label)

        for xi, yi, col, mk, zo, al, ms in zip(
                x_vals, y_vals, colors, markers, zorders, alphas, sizes):
            ax.scatter(xi, yi, c=col, marker=mk, s=ms, zorder=zo,
                       alpha=al, edgecolors='black', linewidths=0.5)

        for xi, yi, lbl in zip(x_vals, y_vals, labels_plot):
            xoff, yoff = label_offsets.get(lbl, (4, 4))
            ax.annotate(lbl, (xi, yi), fontsize=5.2,
                        xytext=(xoff, yoff), textcoords='offset points',
                        color='#333333', zorder=5)

        ci_str = (f'[{ci_lo:.2f}, {ci_hi:.2f}]'
                  if not (np.isnan(ci_lo) or np.isnan(ci_hi)) else '[n/a]')
        r_str = f'{r_pt:.3f}' if not np.isnan(r_pt) else 'n/a'
        ax.text(0.96, 0.05,
                f'Spearman r = {r_str}\n95% CI {ci_str}',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=7.5,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', lw=0.8))
        ax.set_title(title, fontsize=8, pad=4)
        ax.set_xlabel('EM directed synapse-pair density\n(v1718, matched_df only)', fontsize=7.5)
        ax.set_ylabel('FC pair density (pooled states)', fontsize=7.5)
        ax.tick_params(labelsize=7.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=WITHIN_COLOR,
               markeredgecolor='black', markeredgewidth=0.5, markersize=8,
               label='Within-area (diagonal)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=BETWEEN_COLOR,
               markeredgecolor='black', markeredgewidth=0.5, markersize=7,
               label='Between-area'),
    ]
    fig.legend(handles=legend_handles, loc='upper center', ncol=2, fontsize=8,
               framealpha=0.9, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0.01, 1, 0.96])
    fig.savefig(OUT_FIG_PNG, dpi=300, bbox_inches='tight')
    fig.savefig(OUT_FIG_PDF, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved:\n    {OUT_FIG_PNG}\n    {OUT_FIG_PDF}", flush=True)


    # ══════════════════════════════════════════════════════════════════════════════
    # SECTION 9: Final verdict summary
    # ══════════════════════════════════════════════════════════════════════════════

    print("\n" + "="*70, flush=True)
    print("SECTION 9: Final summary and verdict", flush=True)
    print("="*70, flush=True)

    print("\n=== Pair-level SC fold enrichment (summary) ===")
    for method in ['TPC', 'CITS', 'COMBINED', 'NULL_RANDOM']:
        e = enrichment.get(method, {})
        if e:
            print(f"  {method:<15} fold={e['fold_enrichment']:.4f}  "
                  f"SC+|FC-pres={e['N_fcpres_syn']}  p={e['fisher_p']:.2e}")

    print("\n=== Area-level Spearman r vs EM density (str_all16 -- decisive) ===")
    print("  (COMBINED - CITS CI excluding 0 AND > NULL - CITS => lag-0 hypothesis confirmed)")
    for method in ['TPC', 'CITS', 'COMBINED', 'NULL_RANDOM']:
        r_v = area_results.get(('str_all16', method), {})
        if r_v:
            print(f"  {method:<15} r={r_v['r_point']:.4f} "
                  f"CI=[{r_v['ci_lo']:.4f}, {r_v['ci_hi']:.4f}] "
                  f"diff_vs_CITS={r_v['diff_vs_CITS_mean']:.4f} "
                  f"CI=[{r_v['diff_vs_CITS_ci_lo']:.4f}, {r_v['diff_vs_CITS_ci_hi']:.4f}] "
                  f"excl0={r_v['diff_excludes_0']}")

    # Compare with prior COMBINED_CITS_TPCLAG0
    if prior_df is not None:
        prior_str_combined = prior_df[
            (prior_df['method'] == 'COMBINED') &
            (prior_df['test'] == 'area_level_spearman') &
            (prior_df['metric'] == 'str_all16')
        ]
        if not prior_str_combined.empty:
            row_p = prior_str_combined.iloc[0]
            print(f"\n  COMBINED_CITS_TPCLAG0 r={row_p.get('r_point', 'n/a')} "
                  f"(prior run, 2026-05-20 -- included for reference)")

    # Decisive test
    str_comb = area_results.get(('str_all16', 'COMBINED'), {})
    str_null = area_results.get(('str_all16', 'NULL_RANDOM'), {})
    str_cits = area_results.get(('str_all16', 'CITS'), {})
    str_tpc  = area_results.get(('str_all16', 'TPC'), {})

    if str_comb and str_null and str_cits:
        comb_diff = str_comb.get('diff_vs_CITS_mean', np.nan)
        null_diff = str_null.get('diff_vs_CITS_mean', np.nan)
        comb_excl0 = str_comb.get('diff_excludes_0', 'NO')
        null_excl0 = str_null.get('diff_excludes_0', 'NO')

        print("\n=== VERDICT: str_all16 decisive test ===")
        print(f"  COMBINED (CITS+PC-contemp) - CITS: mean={comb_diff:.4f} excl_0={comb_excl0}")
        print(f"  NULL_RANDOM - CITS:               mean={null_diff:.4f} excl_0={null_excl0}")
        if (comb_excl0 == 'YES' and (np.isnan(null_diff) or comb_diff > null_diff)):
            verdict = ("CONFIRMED: adding PC-contemporaneous edges rescues area-strength. "
                       "Lag-0 exclusion is causally responsible for CITS's deficit.")
        elif comb_excl0 == 'YES' and not np.isnan(null_diff) and comb_diff <= null_diff:
            verdict = ("PARTIAL: COMBINED improves but no more than NULL-RANDOM. "
                       "The rescue is due to more edges, not lag-0 specifically.")
        elif comb_excl0 == 'NO':
            verdict = ("NOT CONFIRMED: PC-contemporaneous edges do NOT rescue area-strength. "
                       "Lag-0 exclusion alone does not explain CITS's area-strength deficit.")
        else:
            verdict = "AMBIGUOUS: check CI values manually."
        print(f"\n  {verdict}")

    print(f"\nOutputs:")
    print(f"  Stats CSV:    {OUT_CSV}")
    print(f"  Figure PNG:   {OUT_FIG_PNG}")
    print(f"  Figure PDF:   {OUT_FIG_PDF}")
    print(f"  PC-contemp cache: {CONTEMP_DIR}")
    print("\nDone.", flush=True)
