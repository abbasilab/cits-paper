"""PC-CITS skeleton with Numba-accelerated subset enumeration.

Drop-in replacement for cits_pc_skeleton_optimized.py.  All math, GPU batching,
and partial-correlation logic are kept identical.  Only the hot-path enumeration
and result bookkeeping are replaced.

Bottleneck in the original (Python-side, level l>=1):
  - itertools.combinations(nbrs_sorted, l)  for each edge
  - dict.setdefault(S, []).append(...)      for each (edge, S) pair
  Total ops at l=2, p=850: ~600M Python allocations.

Replacement strategy:
  1. Numba JIT kernel (_enum_combos_into_flat) generates all C(n,l) combos for
     one edge into a pre-allocated int32 buffer in the same lex order as
     itertools.combinations(sorted_input, l).
  2. All per-edge combo buffers are concatenated into one flat (total_nc, l)
     array.  A companion int32 array records which edge index each row belongs
     to.
  3. numpy lexsort (or argsort for l=1) groups identical S values together
     without any Python dict insertion.
  4. For each unique-S group, the GPU residualization is performed (identical
     math to the original).  Results are stored in flat float64/bool arrays.
  5. The first-S-wins removal scan uses the inverse sort to locate each edge's
     combos in the sorted buffer, recovering their lex enumeration order.

Correctness invariants:
  1. Combination lex order: the Numba kernel uses the same increment-and-carry
     algorithm as CPython's itertools.combinations on sorted input.
  2. First-S-wins: for each edge, combos are scanned in lex enumeration order
     (same as combinations(sorted_nbrs, l)).  The first independently-testing S
     removes the edge and is stored in sep_sets.
  3. GPU residualization math is copied verbatim from _apply_results_per_edge
     in cits_pc_skeleton_optimized.  The only difference is that r values are
     stored in a flat array rather than a nested Python dict.
  4. The independence test is the exact same Fisher-z formula as
     _is_cond_indep_pcorr.

Author: MICrONS arousal-paper PC-CITS scaling pass.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np
from scipy import stats as _stats

# ── import original module for components we do NOT replace ──────────────────
import sys as _sys
_sys.path.insert(0, os.path.dirname(__file__))
from cits_pc_skeleton_optimized import (
    _fisher_r_crit,
    _level0_vectorized,
    _is_cond_indep_pcorr,
    _TORCH_OK,
    _NUMBA_OK,
)

from cits import methods as cits_m

try:
    import torch
except Exception:
    torch = None

try:
    from numba import njit as _njit
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False
    def _njit(*a, **kw):  # noqa
        def deco(f):
            return f
        if a and callable(a[0]):
            return a[0]
        return deco


# ─────────────────────────────────────────────────────────────────────────────
# Numba JIT: enumerate all C(n, l) combinations into a flat buffer.
#
# Algorithm: the standard increment-and-carry on a length-l position array c[].
# This is identical to CPython's itertools.combinations internals, so output
# order exactly matches combinations(sorted_input, l).
#
# WHY fastmath=False: we write integer indices, not floats.  No floating-point
# reassociation is involved.
# ─────────────────────────────────────────────────────────────────────────────
if _HAVE_NUMBA:
    @_njit(cache=True, fastmath=False)
    def _enum_combos_into_flat(nbrs, l, out_flat):
        """Fill out_flat with all C(len(nbrs), l) combos in lex order.

        nbrs    : int32/int64 1-D sorted array of neighbor node indices.
        out_flat: int32 1-D array of length C(len(nbrs), l) * l, row-major.
        Returns : number of combos written.
        """
        n = len(nbrs)
        if l == 0 or n < l:
            return np.int64(0)

        c = np.empty(l, dtype=np.int64)
        for i in range(l):
            c[i] = i

        combo_idx = np.int64(0)
        while True:
            base = combo_idx * l
            for i in range(l):
                out_flat[base + i] = nbrs[c[i]]
            combo_idx += 1

            i = l - 1
            while i >= 0 and c[i] == n - l + i:
                i -= 1
            if i < 0:
                break
            c[i] += 1
            for j in range(i + 1, l):
                c[j] = c[j - 1] + 1

        return combo_idx
else:
    def _enum_combos_into_flat(nbrs, l, out_flat):
        n = len(nbrs)
        if l == 0 or n < l:
            return 0
        c = list(range(l))
        combo_idx = 0
        while True:
            base = combo_idx * l
            for i in range(l):
                out_flat[base + i] = nbrs[c[i]]
            combo_idx += 1
            i = l - 1
            while i >= 0 and c[i] == n - l + i:
                i -= 1
            if i < 0:
                break
            c[i] += 1
            for j in range(i + 1, l):
                c[j] = c[j - 1] + 1
        return combo_idx


def _ncombos(n, l):
    """C(n, l) as a Python int, without overflow."""
    if l > n:
        return 0
    if l == 0:
        return 1
    result = 1
    for i in range(l):
        result = result * (n - i) // (i + 1)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Independence test: vectorised Fisher-z (matches _is_cond_indep_pcorr exactly).
# ─────────────────────────────────────────────────────────────────────────────
def _indep_vec(rs, N, k, alpha):
    """Vectorised version of _is_cond_indep_pcorr.

    Returns a bool array of same length as rs.
    Exact same formula as the scalar version in cits_pc_skeleton_optimized.
    """
    df = N - k - 3
    if df <= 0:
        return np.zeros(len(rs), dtype=bool)
    rs_c = np.clip(rs, -0.999999999, 0.999999999)
    # |r|==1 -> dependent
    at_one = np.abs(rs) >= 1.0
    z = 0.5 * np.log((1.0 + rs_c) / (1.0 - rs_c))
    T = np.sqrt(float(df)) * np.abs(z)
    z_crit = _stats.norm.ppf(1.0 - alpha / 2.0)
    indep = T < z_crit
    indep[at_one] = False
    return indep


# ─────────────────────────────────────────────────────────────────────────────
# Core: array-based level-l test.
#
# Replaces both _level_l_batched and _apply_results_per_edge from the original
# module.  GPU math is copied verbatim from _apply_results_per_edge; only the
# bookkeeping structures (Python dicts per combo) are eliminated.
# ─────────────────────────────────────────────────────────────────────────────
def _level_l_batched_numba(A, chi, alpha, t_target, p, tau, n_nodes, sep_sets,
                           l, edges_to_test, use_gpu, device, verbose):
    """Level-l PC test: Numba combo enumeration + array-based result bookkeeping.

    Produces identical A and sep_sets to cits_pc_skeleton_optimized._level_l_batched.
    """
    A_frozen = A.copy()
    A_or = (A_frozen != 0) | (A_frozen.T != 0)

    # ── Phase 1: enumerate combos for each surviving edge ────────────────────
    edge_keys  = []   # (j, i) Python tuples, one per surviving edge
    edge_Ai_np = []   # node index i (int)
    edge_Bi_np = []   # node index j (int)
    edge_nc    = []   # C(|nbrs|, l) per edge
    combo_chunks = [] # list of (nc, l) int32 arrays

    for (v, v1, t1) in edges_to_test:
        i = t_target * p + v
        j = t1 * p + v1
        if A[j, i] == 0:
            continue
        nbr_mask = A_or[i] | A_or[j]
        nbr_mask[i] = False
        nbr_mask[j] = False
        nbrs_arr = np.flatnonzero(nbr_mask).astype(np.int32)
        n_nbrs = nbrs_arr.size
        if n_nbrs < l:
            continue

        nc = _ncombos(n_nbrs, l)
        out_flat = np.empty(nc * l, dtype=np.int32)
        _enum_combos_into_flat(nbrs_arr, l, out_flat)
        combo_chunks.append(out_flat.reshape(nc, l))

        edge_keys.append((j, i))
        edge_Ai_np.append(i)
        edge_Bi_np.append(j)
        edge_nc.append(nc)

    if not edge_keys:
        return False

    n_edges  = len(edge_keys)
    total_nc = sum(edge_nc)

    if verbose:
        print(f"  [numba-l{l}] edges={n_edges} total_combos={total_nc}", flush=True)

    # ── Phase 2: build flat arrays ────────────────────────────────────────────
    # combo_all: (total_nc, l) int32 — all combos in edge-enumeration order
    # eid_all:   (total_nc,)  int32 — which edge each row belongs to
    combo_all = np.concatenate(combo_chunks, axis=0)
    eid_all   = np.empty(total_nc, dtype=np.int32)
    # nc_offsets[ei] = start position of edge ei's combos in combo_all
    nc_offsets = np.empty(n_edges + 1, dtype=np.int64)
    nc_offsets[0] = 0
    ptr = 0
    for ei, nc_ei in enumerate(edge_nc):
        eid_all[ptr: ptr + nc_ei] = ei
        nc_offsets[ei + 1] = nc_offsets[ei] + nc_ei
        ptr += nc_ei

    edge_Ai_arr = np.array(edge_Ai_np, dtype=np.int32)
    edge_Bi_arr = np.array(edge_Bi_np, dtype=np.int32)

    # ── Phase 3: sort by S to group identical S values ────────────────────────
    # After sorting, position `pos` in the sorted buffer corresponds to combo
    # `order[pos]` in the original (edge-enumeration) buffer.
    if l == 1:
        order = np.argsort(combo_all[:, 0], kind='stable')
    else:
        order = np.lexsort(combo_all[:, ::-1].T)

    sorted_combos = combo_all[order]   # (total_nc, l)
    sorted_eid    = eid_all[order]     # edge index for each sorted position

    # Group boundaries in sorted buffer
    if total_nc == 0:
        return False
    if l == 1:
        change = np.empty(total_nc, dtype=bool)
        change[0] = True
        change[1:] = sorted_combos[1:, 0] != sorted_combos[:-1, 0]
    else:
        change = np.empty(total_nc, dtype=bool)
        change[0] = True
        change[1:] = (np.diff(sorted_combos, axis=0) != 0).any(axis=1)
    group_starts_arr = np.where(change)[0]
    n_groups = len(group_starts_arr)

    if verbose:
        print(f"  [numba-l{l}] unique_S={n_groups} total_tests={total_nc}", flush=True)

    # ── Phase 4: GPU/numpy batched residualization per unique S ───────────────
    # Results stored in flat arrays indexed by sorted-buffer position.
    # rs_flat[pos]    = partial correlation r for sorted position pos
    # indep_flat[pos] = True if conditionally independent
    N = chi.shape[1]
    rs_flat    = np.empty(total_nc, dtype=np.float64)
    indep_flat = np.zeros(total_nc, dtype=bool)

    chi_dev = None
    if use_gpu:
        chi_dev = torch.from_numpy(chi).to(device=device, dtype=torch.float64)

    for gi in range(n_groups):
        gs  = int(group_starts_arr[gi])
        ge  = int(group_starts_arr[gi + 1]) if gi + 1 < n_groups else total_nc
        S_arr = sorted_combos[gs]   # int32 array of length l (the S for this group)

        k  = l
        df = N - k - 3
        if df <= 0:
            rs_flat[gs:ge]    = np.nan
            indep_flat[gs:ge] = False
            continue

        Ai_grp = edge_Ai_arr[sorted_eid[gs:ge]]
        Bi_grp = edge_Bi_arr[sorted_eid[gs:ge]]

        if use_gpu:
            S_idx = torch.from_numpy(S_arr.astype(np.int64)).to(device)
            ones  = torch.ones((N, 1), dtype=torch.float64, device=device)
            Z     = torch.cat([ones, chi_dev[S_idx].T], dim=1)
            ZtZ   = Z.T @ Z
            try:
                L = torch.linalg.cholesky(ZtZ)
                use_chol = True
            except Exception:
                use_chol = False
                ZtZ_inv = torch.linalg.pinv(ZtZ)
            all_idx_set = set(Ai_grp.tolist()) | set(Bi_grp.tolist())
            all_idx = np.array(sorted(all_idx_set), dtype=np.int64)
            idx_t   = torch.from_numpy(all_idx).to(device)
            Y       = chi_dev[idx_t]
            ZtY     = Z.T @ Y.T
            if use_chol:
                Beta = torch.cholesky_solve(ZtY, L)
            else:
                Beta = ZtZ_inv @ ZtY
            res      = Y - (Z @ Beta).T
            res_c    = res - res.mean(dim=1, keepdim=True)
            res_n    = res_c.norm(dim=1, keepdim=True).clamp_min(1e-30)
            res_unit = res_c / res_n
            pos_map  = {int(ix): pi for pi, ix in enumerate(all_idx)}
            A_pos    = torch.tensor([pos_map[a] for a in Ai_grp.tolist()],
                                    dtype=torch.long, device=device)
            B_pos    = torch.tensor([pos_map[b] for b in Bi_grp.tolist()],
                                    dtype=torch.long, device=device)
            rs_grp   = (res_unit[A_pos] * res_unit[B_pos]).sum(dim=1).cpu().numpy()
        else:
            S_idx = S_arr.astype(np.int64)
            Z     = np.column_stack([np.ones(N), chi[S_idx].T])
            ZtZ   = Z.T @ Z
            try:
                L = np.linalg.cholesky(ZtZ)
                use_chol = True
            except np.linalg.LinAlgError:
                use_chol = False
                ZtZ_inv = np.linalg.pinv(ZtZ)
            all_idx_set = set(Ai_grp.tolist()) | set(Bi_grp.tolist())
            all_idx = np.array(sorted(all_idx_set), dtype=np.int64)
            Y       = chi[all_idx]
            ZtY     = Z.T @ Y.T
            if use_chol:
                tmp  = np.linalg.solve(L, ZtY)
                Beta = np.linalg.solve(L.T, tmp)
            else:
                Beta = ZtZ_inv @ ZtY
            res      = Y - (Z @ Beta).T
            res_c    = res - res.mean(axis=1, keepdims=True)
            res_n    = np.linalg.norm(res_c, axis=1, keepdims=True)
            res_n    = np.maximum(res_n, 1e-30)
            res_unit = res_c / res_n
            pos_map  = {int(ix): pi for pi, ix in enumerate(all_idx)}
            A_pos    = np.array([pos_map[a] for a in Ai_grp.tolist()], dtype=np.int64)
            B_pos    = np.array([pos_map[b] for b in Bi_grp.tolist()], dtype=np.int64)
            rs_grp   = (res_unit[A_pos] * res_unit[B_pos]).sum(axis=1)

        rs_flat[gs:ge]    = rs_grp
        indep_flat[gs:ge] = _indep_vec(rs_grp, N, k, alpha)

    # ── Phase 5: first-S-wins removal scan ────────────────────────────────────
    # For each edge, we need to scan its C(|nbrs|, l) S candidates in lex order
    # (original enumeration order) and apply the first conditionally-independent S.
    #
    # inverse_order[orig_pos] = sorted_pos: the position in the sorted buffer
    # that corresponds to original (edge-enumeration-order) position orig_pos.
    # This lets us look up indep_flat for a combo given its original index.
    #
    # WHY inverse sort: the sorted buffer positions give us group membership (for
    # residualization batching), but we need to scan combos in their original lex
    # order per edge (nc_offsets[ei] .. nc_offsets[ei+1]) for first-S-wins.
    inverse_order = np.empty(total_nc, dtype=np.int64)
    inverse_order[order] = np.arange(total_nc, dtype=np.int64)

    any_removed = False
    for ei in range(n_edges):
        ek = edge_keys[ei]
        (j, i) = ek
        if A[j, i] == 0:
            continue
        orig_start = int(nc_offsets[ei])
        orig_end   = int(nc_offsets[ei + 1])
        for orig_pos in range(orig_start, orig_end):
            spos = int(inverse_order[orig_pos])
            if indep_flat[spos]:
                A[j, i] = 0
                sep_sets[(j, i)] = tuple(combo_all[orig_pos].tolist())
                any_removed = True
                break

    return any_removed


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point (same signature as cits_pc_skeleton_optimized)
# ─────────────────────────────────────────────────────────────────────────────
def cits_pc_skeleton_opt(X, tau, alpha=0.05,
                         cond_dep='cond_dep_pcorr',
                         max_cond_size=None,
                         n_workers=16,
                         use_gpu=True,
                         verbose=True):
    """Numba-enumeration variant of the optimized PC-CITS skeleton.

    Identical API and semantics to cits_pc_skeleton_optimized.cits_pc_skeleton_opt.
    """
    if cond_dep == 'cond_dep_hsic':
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_serial_proto", "/tmp/cits_pc_skeleton_prototype.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.cits_pc_skeleton(X, tau, alpha, cond_dep,
                                    max_cond_size=max_cond_size,
                                    verbose=verbose)

    p = X.shape[0]
    n_nodes = p * 2 * (tau + 1)
    t_target = 2 * tau + 1

    A = np.zeros((n_nodes, n_nodes), dtype=int)
    for v in range(p):
        for v1 in range(p):
            for t1 in range(tau + 1, 2 * tau + 1):
                A[t1 * p + v1, t_target * p + v] = 1

    chi = cits_m.data_transform(X, tau).astype(np.float64, copy=False)
    sep_sets = {}

    actual_use_gpu = bool(use_gpu and _TORCH_OK and torch is not None
                          and torch.cuda.is_available())
    device = torch.device('cuda:0') if actual_use_gpu else None

    if verbose:
        print(f"  cits_pc_skeleton_numba: p={p}, tau={tau}, alpha={alpha}, "
              f"N={chi.shape[1]}, GPU={actual_use_gpu}, NUMBA={_HAVE_NUMBA}",
              flush=True)

    edges_to_test = [(v, v1, t1)
                     for v in range(p)
                     for v1 in range(p)
                     for t1 in range(tau + 1, 2 * tau + 1)]

    level_times = {}

    import time as _time
    t0 = _time.time()
    _level0_vectorized(A, chi, alpha, t_target, tau, p, n_nodes, sep_sets, verbose)
    level_times[0] = _time.time() - t0
    if verbose:
        print(f"  l=0 done: edges remaining {(A != 0).sum()}, time {level_times[0]:.2f}s",
              flush=True)

    if max_cond_size is not None and max_cond_size < 1:
        cits_pc_skeleton_opt.last_level_times = level_times
        return A, sep_sets

    l = 1
    while True:
        t0 = _time.time()
        any_removed = _level_l_batched_numba(
            A, chi, alpha, t_target, p, tau, n_nodes,
            sep_sets, l, edges_to_test,
            actual_use_gpu, device, verbose)
        level_times[l] = _time.time() - t0
        if verbose:
            print(f"  l={l} done: edges remaining {(A != 0).sum()}, "
                  f"removed={any_removed}, time {level_times[l]:.2f}s", flush=True)
        if not any_removed:
            break
        l += 1
        if max_cond_size is not None and l > max_cond_size:
            break

    cits_pc_skeleton_opt.last_level_times = level_times
    return A, sep_sets


def cits_pc_full_weighted_opt(X, tau, alpha=0.05, cond_dep='cond_dep_pcorr',
                              max_cond_size=None, n_workers=16, use_gpu=True,
                              verbose=False, thresh=10):
    """Full pipeline: Numba-enumeration skeleton + rolling + weighted (LSCM)."""
    import networkx as nx
    p = X.shape[0]
    A, _ = cits_pc_skeleton_opt(X, tau, alpha, cond_dep,
                                max_cond_size=max_cond_size,
                                n_workers=n_workers, use_gpu=use_gpu,
                                verbose=verbose)
    B = cits_m.cits_rolled(A, p, tau)
    U = np.zeros((p * (tau + 1), p * (tau + 1)))
    t = 2 * tau + 1
    for v1 in range(p):
        for v2 in range(p):
            for t1 in range(t - tau, t):
                if A[t1 * p + v1, t * p + v2] != 0:
                    U[(tau + 1) * v1 + (t1 - t + tau),
                      (tau + 1) * v2 + tau] = 1
    g = nx.from_numpy_array(U, create_using=nx.DiGraph())
    data_trans = cits_m.data_transformed(X, tau)
    causaleff_A = cits_m.causaleff_lscm(g, data_trans)
    causaleff_B = cits_m.cits_weighted_rolled(causaleff_A, p, tau)
    if np.max(np.abs(causaleff_B)) > 0:
        causaleff_B[np.abs(causaleff_B) < np.max(np.abs(causaleff_B)) / thresh] = 0
    B_out = (causaleff_B != 0).astype(int)
    return B_out, causaleff_B
