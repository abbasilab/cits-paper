"""
gpu_cits_lag_rcit.py

GPU-accelerated CITS-lag using RCIT. Drop-in replacement for
:func:`cits_lag_rcit.cits_lag_rcit` whose CI tests run on GPU.

Architecture
------------
Same chi-stacked windowing as the CPU reference
(:func:`cits_optimal.data_transform_optimal`):

  - Chi width ``w = 2*tau + 1``.
  - Target slice ``t_target = 2*tau`` (rightmost).
  - Source slices ``t1 in {tau, ..., 2*tau-1}``.
  - For each candidate lag edge ``X_{v1}(t1) -> X_{v2}(t_target)`` we test
    CI under increasingly large conditioning subsets drawn from the
    remaining chi variables until either a separator is found (drop edge)
    or ``|S|`` exceeds ``max_cond_size`` (keep edge).

The CI tests reuse the GPU batched gamma-null RCIT kernel from
:mod:`gpu_pc_skeleton_rcit` (``_batched_rcit_pvalues``,
``_precompute_rff_cache``), residualizing in RFF space with ridge
regularization.

Submission strategy:

  - For each ``|S| = l`` we batch the work *across edges*, i.e. for every
    surviving edge ``(i_idx, j_idx)`` we enumerate up to ``batch`` of its
    candidate ``S`` tuples, run them in a single GPU tile, then perform
    the first-removal walk per edge on the cached p-values.
  - When some edges still have surviving conditioning candidates we
    re-batch on the next pass without re-running CPU work for already
    removed edges, mirroring the depth-l loop in
    :mod:`gpu_pc_skeleton_rcit`.

For ``device='cpu'`` we fall through to the CPU reference
:func:`cits_lag_rcit.cits_lag_rcit`.
"""
from __future__ import annotations

import itertools
import os
import sys
from typing import Optional, Union

import numpy as np
import torch

_ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, _ANALYSIS_DIR)

from cits_optimal import data_transform_optimal
from gpu_pc_skeleton_rcit import (
    _precompute_rff_cache,
    _batched_rcit_pvalues,
)

_DEFAULT_BATCH = 4096


def _cache_key(i: int, j: int, S) -> tuple:
    a, b = (i, j) if i < j else (j, i)
    return (a, b, tuple(sorted(S)))


def _flush_workitems_to_gpu(workitems, PHI, pval_cache, K, device):
    """Submit a list of workitems (all with the same |S|) as a single GPU tile.

    Workitem is ``(i, j, S_sorted_tuple)``.
    """
    if not workitems:
        return
    l_local = len(workitems[0][2])
    i_idx = torch.tensor([w[0] for w in workitems], device=device,
                         dtype=torch.long)
    j_idx = torch.tensor([w[1] for w in workitems], device=device,
                         dtype=torch.long)
    if l_local == 0:
        S_idx = None
    else:
        S_arr = np.empty((len(workitems), l_local), dtype=np.int64)
        for k, w in enumerate(workitems):
            S_arr[k, :] = w[2]
        S_idx = torch.as_tensor(S_arr, device=device, dtype=torch.long)
    pvals_t = _batched_rcit_pvalues(PHI, i_idx, j_idx, S_idx, K=K)
    pvals_h = pvals_t.detach().to('cpu').numpy()
    for k, w in enumerate(workitems):
        pval_cache[_cache_key(w[0], w[1], w[2])] = float(pvals_h[k])


def _ensure_pval(i: int, j: int, S, PHI, pval_cache, K, device) -> float:
    """Return cached p-value for ``(i, j, S)`` or compute one on-demand."""
    key = _cache_key(i, j, S)
    if key in pval_cache:
        return pval_cache[key]
    _flush_workitems_to_gpu([(i, j, tuple(sorted(S)))], PHI, pval_cache, K, device)
    return pval_cache[key]


def gpu_cits_lag_rcit(X: np.ndarray,
                      alpha: float = 0.05,
                      tau: int = 1,
                      K: int = 25,
                      n_perm: int = 100,
                      max_cond_size: int = 5,
                      seed: int = 0,
                      device: Union[str, torch.device] = 'cuda:0',
                      null: str = 'gamma',
                      batch: int = _DEFAULT_BATCH,
                      verbose: bool = False,
                      dtype: torch.dtype = torch.float32) -> np.ndarray:
    """GPU-accelerated CITS-lag with RCIT.

    Same semantics as :func:`cits_lag_rcit.cits_lag_rcit` but the RCIT CI
    tests run on GPU using the gamma moment-matched null. Chi-stacking
    convention is preserved exactly.

    Parameters
    ----------
    X : numpy.ndarray, shape ``(p, T)``
        Raw multivariate time series.
    alpha : float
        Significance level for the per-test independence call.
    tau : int
        Markovian order. Chi window has width ``2*tau + 1``.
    K : int
        Number of RFF features per variable.
    n_perm : int
        Permutation count. Used only when ``null='perm'``; for the
        GPU hot path (``null='gamma'``) the parameter is accepted for API
        parity and ignored.
    max_cond_size : int
        Maximum conditioning-set size (RCIT cost grows quickly).
    seed : int
        Master RNG seed (used for the RFF projection cache and bandwidth
        subsampling).
    device : str or torch.device
        Compute device. ``'cpu'`` triggers a fall-through to
        :func:`cits_lag_rcit.cits_lag_rcit`.
    null : {'gamma', 'perm'}
        Null distribution. Only ``'gamma'`` is GPU-accelerated; ``'perm'``
        falls back to the CPU reference.
    batch : int
        Max workitems per GPU kernel call.
    verbose : bool
        If True, print progress.
    dtype : torch.dtype, default torch.float32
        Internal floating-point dtype. ``float32`` is the Phase 4.6 default
        (~2x speedup over float64); pass ``torch.float64`` for the pre-4.6
        numerics.

    Returns
    -------
    B : numpy.ndarray, shape ``(p, p)``, int
        Rolled lag adjacency. ``B[i, j] = 1`` if any tested
        ``i(t1) -> j(t_target)`` edge survived all RCIT tests.
    """
    dev = torch.device(device) if not isinstance(device, torch.device) else device

    # ---- CPU fall-through ----
    if dev.type == 'cpu' or null in ('perm', 'permutation'):
        from cits_lag_rcit import cits_lag_rcit as _cpu_lag
        return _cpu_lag(
            X, alpha=alpha, tau=tau, K=K, n_perm=n_perm,
            max_cond_size=max_cond_size, seed=seed, verbose=verbose,
        )

    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be (p, T); got {X.shape}")
    p, T = X.shape

    # Build chi: (p*(2*tau+1), N) where N = T // (2*tau+1)
    chi = data_transform_optimal(X, tau)  # (n_feat, N)
    n_feat, N_obs = chi.shape
    w = 2 * tau + 1
    if n_feat != p * w:
        raise RuntimeError(f"chi has unexpected shape: {chi.shape}")

    # Switch to (N, n_feat) for RCIT (observations as rows). Move to GPU.
    C_gpu = torch.as_tensor(chi.T, device=dev, dtype=dtype).contiguous()

    if verbose:
        print(f"[gpu_cits_lag_rcit] chi shape={chi.shape}; n_feat={n_feat}, "
              f"N={N_obs}; device={dev}", flush=True)

    # Precompute per-column RFF embeddings (already centered).
    PHI = _precompute_rff_cache(C_gpu, K=K, seed=seed, device=dev)

    # CPU-side p-value cache. Key: (min(i,j), max(i,j), sorted_S).
    pval_cache: dict = {}

    t_target = 2 * tau
    t_source_range = list(range(tau, 2 * tau))

    # ---- Enumerate candidate edges in chi space ----
    # candidate = (i_chi, j_chi, v1, v2)
    candidates = []
    for v2 in range(p):
        for v1 in range(p):
            for t1 in t_source_range:
                i_idx = t1 * p + v1
                j_idx = t_target * p + v2
                if i_idx == j_idx:
                    continue
                candidates.append((i_idx, j_idx, v1, v2))

    if not candidates:
        return np.zeros((p, p), dtype=int)

    # ---- Depth 0 (marginal) — batch all candidate (i,j) once ----
    pending = []
    seen_keys = set()
    for (i_chi, j_chi, _v1, _v2) in candidates:
        key = _cache_key(i_chi, j_chi, ())
        if key in seen_keys:
            continue
        seen_keys.add(key)
        pending.append((i_chi, j_chi, ()))
        if len(pending) >= batch:
            _flush_workitems_to_gpu(pending, PHI, pval_cache, K, dev)
            pending = []
    if pending:
        _flush_workitems_to_gpu(pending, PHI, pval_cache, K, dev)

    # ---- Determine which candidate edges survive the marginal test ----
    # `alive[k]` is True until we drop edge k.
    # `separated[k]` is True once we've found a separating set.
    n_cand = len(candidates)
    alive = np.zeros(n_cand, dtype=bool)
    separated = np.zeros(n_cand, dtype=bool)
    for k, (i_chi, j_chi, _v1, _v2) in enumerate(candidates):
        pval = pval_cache[_cache_key(i_chi, j_chi, ())]
        if pval > alpha:
            # marginally independent => no edge; treat as separated (sepset = empty)
            separated[k] = True
        else:
            alive[k] = True

    # ---- Stagewise: for each l in 1..max_cond_size, batch enumerate ----
    for l in range(1, max_cond_size + 1):
        if not alive.any():
            break
        if l > (n_feat - 2):
            break

        # Stage 1: pre-fill GPU cache with every uncached (i, j, S) workitem at depth l
        # across all alive edges.
        pending = []
        seen_keys = set()
        for k in np.flatnonzero(alive):
            i_chi, j_chi, _v1, _v2 = candidates[k]
            cand = [m for m in range(n_feat) if m != i_chi and m != j_chi]
            if len(cand) < l:
                continue
            for S_list in itertools.combinations(cand, l):
                key = _cache_key(i_chi, j_chi, S_list)
                if key in pval_cache or key in seen_keys:
                    continue
                seen_keys.add(key)
                pending.append((i_chi, j_chi, tuple(sorted(S_list))))
                if len(pending) >= batch:
                    _flush_workitems_to_gpu(pending, PHI, pval_cache, K, dev)
                    pending = []
        if pending:
            _flush_workitems_to_gpu(pending, PHI, pval_cache, K, dev)

        # Stage 2: walk alive edges in candidate-order, search for any S that
        # makes pval > alpha. First found wins (matches CPU semantics).
        for k in np.flatnonzero(alive):
            i_chi, j_chi, _v1, _v2 = candidates[k]
            cand = [m for m in range(n_feat) if m != i_chi and m != j_chi]
            if len(cand) < l:
                # cannot enlarge any further => treat as not separated; kill loop
                continue
            found_separator = False
            for S_list in itertools.combinations(cand, l):
                pval = _ensure_pval(i_chi, j_chi, S_list, PHI, pval_cache, K, dev)
                if pval > alpha:
                    separated[k] = True
                    alive[k] = False
                    found_separator = True
                    break
            # If no separator at this depth, keep alive[k] = True so we try l+1.

        if verbose:
            n_alive = int(alive.sum())
            n_sep = int(separated.sum())
            print(f"[gpu_cits_lag_rcit] depth l={l}: alive={n_alive}, "
                  f"separated_so_far={n_sep}", flush=True)

    # ---- Build rolled adjacency ----
    B = np.zeros((p, p), dtype=int)
    for k, (_i_chi, _j_chi, v1, v2) in enumerate(candidates):
        if not separated[k]:
            B[v1, v2] = 1

    if verbose:
        n_edges = int(B.sum())
        print(f"[gpu_cits_lag_rcit] retained lag edges = {n_edges}", flush=True)

    # Free RFF cache.
    del PHI
    if dev.type == 'cuda':
        torch.cuda.empty_cache()

    return B


# ----------------------------------------------------------------------
# Phase 4.6 batched API
# ----------------------------------------------------------------------


def gpu_cits_lag_rcit_batched(X_list,
                                alpha: float = 0.05,
                                tau: int = 1,
                                K: int = 25,
                                n_perm: int = 100,
                                max_cond_size: int = 5,
                                seed: int = 0,
                                device: Union[str, torch.device] = 'cuda:0',
                                null: str = 'gamma',
                                batch: int = _DEFAULT_BATCH,
                                verbose: bool = False,
                                dtype: torch.dtype = torch.float32,
                                seeds=None):
    """Sequential wrapper that loops :func:`gpu_cits_lag_rcit` over X_list.

    Phase 4.6: amortizes import / GPU init cost across multiple simulations.
    """
    n = len(X_list)
    if seeds is None:
        seeds_eff = [int(seed) + k for k in range(n)]
    else:
        seeds_eff = [int(s) for s in seeds]
        if len(seeds_eff) != n:
            raise ValueError("`seeds` length must equal len(X_list)")
    out = []
    for k, X in enumerate(X_list):
        B = gpu_cits_lag_rcit(X, alpha=alpha, tau=tau, K=K, n_perm=n_perm,
                                max_cond_size=max_cond_size, seed=seeds_eff[k],
                                device=device, null=null, batch=batch,
                                verbose=verbose, dtype=dtype)
        out.append(B)
    return out


# ============================================================================
#  Validation
# ============================================================================


def _jaccard(A1: np.ndarray, A2: np.ndarray) -> float:
    """Jaccard similarity over all directed entries (including the diagonal).

    The lag adjacency is intrinsically directed (lag-tau source -> target) and
    CITS allows self-edges (X_v at t1 vs X_v at t_target sit in different chi
    time-slices), so the diagonal is meaningful and is included in the
    comparison.
    """
    e1 = A1.astype(bool).ravel()
    e2 = A2.astype(bool).ravel()
    inter = int((e1 & e2).sum())
    union = int((e1 | e2).sum())
    if union == 0:
        return 1.0
    return inter / union


def _run_adjacency_match(device: str = 'cuda:0', n_seeds: int = 5,
                          T: int = 1000) -> bool:
    """CPU vs GPU adjacency match at tau=1 across 5 paradigms."""
    try:
        from cits_lag_rcit import cits_lag_rcit as cpu_lag
        from simulation_benchmark_fc_methods_v3 import simulate_extended
    except ImportError as exc:
        print(f"[FATAL] could not import deps: {exc}")
        return False

    paradigms = [
        'lingauss1+mixed',
        'lingauss2+mixed',
        'nonlinnongauss1+mixed',
        'ctrnn',
        'ctrnn+mixed',
    ]

    print("=== gpu_cits_lag_rcit adjacency match ===\n")
    print(f"Device: {device}    Seeds per paradigm: {n_seeds}    T: {T}\n")

    all_pass = True
    summary = {}
    for paradigm in paradigms:
        jaccs = []
        for s in range(n_seeds):
            try:
                (X_pt, *_rest) = simulate_extended(paradigm, noise=0.5,
                                                    T=T, seed=s)
            except Exception as e:
                print(f"  [{paradigm} seed {s}] sim failed: {e}")
                continue
            X = X_pt

            B_cpu = cpu_lag(X, alpha=0.05, tau=1, K=25, n_perm=100,
                            max_cond_size=5, seed=s)
            B_gpu = gpu_cits_lag_rcit(X, alpha=0.05, tau=1, K=25, n_perm=100,
                                       max_cond_size=5, seed=s,
                                       device=device, null='gamma')
            j = _jaccard(B_cpu, B_gpu)
            jaccs.append(j)
            print(f"  [{paradigm} seed {s}] CPU edges={int(B_cpu.sum())}, "
                  f"GPU edges={int(B_gpu.sum())}, Jaccard={j:.3f}")
        mean_j = float(np.mean(jaccs)) if jaccs else 0.0
        ok = mean_j >= 0.85
        all_pass = all_pass and ok
        summary[paradigm] = (mean_j, ok)
        print(f"  [{paradigm}] mean Jaccard = {mean_j:.3f}  "
              f"({'PASS' if ok else 'FAIL'})\n")

    print("--- Summary ---")
    print(f"{'Paradigm':<30} {'Mean Jaccard':>14} {'Pass':>6}")
    print('-' * 54)
    for paradigm in paradigms:
        mean_j, ok = summary.get(paradigm, (0.0, False))
        print(f"{paradigm:<30} {mean_j:>14.3f} {('YES' if ok else 'NO'):>6}")
    print()
    return all_pass


def _speed_benchmark(device: str = 'cuda:0', T: int = 1000) -> None:
    import time
    try:
        from cits_lag_rcit import cits_lag_rcit as cpu_lag
        from simulation_benchmark_fc_methods_v3 import simulate_extended
    except ImportError as exc:
        print(f"[FATAL] could not import deps: {exc}")
        return

    cases = [
        ('lingauss1+mixed', 1),
        ('ctrnn', 1),
        ('lingauss1+mixed', 2),
        ('ctrnn', 2),
    ]

    print(f"\n=== Speed benchmark (T={T}) ===\n")
    print(f"Device: {device}\n")
    print(f"{'Paradigm':<25} {'tau':>4} {'CPU sec':>10} {'GPU sec':>10} "
          f"{'Speedup':>10}")
    print('-' * 64)
    for paradigm, tau in cases:
        try:
            (X_pt, *_rest) = simulate_extended(paradigm, noise=0.5,
                                                T=T, seed=0)
        except Exception as exc:
            print(f"{paradigm:<25} {tau:>4}  sim failed: {exc}")
            continue
        X = X_pt

        # CPU
        t0 = time.perf_counter()
        try:
            _ = cpu_lag(X, alpha=0.05, tau=tau, K=25, n_perm=100,
                        max_cond_size=5, seed=0)
            t_cpu = time.perf_counter() - t0
        except Exception as exc:
            print(f"{paradigm:<25} {tau:>4}  CPU failed: {exc}")
            continue

        # GPU warm-up
        try:
            _ = gpu_cits_lag_rcit(X, alpha=0.05, tau=tau, K=25,
                                   max_cond_size=5, seed=0,
                                   device=device, null='gamma')
            if str(device).startswith('cuda'):
                torch.cuda.synchronize()
        except Exception as exc:
            print(f"{paradigm:<25} {tau:>4}  GPU warm-up failed: {exc}")
            continue

        t0 = time.perf_counter()
        try:
            _ = gpu_cits_lag_rcit(X, alpha=0.05, tau=tau, K=25,
                                   max_cond_size=5, seed=0,
                                   device=device, null='gamma')
            if str(device).startswith('cuda'):
                torch.cuda.synchronize()
            t_gpu = time.perf_counter() - t0
        except Exception as exc:
            print(f"{paradigm:<25} {tau:>4}  GPU failed: {exc}")
            continue

        speedup = t_cpu / max(t_gpu, 1e-9)
        print(f"{paradigm:<25} {tau:>4} {t_cpu:>10.2f} {t_gpu:>10.2f} "
              f"{speedup:>9.1f}x")


def _end_to_end_before_after(device: str = 'cuda:0', T: int = 1000) -> None:
    """End-to-end benchmark of the full GPU driver, BEFORE vs AFTER lag-side port."""
    import time
    try:
        from simulation_benchmark_fc_methods_v3 import simulate_extended
        import cits_plus_v_opt_rcit_nosp_gpu as driver_mod
        from cits_lag_rcit import cits_lag_rcit as cpu_lag
    except ImportError as exc:
        print(f"[FATAL] could not import deps: {exc}")
        return

    paradigms = ['ctrnn', 'ctrnn+mixed']
    print("\n=== End-to-end driver: BEFORE vs AFTER (tau=2) ===\n")
    print(f"Device: {device}    T: {T}\n")
    print(f"{'Paradigm':<20} {'seed':>4} {'BEFORE (CPU lag) sec':>22} "
          f"{'AFTER (GPU lag) sec':>22} {'Speedup':>10}")
    print('-' * 84)

    for paradigm in paradigms:
        for s in range(3):
            try:
                (X_pt, *_rest) = simulate_extended(paradigm, noise=0.5,
                                                    T=T, seed=s)
            except Exception as exc:
                print(f"{paradigm:<20} {s:>4}  sim failed: {exc}")
                continue
            X = X_pt

            # BEFORE: monkey-patch the GPU lag stub back to the CPU lag to
            # simulate the pre-Phase-4.5 driver (contemp on GPU, lag on CPU).
            orig_gpu_lag = driver_mod.gpu_cits_lag_rcit

            def _cpu_lag_shim(X_arg, alpha=0.05, tau=1, K=25, n_perm=100,
                              max_cond_size=5, seed=0, device=None,
                              null='gamma', verbose=False):
                return cpu_lag(X_arg, alpha=alpha, tau=tau, K=K,
                                n_perm=n_perm, max_cond_size=max_cond_size,
                                seed=seed, verbose=verbose)

            driver_mod.gpu_cits_lag_rcit = _cpu_lag_shim
            try:
                # Warm-up GPU (PC side) once with the patched lag in place.
                _ = driver_mod.run_cits_plus_v_opt_rcit_nosp_gpu(
                    X, alpha=0.05, tau=2, K=25, max_cond_size=5,
                    seed=0, device=device, null='gamma')
                if str(device).startswith('cuda'):
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = driver_mod.run_cits_plus_v_opt_rcit_nosp_gpu(
                    X, alpha=0.05, tau=2, K=25, max_cond_size=5,
                    seed=s, device=device, null='gamma')
                if str(device).startswith('cuda'):
                    torch.cuda.synchronize()
                t_before = time.perf_counter() - t0
            except Exception as exc:
                print(f"{paradigm:<20} {s:>4}  BEFORE failed: {exc}")
                driver_mod.gpu_cits_lag_rcit = orig_gpu_lag
                continue
            finally:
                driver_mod.gpu_cits_lag_rcit = orig_gpu_lag

            # AFTER: full GPU driver (lag side already swapped via the file).
            try:
                if str(device).startswith('cuda'):
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = driver_mod.run_cits_plus_v_opt_rcit_nosp_gpu(
                    X, alpha=0.05, tau=2, K=25, max_cond_size=5,
                    seed=s, device=device, null='gamma')
                if str(device).startswith('cuda'):
                    torch.cuda.synchronize()
                t_after = time.perf_counter() - t0
            except Exception as exc:
                print(f"{paradigm:<20} {s:>4}  AFTER failed: {exc}")
                continue

            speedup = t_before / max(t_after, 1e-9)
            print(f"{paradigm:<20} {s:>4} {t_before:>22.2f} "
                  f"{t_after:>22.2f} {speedup:>9.1f}x")


def _phase46_validate_lag_dtype(device: str = 'cuda:0', n_seeds: int = 10,
                                  T: int = 1000) -> bool:
    """Phase 4.6 validation: float32 vs float64 lag adjacency match.

    Acceptance:
      - per-cell agreement rate >= 0.95 across seeds (p=4 means a single
        edge disagreement drops Jaccard sharply, so we report cell-level
        agreement as well)
      - mean Jaccard across seeds >= 0.85.
    """
    try:
        from simulation_benchmark_fc_methods_v3 import simulate_extended
    except ImportError as e:
        print(f"[FATAL] {e}")
        return False
    print("\n=== Phase 4.6: fp32 vs fp64 lag adjacency ===\n")
    jaccs = []
    cell_match = 0
    cell_total = 0
    for s in range(n_seeds):
        (X_pt, *_) = simulate_extended('lingauss2+mixed', noise=0.5, T=T, seed=s)
        X = X_pt
        B32 = gpu_cits_lag_rcit(X, alpha=0.05, tau=1, K=25, max_cond_size=5,
                                  seed=s, device=device, null='gamma',
                                  dtype=torch.float32)
        B64 = gpu_cits_lag_rcit(X, alpha=0.05, tau=1, K=25, max_cond_size=5,
                                  seed=s, device=device, null='gamma',
                                  dtype=torch.float64)
        j = _jaccard(B32, B64)
        jaccs.append(j)
        cell_match += int((B32 == B64).sum())
        cell_total += int(B32.size)
        print(f"  seed {s}: fp32 edges={int(B32.sum())} "
              f"fp64 edges={int(B64.sum())} Jaccard={j:.3f}")
    mean_j = float(np.mean(jaccs))
    cell_rate = cell_match / max(cell_total, 1)
    ok = (mean_j >= 0.85) and (cell_rate >= 0.95)
    print(f"  -> mean Jaccard={mean_j:.3f}, cell agreement={cell_rate:.4f}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def _phase46_validate_lag_batched(device: str = 'cuda:0', n_sims: int = 4,
                                    T: int = 1000) -> bool:
    """Phase 4.6 validation: batched vs sequential lag exactly match."""
    try:
        from simulation_benchmark_fc_methods_v3 import simulate_extended
    except ImportError as e:
        print(f"[FATAL] {e}")
        return False
    print("\n=== Phase 4.6: lag batched vs sequential ===\n")
    X_list = []
    for s in range(n_sims):
        (X_pt, *_) = simulate_extended('lingauss2+mixed', noise=0.5, T=T, seed=s)
        X_list.append(X_pt)
    seeds = list(range(n_sims))
    seq = [gpu_cits_lag_rcit(X, alpha=0.05, tau=1, K=25, max_cond_size=5,
                                seed=s, device=device, null='gamma')
              for X, s in zip(X_list, seeds)]
    bat = gpu_cits_lag_rcit_batched(X_list, alpha=0.05, tau=1, K=25,
                                       max_cond_size=5, seeds=seeds,
                                       device=device, null='gamma')
    all_ok = True
    for k, (a, b) in enumerate(zip(seq, bat)):
        match = np.array_equal(a, b)
        print(f"  sim {k}: seq edges={int(a.sum())} bat edges={int(b.sum())} match={match}")
        all_ok = all_ok and match
    print(f"  -> {'PASS' if all_ok else 'FAIL'}")
    return all_ok


if __name__ == '__main__':
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {dev}\n")

    ok = _run_adjacency_match(device=dev, n_seeds=5, T=1000)
    _speed_benchmark(device=dev, T=1000)
    _end_to_end_before_after(device=dev, T=1000)

    # Phase 4.6 additions
    ok_dtype = _phase46_validate_lag_dtype(device=dev, n_seeds=3, T=1000)
    ok_batch = _phase46_validate_lag_batched(device=dev, n_sims=4, T=1000)
    overall = ok and ok_dtype and ok_batch
    print(f"\nOVERALL: {'PASSED' if overall else 'FAILED'}")
    sys.exit(0 if overall else 1)
