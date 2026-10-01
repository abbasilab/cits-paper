"""
directed_benchmark_gpu.py

Directed-CS benchmark for all GPU methods (methods 1-17 from the 26-method plan).

Each method runs on the standard directed benchmark grid:
  10 paradigms x 50 seeds x T=1000 x noise=1.0 = 500 evals per method.

Output per method:
  simulation_results_directed_<METHOD_TAG>.csv
  in /home/rbiswas1/microns/arousal_paper_overleaf/figures/2026-06-06_directed_cs/

Uses CUDA_VISIBLE_DEVICES=3,4,5,6,7 (set in the calling environment).
Workers pin to a distinct GPU by pool rank.

Directed metrics come from compute_directed_metrics() (strict mode).
Output columns: model, seed, T, noise, method, directed_TP, directed_FP,
  directed_FN, directed_TN, directed_TPR, directed_FPR, directed_CS,
  plus lenient variants and undirected breakdown.

Methods included (by METHOD_TAG):
  1.  v_opt_nosp_gpu_tau1_gamma
  2.  v_opt_nosp_gpu_adapttau_gamma
  3.  v_opt_nosp_gpu_tau1_perm
  4.  v_opt_nosp_gpu_adapttau_perm
  5.  v_opt_nosp_gpu_nosp_off_gamma        (restrict_self_past_contemp=False)
  6.  v1_rcit_gpu_gamma
  7.  v1_rcit_gpu_perm
  8.  tpc_rcit_gpu_gamma
  9.  v_lag_inf_direct_gamma
  10. v_lag_inf_full_direct_gamma
  11. v_lag_inf_full_direct_alpha001_gamma
  12. v_lag_inf_chi_gamma
  13. v_lag_inf_ges_gamma
  14. pcmci_plus
  15. v_lag_inf_direct_perm
  16. v_lag_inf_full_direct_perm
  17. v_lag_inf_chi_perm

Usage:
  # Full run (all 17 methods)
  CUDA_VISIBLE_DEVICES=3,4,5,6,7 python directed_benchmark_gpu.py --workers 15

  # Run specific methods only (comma-sep indices 1-17)
  CUDA_VISIBLE_DEVICES=3,4,5,6,7 python directed_benchmark_gpu.py --methods 1,2,5

  # Smoke test (3 paradigms x 3 seeds, method 1 only)
  CUDA_VISIBLE_DEVICES=3,4,5,6,7 python directed_benchmark_gpu.py --smoke

  # Skip methods whose output CSV already exists
  CUDA_VISIBLE_DEVICES=3,4,5,6,7 python directed_benchmark_gpu.py --skip-existing

CLI flags:
  --workers N       Pool workers (default 15)
  --seeds N         Seeds per combo (default 50)
  --smoke           Fast smoke: 3 paradigms x 3 seeds, method 1 only
  --methods STR     Comma-sep method indices to run (default: all)
  --skip-existing   Skip a method if its output CSV already exists and has rows
  --batch-size N    Tasks per worker call (default 8)
"""
from __future__ import annotations

import os
import sys
import time
import argparse
import warnings
import itertools
import subprocess
import multiprocessing as mp

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_REPO_CITS = '/home/rbiswas1/repos/cits'
_OVERLEAF  = '/home/rbiswas1/microns/arousal_paper_overleaf'
OUT_DIR    = os.path.join(_OVERLEAF, 'figures/2026-06-06_directed_cs')

for _p in [_THIS_DIR, _REPO_CITS]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

ALPHA        = 0.05
N_NEURONS    = 4
T            = 1000
NOISE        = 1.0
N_SEEDS      = 50
RCIT_K       = 25
RCIT_NPERM   = 100
RCIT_MAX_COND_DEFAULT = 5
CHECKPOINT_EVERY = 100

MODEL_NAMES = [
    'lingauss1',        'lingauss1+mixed',
    'lingauss2',        'lingauss2+mixed',
    'nonlinnongauss1',  'nonlinnongauss1+mixed',
    'nonlinnongauss2',  'nonlinnongauss2+mixed',
    'ctrnn',            'ctrnn+mixed',
]
SMOKE_MODELS = ['lingauss1+mixed', 'lingauss2+mixed', 'ctrnn+mixed']

# ---------------------------------------------------------------------------
# Method registry
# Each entry: (tag, method_name_in_csv, driver_key, null, restrict_nosp,
#              tau_or_None, max_cond, is_perm_cpu_only)
# driver_key: identifies which import / run function to use
# ---------------------------------------------------------------------------
METHOD_DEFS = [
    # idx  tag                              csv_name                                       driver                        null     nosp_flag  tau     mc    perm_cpu
    ( 1, 'v_opt_nosp_gpu_tau1_gamma',       'CITSplus_v_opt_rcit_nosp_gpu_tau1',           'v_opt_nosp_gpu',            'gamma', True,       1,      5,    False),
    ( 2, 'v_opt_nosp_gpu_adapttau_gamma',   'CITSplus_v_opt_rcit_nosp_gpu_adapttau',       'v_opt_nosp_gpu_adapttau',   'gamma', True,       None,   5,    False),
    ( 3, 'v_opt_nosp_gpu_tau1_perm',        'CITSplus_v_opt_rcit_nosp_gpu_tau1_perm',      'v_opt_nosp_gpu',            'perm',  True,       1,      5,    False),
    ( 4, 'v_opt_nosp_gpu_adapttau_perm',    'CITSplus_v_opt_rcit_nosp_gpu_adapttau_perm',  'v_opt_nosp_gpu_adapttau',   'perm',  True,       None,   5,    False),
    ( 5, 'v_opt_nosp_off_gamma',            'CITSplus_v_opt_rcit_gpu_nosp_off',            'v_opt_nosp_gpu',            'gamma', False,      1,      5,    False),
    ( 6, 'v1_rcit_gpu_gamma',               'CITSplus_v1_rcit_gpu',                        'v1_rcit_gpu',               'gamma', None,       None,   5,    False),
    ( 7, 'v1_rcit_gpu_perm',                'CITSplus_v1_rcit_gpu_perm',                   'v1_rcit_gpu',               'perm',  None,       None,   5,    False),
    ( 8, 'tpc_rcit_gpu_gamma',              'TPC_rcit_gpu',                                'tpc_rcit_gpu',              'gamma', False,      1,      None, False),
    ( 9, 'v_lag_inf_direct_gamma',          'CITSplus_v_lag_informed_direct_rcit_gpu',     'v_lag_direct',              'gamma', None,       2,      None, False),
    (10, 'v_lag_inf_full_direct_gamma',     'CITSplus_v_lag_informed_full_direct_rcit_gpu','v_lag_full_direct',         'gamma', None,       2,      None, False),
    (11, 'v_lag_inf_full_direct_a001_gamma','CITSplus_v_lag_informed_full_direct_rcit_gpu_alpha001','v_lag_full_direct_a001','gamma',None,    2,      None, False),
    (12, 'v_lag_inf_chi_gamma',             'CITSplus_v_lag_informed_chi_rcit_gpu',        'v_lag_chi',                 'gamma', None,       2,      None, False),
    (13, 'v_lag_inf_ges_gamma',             'CITSplus_v_lag_informed_ges_rcit_gpu',        'v_lag_ges',                 'gamma', None,       2,      None, False),
    (14, 'pcmci_plus',                      'PCMCI_plus',                                  'pcmci_plus',                None,    None,       None,   None, True),
    (15, 'v_lag_inf_direct_perm',           'CITSplus_v_lag_informed_direct_rcit_gpu_perm','v_lag_direct',              'perm',  None,       2,      None, False),
    (16, 'v_lag_inf_full_direct_perm',      'CITSplus_v_lag_informed_full_direct_rcit_gpu_perm','v_lag_full_direct',    'perm',  None,       2,      None, False),
    (17, 'v_lag_inf_chi_perm',              'CITSplus_v_lag_informed_chi_rcit_gpu_perm',   'v_lag_chi',                 'perm',  None,       2,      None, False),
]

# Per-worker globals (set in _init_worker)
_WORKER_DEVICE = 'cuda:0'
_WORKER_NULL   = 'gamma'
_WORKER_METHOD = None   # set per-subprocess via initargs


def _init_worker(n_gpus: int, null_mode: str, method_def: tuple) -> None:
    """Pin each spawned worker to a distinct GPU by pool rank."""
    global _WORKER_DEVICE, _WORKER_NULL, _WORKER_METHOD
    _WORKER_NULL   = null_mode
    _WORKER_METHOD = method_def
    # method_def index 8 = perm_cpu_only flag
    perm_cpu_only = method_def[8]
    if null_mode in ('perm', 'permutation') or perm_cpu_only:
        _WORKER_DEVICE = 'cpu'
        return
    try:
        import torch
        ident  = mp.current_process()._identity
        rank   = (ident[0] - 1) if ident else 0
        gpu_id = rank % max(n_gpus, 1)
        torch.cuda.set_device(gpu_id)
        _WORKER_DEVICE = f'cuda:{gpu_id}'
    except Exception:
        _WORKER_DEVICE = 'cuda:0'


def _run_one_batch(batch: list) -> list:
    """Run a batch of (model, T, noise, seed) tuples using the current worker's method."""
    import torch

    from simulation_benchmark_fc_methods_v3 import simulate_extended
    from directed_metrics import compute_directed_metrics

    (_idx, tag, csv_name, driver_key, null, nosp_flag,
     tau, max_cond, perm_cpu_only) = _WORKER_METHOD

    rows    = []
    X_list   = []
    metas    = []

    for (model_name, _T, _noise, seed) in batch:
        try:
            (X,
             gt_lag_uw,  gt_lag_w,
             gt_contemp_uw, gt_contemp_w,
             gt_both_uw, gt_both_lag_w, gt_both_contemp_w,
            ) = simulate_extended(model_name, _noise, _T, seed)
        except Exception as exc:
            warnings.warn(f"Sim failed {model_name} T={_T} seed={seed}: {exc}")
            continue
        X_list.append(X.astype(np.float64))
        metas.append((model_name, _T, _noise, seed,
                      gt_lag_uw,  gt_lag_w,
                      gt_contemp_uw, gt_contemp_w,
                      gt_both_uw, gt_both_lag_w, gt_both_contemp_w))

    if not X_list:
        return []

    t0    = time.perf_counter()
    preds = _dispatch_driver(driver_key, X_list, metas, null, nosp_flag,
                             tau, max_cond)
    elapsed  = time.perf_counter() - t0
    per_eval = elapsed / max(len(metas), 1)

    for pred, meta in zip(preds, metas):
        (model_name, _T, _noise, seed,
         gt_lag_uw,  gt_lag_w,
         gt_contemp_uw, gt_contemp_w,
         gt_both_uw, gt_both_lag_w, gt_both_contemp_w) = meta

        m = compute_directed_metrics(
            pred,
            gt_lag_w, gt_contemp_w,
            gt_both_lag_w, gt_both_contemp_w,
            gt_lag_uw, gt_contemp_uw, gt_both_uw,
        )

        # Canonical column names requested in spec
        row = {
            'model'           : model_name,
            'seed'            : seed,
            'T'               : _T,
            'noise'           : _noise,
            'method'          : csv_name,
            'device'          : _WORKER_DEVICE,
            'runtime_sec'     : round(per_eval, 4),
            # Directed metrics (strict) — canonical names
            'directed_TP'     : m['directed_tp_strict'],
            'directed_FP'     : m['directed_fp_strict'],
            'directed_FN'     : m['directed_fn_strict'],
            'directed_TN'     : m['n_neg_directed'] - m['directed_fp_strict'],
            'directed_TPR'    : m['directed_TPR_strict'],
            'directed_FPR'    : m['directed_FPR_strict'],
            'directed_CS'     : m['directed_CS_strict'],
            # Lenient extras
            'directed_CS_lenient'  : m['directed_CS_lenient'],
            'directed_TPR_lenient' : m['directed_TPR_lenient'],
            'directed_FPR_lenient' : m['directed_FPR_lenient'],
            # Counts
            'n_true_directed' : m['n_true_directed'],
            'n_neg_directed'  : m['n_neg_directed'],
        }
        # Append undirected metrics for reference
        for k in ('edges_tp', 'edges_fp', 'edges_fn', 'edges_F1',
                  'edges_precision', 'edges_recall', 'edges_SHD'):
            row[k] = m.get(k, float('nan'))
        rows.append(row)

    return rows


def _dispatch_driver(driver_key, X_list, metas, null, nosp_flag, tau, max_cond):
    """Call the appropriate GPU/CPU driver and return a list of pred matrices."""
    import torch
    n = len(X_list)
    seeds = [m[3] for m in metas]

    try:
        if driver_key == 'v_opt_nosp_gpu':
            from cits_plus_v_opt_rcit_nosp_gpu import (
                run_cits_plus_v_opt_rcit_nosp_gpu_batched,
            )
            _tau = tau if tau is not None else 1
            return run_cits_plus_v_opt_rcit_nosp_gpu_batched(
                X_list, alpha=ALPHA, tau=_tau,
                K=RCIT_K, n_perm=RCIT_NPERM,
                max_cond_size=max_cond,
                restrict_self_past_contemp=bool(nosp_flag),
                seeds=seeds, device=_WORKER_DEVICE,
                null=null, dtype=torch.float32,
            )

        elif driver_key == 'v_opt_nosp_gpu_adapttau':
            from cits_plus_v_opt_rcit_nosp_gpu import (
                run_cits_plus_v_opt_rcit_nosp_gpu_batched,
            )
            from tau_max_pacf import (
                estimate_tau_max_global_var_bic, GLOBAL_TAU_MAX,
            )
            from collections import defaultdict
            metas_by_tau = defaultdict(list)
            X_by_tau     = defaultdict(list)
            for X, seed in zip(X_list, seeds):
                try:
                    _tau = int(estimate_tau_max_global_var_bic(
                        X.astype(np.float64), global_cap=GLOBAL_TAU_MAX))
                    _tau = max(1, min(_tau, GLOBAL_TAU_MAX))
                except Exception:
                    _tau = 1
                X_by_tau[_tau].append(X)
                metas_by_tau[_tau].append(seed)
            preds_by_seed = {}
            for _tau_val, X_group in X_by_tau.items():
                _seeds = metas_by_tau[_tau_val]
                p_ = run_cits_plus_v_opt_rcit_nosp_gpu_batched(
                    X_group, alpha=ALPHA, tau=_tau_val,
                    K=RCIT_K, n_perm=RCIT_NPERM,
                    max_cond_size=max_cond,
                    restrict_self_past_contemp=True,
                    seeds=_seeds, device=_WORKER_DEVICE,
                    null=null, dtype=torch.float32,
                )
                for s, p__ in zip(_seeds, p_):
                    preds_by_seed[s] = p__
            return [preds_by_seed[s] for s in seeds]

        elif driver_key == 'v1_rcit_gpu':
            from cits_plus_v1_rcit_gpu import (
                run_cits_plus_v1_rcit_gpu_batched,
            )
            return run_cits_plus_v1_rcit_gpu_batched(
                X_list, alpha=ALPHA,
                K=RCIT_K, n_perm=RCIT_NPERM,
                max_cond_size=max_cond,
                seeds=seeds, device=_WORKER_DEVICE,
                null=null, dtype=torch.float32,
            )

        elif driver_key == 'tpc_rcit_gpu':
            from tpc_rcit_gpu import run_tpc_rcit_gpu_batched
            return run_tpc_rcit_gpu_batched(
                X_list, alpha=ALPHA, tau=1,
                K=RCIT_K,
                max_cond_size=None,
                seeds=seeds, device=_WORKER_DEVICE,
                null='gamma', dtype=torch.float32,
            )

        elif driver_key == 'v_lag_direct':
            from cits_plus_v_lag_informed_direct_rcit_gpu import (
                run_cits_plus_v_lag_informed_direct_rcit_gpu_batched,
            )
            _tau = tau if tau is not None else 2
            return run_cits_plus_v_lag_informed_direct_rcit_gpu_batched(
                X_list, alpha=ALPHA, tau=_tau,
                K=RCIT_K, n_perm=RCIT_NPERM,
                max_cond_size=None,
                seeds=seeds, device=_WORKER_DEVICE,
                null=null, dtype=torch.float32,
            )

        elif driver_key in ('v_lag_full_direct', 'v_lag_full_direct_a001'):
            from cits_plus_v_lag_informed_full_direct_rcit_gpu import (
                run_cits_plus_v_lag_informed_full_direct_rcit_gpu_batched,
            )
            _tau   = tau if tau is not None else 2
            _alpha = 0.01 if driver_key == 'v_lag_full_direct_a001' else ALPHA
            return run_cits_plus_v_lag_informed_full_direct_rcit_gpu_batched(
                X_list, alpha=_alpha, tau=_tau,
                K=RCIT_K, n_perm=RCIT_NPERM,
                max_cond_size=None,
                seeds=seeds, device=_WORKER_DEVICE,
                null=null, dtype=torch.float32,
            )

        elif driver_key == 'v_lag_chi':
            from cits_plus_v_lag_informed_chi_rcit_gpu import (
                run_cits_plus_v_lag_informed_chi_rcit_gpu_batched,
            )
            _tau = tau if tau is not None else 2
            return run_cits_plus_v_lag_informed_chi_rcit_gpu_batched(
                X_list, alpha=ALPHA, tau=_tau,
                K=RCIT_K, n_perm=RCIT_NPERM,
                max_cond_size=None,
                seeds=seeds, device=_WORKER_DEVICE,
                null=null, dtype=torch.float32,
            )

        elif driver_key == 'v_lag_ges':
            from cits_plus_v_lag_informed_ges_rcit_gpu import (
                run_cits_plus_v_lag_informed_ges_rcit_gpu_batched,
            )
            _tau = tau if tau is not None else 2
            return run_cits_plus_v_lag_informed_ges_rcit_gpu_batched(
                X_list, alpha=ALPHA, tau=_tau,
                K=RCIT_K, n_perm=RCIT_NPERM,
                max_cond_size=None,
                seeds=seeds, device=_WORKER_DEVICE,
                null='gamma', dtype=torch.float32,
            )

        elif driver_key == 'pcmci_plus':
            from pcmci_plus_baseline import run_pcmci_plus
            preds = []
            for X in X_list:
                preds.append(run_pcmci_plus(X, pc_alpha=ALPHA, tau_max=2))
            return preds

        else:
            warnings.warn(f"Unknown driver_key: {driver_key}")
            p_ = X_list[0].shape[0]
            return [np.zeros((p_, p_), dtype=int) for _ in X_list]

    except Exception as exc:
        import traceback
        print(f"!!! Driver '{driver_key}' failed: {exc}")
        traceback.print_exc()
        warnings.warn(f"Driver '{driver_key}' failed: {exc}")
        p_ = X_list[0].shape[0] if X_list else N_NEURONS
        return [np.zeros((p_, p_), dtype=int) for _ in X_list]


def _run_method(method_def: tuple, models: list, n_seeds: int,
                workers: int, n_gpus: int, batch_size: int,
                skip_existing: bool, is_smoke: bool) -> str:
    """Run one method end-to-end. Returns path to output CSV."""
    (_idx, tag, csv_name, driver_key, null, nosp_flag,
     tau, max_cond, perm_cpu_only) = method_def

    out_csv = os.path.join(OUT_DIR, f'simulation_results_directed_{tag}.csv')

    if skip_existing and os.path.exists(out_csv):
        df_exist = pd.read_csv(out_csv)
        expected = len(models) * n_seeds
        if len(df_exist) >= expected:
            print(f"[SKIP] {tag}: {out_csv} already has {len(df_exist)} rows.",
                  flush=True)
            return out_csv
        else:
            print(f"[RESUME] {tag}: {out_csv} has {len(df_exist)} rows "
                  f"(expected {expected}). Rerunning.", flush=True)

    tasks = [(model, T, NOISE, seed)
             for model, seed in itertools.product(models, range(n_seeds))]
    task_batches = [tasks[i:i + batch_size]
                    for i in range(0, len(tasks), batch_size)]

    print(f"\n[{tag}] {len(tasks)} evals | workers={workers} | "
          f"n_gpus={n_gpus} | null={null}", flush=True)

    t_start  = time.time()
    all_rows = []
    partial_csv  = out_csv.replace('.csv', '_partial.csv') if not is_smoke else None
    last_ckpt    = 0

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=workers,
                  initializer=_init_worker,
                  initargs=(n_gpus, null or 'gamma', method_def)) as pool:
        for i, result in enumerate(
                pool.imap_unordered(_run_one_batch, task_batches, chunksize=1)):
            all_rows.extend(result)
            done  = min((i + 1) * batch_size, len(tasks))
            if (i + 1) % max(1, len(task_batches) // 10) == 0 or (i + 1) == len(task_batches):
                elapsed = time.time() - t_start
                eta     = elapsed / max(done, 1) * (len(tasks) - done)
                print(f"  [{tag}] {done}/{len(tasks)} "
                      f"elapsed={elapsed:.0f}s ETA={eta:.0f}s", flush=True)
            if (partial_csv is not None
                    and len(all_rows) - last_ckpt >= CHECKPOINT_EVERY):
                try:
                    pd.DataFrame(all_rows).to_csv(partial_csv, index=False)
                    last_ckpt = len(all_rows)
                except Exception:
                    pass

    df = pd.DataFrame(all_rows)
    if is_smoke:
        smoke_csv = out_csv.replace('.csv', '_smoke.csv')
        df.to_csv(smoke_csv, index=False)
        print(f"[{tag}] Smoke -> {smoke_csv}  ({len(df)} rows)", flush=True)
        if len(df):
            print(f"  Mean directed_CS: {df['directed_CS'].mean():.4f}")
        return smoke_csv
    else:
        df.to_csv(out_csv, index=False)
        print(f"[{tag}] Saved -> {out_csv}  ({len(df)} rows)", flush=True)
        if len(df):
            print(f"  Mean directed_CS: {df['directed_CS'].mean():.4f}")
        return out_csv


def main():
    parser = argparse.ArgumentParser(
        description='Directed-CS benchmark for all GPU methods.')
    parser.add_argument('--workers',       type=int,  default=15)
    parser.add_argument('--n-gpus',        type=int,  default=None)
    parser.add_argument('--seeds',         type=int,  default=N_SEEDS)
    parser.add_argument('--smoke',         action='store_true',
                        help='3 paradigms x 3 seeds, method 1 only.')
    parser.add_argument('--methods',       type=str,  default=None,
                        help='Comma-sep method indices (1-17). Default: all.')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip a method if its full output CSV exists.')
    parser.add_argument('--batch-size',    type=int,  default=8)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    import torch
    avail  = torch.cuda.device_count()
    n_gpus = args.n_gpus if args.n_gpus is not None else min(args.workers, avail)
    n_gpus = max(1, n_gpus)

    print(f"[directed_benchmark_gpu] GPUs avail={avail}, using {n_gpus}", flush=True)
    print(f"[directed_benchmark_gpu] OUT_DIR={OUT_DIR}", flush=True)

    if args.smoke:
        method_indices = [1]
        models  = SMOKE_MODELS
        n_seeds = 3
    else:
        if args.methods:
            method_indices = [int(x) for x in args.methods.split(',')]
        else:
            method_indices = list(range(1, 18))
        models  = MODEL_NAMES
        n_seeds = args.seeds

    selected = [m for m in METHOD_DEFS if m[0] in method_indices]
    if not selected:
        print("No methods matched. Check --methods values.", flush=True)
        return

    print(f"[directed_benchmark_gpu] Running {len(selected)} methods:", flush=True)
    for m in selected:
        print(f"  [{m[0]:2d}] {m[1]}", flush=True)

    t_global = time.time()
    results  = []
    for mdef in selected:
        try:
            out_csv = _run_method(
                mdef, models, n_seeds,
                workers=args.workers, n_gpus=n_gpus,
                batch_size=args.batch_size,
                skip_existing=args.skip_existing,
                is_smoke=args.smoke,
            )
            results.append((mdef[1], 'OK', out_csv))
        except Exception as exc:
            results.append((mdef[1], f'FAILED: {exc}', ''))
            print(f"[ERROR] Method {mdef[1]} failed: {exc}", flush=True)

    print(f"\n[directed_benchmark_gpu] All done in "
          f"{(time.time() - t_global)/3600:.2f} hr", flush=True)
    print("\nSummary:")
    for tag, status, path in results:
        print(f"  {tag:<55s}  {status}")


if __name__ == '__main__':
    main()
