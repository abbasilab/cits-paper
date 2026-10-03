"""
directed_benchmark_cpu.py

Directed-CS benchmark for all CPU methods (methods 18-26 from the 26-method plan).

Each method runs on the standard directed benchmark grid:
  10 paradigms x 50 seeds x T=1000 x noise=1.0 = 500 evals per method.

Output per method:
  simulation_results_directed_<METHOD_TAG>.csv
  in $CITS_PAPER_OUT/table1_baselines/ (originally arousal_paper_overleaf/figures/2026-06-06_directed_cs/)

Methods included:
  18. granger
  19. tpc_original
  20. pc_vanilla
  21. cits_lag_only
  22. verb_meek
  23. cits_plus_v1
  24. cits_plus_v2
  25. cits_plus_v_latest
  26. cits_plus_v_optimal

Usage:
  python directed_benchmark_cpu.py --workers 16

  # Specific methods
  python directed_benchmark_cpu.py --methods 18,19,20

  # Smoke test
  python directed_benchmark_cpu.py --smoke

  # Skip if CSV exists
  python directed_benchmark_cpu.py --skip-existing

CLI flags:
  --workers N         Number of parallel worker processes (default 16)
  --seeds N           Seeds per (model, T) combo (default 50)
  --smoke             3 paradigms x 3 seeds, method 18 (Granger) only
  --methods STR       Comma-sep method indices (18-26). Default: all
  --skip-existing     Skip if output CSV already exists with full row count
  --batch-size N      Tasks per worker call (default 4)
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

warnings.filterwarnings('ignore')

_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_SHARED = os.path.abspath(os.path.join(_THIS_DIR, '..', 'shared'))
sys.path.insert(0, _SHARED)
from paths import outdir as _outdir
OUT_DIR    = _outdir('table1_baselines')   # was arousal_paper_overleaf/figures/2026-06-06_directed_cs

for _p in [_THIS_DIR, _SHARED]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

ALPHA        = 0.05
N_NEURONS    = 4
T            = 1000
NOISE        = 1.0
N_SEEDS      = 50
CHECKPOINT_EVERY = 50

MODEL_NAMES = [
    'lingauss1',        'lingauss1+mixed',
    'lingauss2',        'lingauss2+mixed',
    'nonlinnongauss1',  'nonlinnongauss1+mixed',
    'nonlinnongauss2',  'nonlinnongauss2+mixed',
    'ctrnn',            'ctrnn+mixed',
]
SMOKE_MODELS = ['lingauss1+mixed', 'lingauss2+mixed', 'ctrnn+mixed']

# Method registry for CPU methods
# (index, tag, csv_method_name, run_fn_key)
CPU_METHOD_DEFS = [
    (18, 'granger',            'Granger',                  'run_granger'),
    (19, 'tpc_original',       'TPC',                      'run_tpc'),
    (20, 'pc_vanilla',         'PC',                       'run_pc_vanilla'),
    (21, 'cits_lag_only',      'CITS',                     'run_cits'),
    (22, 'verb_meek',          'VerB',                     'run_ver_b'),
    (23, 'cits_plus_v1',       'CITSplus_v1',              'run_cits_plus_v1'),
    (24, 'cits_plus_v2',       'CITSplus_v2',              'run_cits_plus_v2'),
    (25, 'cits_plus_v_latest', 'CITSplus_v_latest',        'run_cits_plus_v_latest'),
    (26, 'cits_plus_v_optimal','CITSplus_v_optimal',       'run_cits_plus_v_optimal'),
]

# Per-worker method key (set in initializer)
_WORKER_RUN_KEY = None
_WORKER_CSV_NAME = None


def _init_cpu_worker(run_key: str, csv_name: str) -> None:
    global _WORKER_RUN_KEY, _WORKER_CSV_NAME
    _WORKER_RUN_KEY  = run_key
    _WORKER_CSV_NAME = csv_name


def _run_cpu_fn(X: np.ndarray, run_key: str) -> np.ndarray:
    """Dispatch to the appropriate CPU run function."""
    from simulation_benchmark_fc_methods_v3 import (
        run_granger, run_tpc, run_pc as run_pc_v3, run_cits,
        run_verB, run_cits_plus_v2,
        run_cits_plus_v_latest_bench,
    )
    from cits_plus_v_optimal import run_cits_plus_v_optimal as _v_opt

    if run_key == 'run_granger':
        return run_granger(X, ALPHA)
    elif run_key == 'run_tpc':
        return run_tpc(X, ALPHA)
    elif run_key == 'run_pc_vanilla':
        return run_pc_v3(X, ALPHA)
    elif run_key == 'run_cits':
        return run_cits(X, ALPHA)
    elif run_key == 'run_ver_b':
        return run_verB(X, ALPHA, meek=True)
    elif run_key == 'run_cits_plus_v1':
        return run_verB(X, ALPHA, meek=False)
    elif run_key == 'run_cits_plus_v2':
        return run_cits_plus_v2(X, ALPHA)
    elif run_key == 'run_cits_plus_v_latest':
        return run_cits_plus_v_latest_bench(X, ALPHA)
    elif run_key == 'run_cits_plus_v_optimal':
        return _v_opt(X, alpha=ALPHA)
    else:
        raise ValueError(f"Unknown CPU run key: {run_key}")


def _run_cpu_batch(batch: list) -> list:
    """Process a batch of (model, T, noise, seed) for the worker's method."""
    from simulation_benchmark_fc_methods_v3 import simulate_extended
    from directed_metrics import compute_directed_metrics

    rows = []
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

        t0 = time.perf_counter()
        try:
            pred = _run_cpu_fn(X, _WORKER_RUN_KEY)
        except Exception as exc:
            warnings.warn(f"{_WORKER_RUN_KEY} failed {model_name} seed={seed}: {exc}")
            pred = np.zeros((N_NEURONS, N_NEURONS), dtype=int)
        elapsed = time.perf_counter() - t0

        m = compute_directed_metrics(
            pred,
            gt_lag_w, gt_contemp_w,
            gt_both_lag_w, gt_both_contemp_w,
            gt_lag_uw, gt_contemp_uw, gt_both_uw,
        )

        row = {
            'model'           : model_name,
            'seed'            : seed,
            'T'               : _T,
            'noise'           : _noise,
            'method'          : _WORKER_CSV_NAME,
            'device'          : 'cpu',
            'runtime_sec'     : round(elapsed, 4),
            'directed_TP'     : m['directed_tp_strict'],
            'directed_FP'     : m['directed_fp_strict'],
            'directed_FN'     : m['directed_fn_strict'],
            'directed_TN'     : m['n_neg_directed'] - m['directed_fp_strict'],
            'directed_TPR'    : m['directed_TPR_strict'],
            'directed_FPR'    : m['directed_FPR_strict'],
            'directed_CS'     : m['directed_CS_strict'],
            'directed_CS_lenient'  : m['directed_CS_lenient'],
            'directed_TPR_lenient' : m['directed_TPR_lenient'],
            'directed_FPR_lenient' : m['directed_FPR_lenient'],
            'n_true_directed' : m['n_true_directed'],
            'n_neg_directed'  : m['n_neg_directed'],
        }
        for k in ('edges_tp', 'edges_fp', 'edges_fn', 'edges_F1',
                  'edges_precision', 'edges_recall', 'edges_SHD'):
            row[k] = m.get(k, float('nan'))
        rows.append(row)

    return rows


def _run_cpu_method(method_def: tuple, models: list, n_seeds: int,
                    workers: int, batch_size: int,
                    skip_existing: bool, is_smoke: bool) -> str:
    (_idx, tag, csv_name, run_key) = method_def

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

    print(f"\n[{tag}] {len(tasks)} evals | workers={workers}", flush=True)

    t_start  = time.time()
    all_rows = []
    partial_csv = out_csv.replace('.csv', '_partial.csv') if not is_smoke else None
    last_ckpt   = 0

    ctx = mp.get_context('fork')
    with ctx.Pool(processes=workers,
                  initializer=_init_cpu_worker,
                  initargs=(run_key, csv_name)) as pool:
        for i, result in enumerate(
                pool.imap_unordered(_run_cpu_batch, task_batches, chunksize=1)):
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
        description='Directed-CS benchmark for CPU methods (18-26).')
    parser.add_argument('--workers',       type=int, default=16)
    parser.add_argument('--seeds',         type=int, default=N_SEEDS)
    parser.add_argument('--smoke',         action='store_true',
                        help='3 paradigms x 3 seeds, method 18 only.')
    parser.add_argument('--methods',       type=str, default=None,
                        help='Comma-sep method indices (18-26). Default: all.')
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--batch-size',    type=int, default=4)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.smoke:
        method_indices = [18]
        models  = SMOKE_MODELS
        n_seeds = 3
    else:
        if args.methods:
            method_indices = [int(x) for x in args.methods.split(',')]
        else:
            method_indices = list(range(18, 27))
        models  = MODEL_NAMES
        n_seeds = args.seeds

    selected = [m for m in CPU_METHOD_DEFS if m[0] in method_indices]
    if not selected:
        print("No methods matched. Check --methods values.", flush=True)
        return

    print(f"[directed_benchmark_cpu] Running {len(selected)} CPU methods:", flush=True)
    for m in selected:
        print(f"  [{m[0]:2d}] {m[1]}", flush=True)

    t_global = time.time()
    results  = []
    for mdef in selected:
        try:
            out_csv = _run_cpu_method(
                mdef, models, n_seeds,
                workers=args.workers,
                batch_size=args.batch_size,
                skip_existing=args.skip_existing,
                is_smoke=args.smoke,
            )
            results.append((mdef[1], 'OK', out_csv))
        except Exception as exc:
            results.append((mdef[1], f'FAILED: {exc}', ''))
            print(f"[ERROR] Method {mdef[1]} failed: {exc}", flush=True)

    print(f"\n[directed_benchmark_cpu] All done in "
          f"{(time.time() - t_global)/3600:.2f} hr", flush=True)
    print("\nSummary:")
    for tag, status, path in results:
        print(f"  {tag:<30s}  {status}")


if __name__ == '__main__':
    main()
