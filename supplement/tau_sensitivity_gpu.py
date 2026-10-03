"""
tau_sensitivity_gpu.py

Item 1: τ sensitivity on the 5 CITS-paper base regimes using GPU CITS with
max_cond_size=5 (standard practical cap). Sidesteps the O(2^k) powerset
explosion that made CPU CITS at τ>1 intractable.

Regimes:  lingauss1, lingauss2, nonlinnongauss1, nonlinnongauss2, ctrnn
Tau:      1, 2, 3
Seeds:    20 per (regime, tau)
CI test:  RCIT-GPU (HSIC-approx per CITS paper spec for non-Gaussian).
          RCIT also works for Gaussian regimes; CS numbers are directly
          comparable across regimes.

Output:
  simulation_results_directed_tau_sensitivity_gpu.csv
"""
from __future__ import annotations
import os, sys, time, argparse, warnings, itertools
import multiprocessing as mp
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SHARED = os.path.abspath(os.path.join(_THIS_DIR, '..', 'shared'))
sys.path.insert(0, _SHARED)
from paths import outdir as _outdir
OUT_DIR = _outdir('supplement')   # was arousal_paper_overleaf/figures/2026-06-06_directed_cs
OUT_CSV = os.path.join(OUT_DIR, 'simulation_results_directed_tau_sensitivity_gpu.csv')

for _p in [_THIS_DIR, _SHARED]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

T = 1000
NOISE = 1.0
ALPHA = 0.05
N_SEEDS = 20
MAX_COND_SIZE = 5   # standard practical cap; matches gpu_cits_lag_rcit default
N_GPUS = 8

REGIMES = ['lingauss1', 'lingauss2', 'nonlinnongauss1', 'nonlinnongauss2', 'ctrnn']
TAU_VALUES = [1, 2, 3]


def _run_one(args) -> dict:
    from simulation_benchmark_fc_methods_v3 import simulate_extended
    from directed_metrics import compute_directed_metrics
    from gpu_cits_lag_rcit import gpu_cits_lag_rcit

    regime, seed, tau, gpu_id = args
    (X, gt_lag_uw, gt_lag_w, gt_c_uw, gt_c_w,
     gt_b_uw, gt_bl_w, gt_bc_w) = simulate_extended(regime, NOISE, T, seed)

    t0 = time.perf_counter()
    try:
        pred = gpu_cits_lag_rcit(X.astype(np.float64), alpha=ALPHA, tau=tau,
                                  max_cond_size=MAX_COND_SIZE,
                                  device=f'cuda:{gpu_id}')
        note = ''
    except Exception as exc:
        pred = np.zeros_like(gt_lag_uw)
        note = f'failed: {exc}'
    elapsed = time.perf_counter() - t0

    m = compute_directed_metrics(pred, gt_lag_w, gt_c_w, gt_bl_w, gt_bc_w,
                                  gt_lag_uw, gt_c_uw, gt_b_uw)
    return {
        'regime': regime, 'tau': tau, 'seed': seed, 'T': T, 'noise': NOISE,
        'method': 'CITS_gpu_rcit', 'max_cond_size': MAX_COND_SIZE,
        'runtime_sec': round(elapsed, 4), 'note': note,
        'directed_TPR': m['directed_TPR_strict'],
        'directed_FPR': m['directed_FPR_strict'],
        'directed_CS':  m['directed_CS_strict'],
        'directed_CS_lenient': m['directed_CS_lenient'],
        'n_true_directed': m['n_true_directed'],
        'n_neg_directed':  m['n_neg_directed'],
        'n_pred_edges': int(pred.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=N_SEEDS)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    n_seeds = 3 if args.smoke else args.seeds
    tasks = []
    for i, (r, s, tau) in enumerate(itertools.product(REGIMES, range(n_seeds), TAU_VALUES)):
        tasks.append((r, s, tau, i % N_GPUS))

    print(f"[tau_sensitivity_gpu] {len(tasks)} tasks, {N_GPUS} GPUs", flush=True)

    all_rows = []
    t_start = time.time()
    partial_csv = OUT_CSV.replace('.csv', '_partial.csv')
    ctx = mp.get_context('spawn')
    with ctx.Pool(N_GPUS) as pool:
        for i, row in enumerate(pool.imap_unordered(_run_one, tasks)):
            all_rows.append(row)
            if (i + 1) % 10 == 0:
                pd.DataFrame(all_rows).to_csv(partial_csv, index=False)
                elapsed = time.time() - t_start
                print(f"  done {i+1}/{len(tasks)}  ({elapsed:.0f}s)", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"[tau_sensitivity_gpu] SAVED -> {OUT_CSV}")
    print("\nMean CS by regime × tau:")
    print(df.pivot_table(index='regime', columns='tau',
                          values='directed_CS', aggfunc='mean').round(3))


if __name__ == '__main__':
    main()
