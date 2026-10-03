"""
directed_benchmark_kernel_gc.py

Kernel Granger causality (Marinazzo, Pellicoro, Stramaglia PRL 2008) on the
10-regime directed-CS benchmark. Follows directed_benchmark_lpcmci.py pattern.

Output:
  simulation_results_directed_kernel_gc.csv
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
OUT_DIR = _outdir('table1_baselines')   # was arousal_paper_overleaf/figures/2026-06-06_directed_cs
OUT_CSV = os.path.join(OUT_DIR, 'simulation_results_directed_kernel_gc.csv')

for _p in [_THIS_DIR, _SHARED]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

ALPHA = 0.05
N_NEURONS = 4
T = 1000
NOISE = 1.0
N_SEEDS = 50
MAX_LAG = 2
CHECKPOINT_EVERY = 50

METHOD_TAG = 'kernel_gc'
METHOD_CSV_NAME = 'KernelGC'

MODEL_NAMES = [
    'lingauss1',        'lingauss1+mixed',
    'lingauss2',        'lingauss2+mixed',
    'nonlinnongauss1',  'nonlinnongauss1+mixed',
    'nonlinnongauss2',  'nonlinnongauss2+mixed',
    'ctrnn',            'ctrnn+mixed',
]
SMOKE_MODELS = ['lingauss1+mixed', 'nonlinnongauss1+mixed', 'ctrnn+mixed']


def _run_batch(batch: list) -> list:
    from simulation_benchmark_fc_methods_v3 import simulate_extended
    from directed_metrics import compute_directed_metrics
    from kernel_granger_baseline import run_kernel_granger

    rows = []
    for (model_name, _T, _noise, seed) in batch:
        try:
            (X, gt_lag_uw, gt_lag_w, gt_contemp_uw, gt_contemp_w,
             gt_both_uw, gt_both_lag_w, gt_both_contemp_w,
            ) = simulate_extended(model_name, _noise, _T, seed)
        except Exception as exc:
            warnings.warn(f"Sim failed {model_name} seed={seed}: {exc}")
            continue

        t0 = time.perf_counter()
        try:
            pred = run_kernel_granger(X.astype(np.float64), max_lag=MAX_LAG,
                                       alpha=ALPHA)
        except Exception as exc:
            warnings.warn(f"KernelGC failed {model_name} seed={seed}: {exc}")
            pred = np.zeros((N_NEURONS, N_NEURONS), dtype=int)
        elapsed = time.perf_counter() - t0

        m = compute_directed_metrics(
            pred, gt_lag_w, gt_contemp_w, gt_both_lag_w, gt_both_contemp_w,
            gt_lag_uw, gt_contemp_uw, gt_both_uw,
        )
        row = {
            'model':       model_name, 'seed': seed, 'T': _T, 'noise': _noise,
            'method':      METHOD_CSV_NAME, 'device': 'cpu',
            'runtime_sec': round(elapsed, 4),
            'directed_TP': m['directed_tp_strict'],
            'directed_FP': m['directed_fp_strict'],
            'directed_FN': m['directed_fn_strict'],
            'directed_TN': m['n_neg_directed'] - m['directed_fp_strict'],
            'directed_TPR': m['directed_TPR_strict'],
            'directed_FPR': m['directed_FPR_strict'],
            'directed_CS':  m['directed_CS_strict'],
            'directed_CS_lenient':  m['directed_CS_lenient'],
            'directed_TPR_lenient': m['directed_TPR_lenient'],
            'directed_FPR_lenient': m['directed_FPR_lenient'],
            'n_true_directed': m['n_true_directed'],
            'n_neg_directed':  m['n_neg_directed'],
        }
        for k in ('edges_tp', 'edges_fp', 'edges_fn', 'edges_F1',
                  'edges_precision', 'edges_recall', 'edges_SHD'):
            row[k] = m.get(k, float('nan'))
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description='Kernel-GC benchmark')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--seeds', type=int, default=N_SEEDS)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--skip-existing', action='store_true')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    if args.skip_existing and os.path.exists(OUT_CSV):
        df_exist = pd.read_csv(OUT_CSV)
        expected = len(MODEL_NAMES) * args.seeds
        if len(df_exist) >= expected:
            print(f"[SKIP] {OUT_CSV} has {len(df_exist)} rows.")
            return

    models = SMOKE_MODELS if args.smoke else MODEL_NAMES
    n_seeds = 3 if args.smoke else args.seeds
    tasks = [(m, T, NOISE, s) for m, s in itertools.product(models, range(n_seeds))]
    batches = [tasks[i:i + args.batch_size]
               for i in range(0, len(tasks), args.batch_size)]

    print(f"[{METHOD_TAG}] {len(tasks)} evals, workers={args.workers}",
          flush=True)

    all_rows = []
    t_start = time.time()
    partial_csv = OUT_CSV.replace('.csv', '_partial.csv') if not args.smoke else None
    ctx = mp.get_context('fork')
    with ctx.Pool(args.workers) as pool:
        for batch_rows in pool.imap_unordered(_run_batch, batches):
            all_rows.extend(batch_rows)
            if partial_csv and len(all_rows) >= CHECKPOINT_EVERY \
                    and (len(all_rows) % CHECKPOINT_EVERY < args.batch_size):
                pd.DataFrame(all_rows).to_csv(partial_csv, index=False)
                elapsed = time.time() - t_start
                print(f"  {len(all_rows)}/{len(tasks)} done ({elapsed:.0f}s)",
                      flush=True)

    df = pd.DataFrame(all_rows)
    out_csv = OUT_CSV if not args.smoke else OUT_CSV.replace('.csv', '_smoke.csv')
    df.to_csv(out_csv, index=False)
    print(f"[{METHOD_TAG}] SAVED -> {out_csv}")
    print(f"  N={len(df)}, mean directed_CS={df['directed_CS'].mean():.4f}")


if __name__ == '__main__':
    main()
