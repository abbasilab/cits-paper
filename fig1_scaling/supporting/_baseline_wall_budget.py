"""Baseline feasibility under a uniform per-run wall-clock budget (paper-grade).

For each baseline method, climb p; at each p climb N until mean CS>=0.95 (N* found)
or a run exceeds the BUDGET (the method's wall at that p). Once a method walls at
its minimal N for some p, it is walled and larger p is skipped. Records runtime so
we can also report actual wall-clock. One method at a time -> no CPU contention.

Reported per (method,p): N* (min N for CS>=0.95) if reached within budget; or
'plateau CS=x' if it completes but never reaches 0.95; or 'WALL >Bs' at the p where
it first exceeds the budget. CITS-GPU is judged by the SAME budget separately."""
import os, sys, time, warnings
import multiprocessing as mp
warnings.filterwarnings('ignore')
_THIS = os.path.dirname(os.path.abspath(__file__))
for _p in (_THIS, '/home/rbiswas1/repos/cits'):
    if _p not in sys.path: sys.path.insert(0, _p)
import numpy as np, pandas as pd
from scaling_benchmark_lg import lg_var, directed_cs, _worker

ALPHA, TAU = 0.05, 1
THRESH = 0.95
BUDGET = 3600            # 1 hour per run, uniform
SEEDS = 2
P_VALUES = [25, 50, 100, 250, 500]
N_GRID = [250, 500, 1000]
METHODS = ['PCMCI+', 'LPCMCI', 'TPC', 'KernelGC']
OUT = os.path.join(_THIS, 'baseline_wall_budget.csv')


def run_timed(method, X):
    ctx = mp.get_context('fork')
    q = ctx.Queue()
    pr = ctx.Process(target=_worker, args=(method, X, q))
    t0 = time.perf_counter()
    pr.start(); pr.join(BUDGET)
    if pr.is_alive():
        pr.terminate(); pr.join()
        return ('timeout', BUDGET, None)
    if q.empty():
        return ('err', time.perf_counter()-t0, None)
    st, rt, A = q.get()
    return (st, rt, A)


rows = []
print(f"=== baseline feasibility under {BUDGET}s ({BUDGET//60}-min) uniform budget ===", flush=True)
for method in METHODS:
    walled = False
    for p in P_VALUES:
        if walled:
            print(f"  {method:9s} p={p:4d}: SKIP (walled at smaller p)", flush=True)
            continue
        nstar = None; best = -1.0; wall = False
        for N in N_GRID:
            T = N * (2*(TAU+1))
            css, rts = [], []
            for s in range(SEEDS):
                X, GT = lg_var(p, T, s)
                st, rt, A = run_timed(method, X)
                rows.append({'method': method, 'p': p, 'N': N, 'seed': s,
                             'status': st, 'runtime_sec': round(rt, 1),
                             'cs': round(directed_cs(A, GT), 3) if st == 'ok' else None})
                pd.DataFrame(rows).to_csv(OUT, index=False)
                if st == 'timeout':
                    wall = True; break
                css.append(directed_cs(A, GT)); rts.append(rt)
            if wall:
                print(f"  {method:9s} p={p:4d} N={N:5d}: WALL (>{BUDGET}s)", flush=True)
                walled = True; break
            m = float(np.nanmean(css)); best = max(best, m)
            print(f"  {method:9s} p={p:4d} N={N:5d}: meanCS={m:.3f} meanRT={np.mean(rts):.0f}s", flush=True)
            if m >= THRESH:
                nstar = N; break
        if not wall:
            tag = f"N*={nstar}" if nstar else f"plateau(bestCS={best:.3f})"
            print(f"  -> {method} p={p}: {tag}", flush=True)
    print("", flush=True)

print("=== summary ===", flush=True)
df = pd.DataFrame(rows)
print(df.to_string(), flush=True)
print("DONE", flush=True)
