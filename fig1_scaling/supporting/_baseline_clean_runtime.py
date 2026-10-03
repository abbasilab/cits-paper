"""CLEAN baseline runtime-vs-p sweep. Run standalone, ONE method/config at a time,
nothing else of ours running, default (multi-threaded) BLAS so each baseline gets
the full machine -- its best case. Finds the TRUE feasibility wall per method.

The previous _baseline_wall_budget sweep was contaminated by a concurrent 3-hour
GPU job (evidence: PCMCI+ p=100 falsely 'walled' at >3600s but runs in 111s
standalone; LPCMCI p=25 swung 28s<->2197s across seeds). This redo fixes that.

N=500 (T=2000): enough samples for CS>=0.95 across p (N* curve). 1-hour cap/run.
"""
import os, sys, time
import multiprocessing as mp
_THIS = os.path.dirname(os.path.abspath(__file__))
# _THIS, the parent fig1_scaling/ (scaling_benchmark_lg) and the repo's shared/
for _p in (_THIS, os.path.dirname(_THIS), os.path.join(os.path.dirname(os.path.dirname(_THIS)), 'shared')):
    if _p not in sys.path: sys.path.insert(0, _p)
import numpy as np, pandas as pd
from scaling_benchmark_lg import lg_var, directed_cs, _worker

TAU = 1
BUDGET = 3600
N = 500
T = N * (2*(TAU+1))       # 2000
P_VALUES = [50, 100, 250, 500, 1000]
METHODS = ['PCMCI+', 'TPC', 'KernelGC', 'LPCMCI']
from paths import out as _out
OUT = _out('fig1_scaling/supporting', 'baseline_clean_runtime.csv')


def run_timed(method, X):
    ctx = mp.get_context('fork')
    q = ctx.Queue()
    pr = ctx.Process(target=_worker, args=(method, X, q))
    t0 = time.perf_counter()
    pr.start(); pr.join(BUDGET)
    if pr.is_alive():
        pr.terminate(); pr.join(); return ('timeout', BUDGET, None)
    if q.empty():
        return ('err', time.perf_counter()-t0, None)
    st, rt, A = q.get()
    return (st, rt, A)


rows = []
print(f"=== CLEAN baseline runtime, N={N} (T={T}), 1-hour cap, standalone ===", flush=True)
for method in METHODS:
    for p in P_VALUES:
        X, GT = lg_var(p, T, 0)
        st, rt, A = run_timed(method, X)
        cs = directed_cs(A, GT) if st == 'ok' else float('nan')
        rows.append({'method': method, 'p': p, 'N': N, 'status': st,
                     'runtime_sec': round(rt, 1), 'cs': round(cs, 3) if cs == cs else None})
        pd.DataFrame(rows).to_csv(OUT, index=False)
        tag = f"{rt:7.1f}s  CS={cs:.3f}" if st == 'ok' else f"{st.upper()} (>{BUDGET}s)"
        print(f"  {method:9s} p={p:4d}: {tag}", flush=True)
        if st == 'timeout':
            print(f"  -> {method} walls at p={p} (N={N}); higher p skipped", flush=True)
            break
print("DONE", flush=True)
