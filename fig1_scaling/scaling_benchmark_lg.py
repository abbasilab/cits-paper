"""
scaling_benchmark_lg.py

Rigorous scaling benchmark (standard time-series causal-discovery design):
random sparse linear-Gaussian DAGs, sweep number of variables p, report BOTH
runtime and accuracy (directed Combined Score) for every method on identical
data. This is the recoverable regime the field uses for scalability (unlike
CTRNN, where no method recovers the graph and autocorrelation inflates
baseline runtimes).

Data: VAR(1), Erdos-Renyi DAG, average degree ~2, Gaussian noise.
Methods: CITS-GPU (cuPC/Fisher-z), PCMCI+, LPCMCI, TPC, KernelGC.
Phase 1 = CPU baselines (fork + timeout); Phase 2 = CITS-GPU (main proc, CUDA).

Output: scaling_benchmark_lg.csv  (p, seed, method, status, runtime_sec, cs)
"""
from __future__ import annotations
import os, sys, time, warnings
import multiprocessing as mp
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = '/home/rbiswas1/repos/cits'
for _p in (_THIS, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT = os.path.join(_THIS, 'scaling_benchmark_lg.csv')
ALPHA, T, TAU = 0.05, 1000, 1
P_VALUES = [10, 25, 50, 100]
SEEDS = 3
TIMEOUT = 300          # clean solo run


def lg_var(p, T, seed, deg=2):
    """Sparse linear-Gaussian VAR(1). GT[a,b]=1 iff a->b (a at t-1 drives b at t)."""
    rng = np.random.default_rng(seed)
    GT = np.zeros((p, p), int); W = np.zeros((p, p))
    for _ in range(int(deg * p)):
        a, b = int(rng.integers(0, p)), int(rng.integers(0, p))
        if a != b:
            GT[a, b] = 1
            W[b, a] = rng.uniform(0.3, 0.6) * rng.choice([-1, 1])
    ev = max(abs(np.linalg.eigvals(W)))
    if ev > 0.95:
        W = W * (0.95 / ev)
    X = np.zeros((p, T))
    for t in range(1, T):
        X[:, t] = W @ X[:, t - 1] + rng.standard_normal(p)
    return X.astype(np.float64), GT


def directed_cs(pred, GT):
    if pred is None:
        return float('nan')
    pred = (np.asarray(pred) != 0).astype(int); GT = (np.asarray(GT) != 0).astype(int)
    m = ~np.eye(GT.shape[0], dtype=bool)
    tp = ((pred == 1) & (GT == 1) & m).sum(); fn = ((pred == 0) & (GT == 1) & m).sum()
    fp = ((pred == 1) & (GT == 0) & m).sum(); tn = ((pred == 0) & (GT == 0) & m).sum()
    return float(tp / max(tp + fn, 1) - fp / max(fp + tn, 1))


def _worker(method, X, q):
    try:
        from simulation_benchmark_fc_methods_v3 import run_tpc
        from pcmci_plus_baseline import run_pcmci_plus
        from lpcmci_baseline import run_lpcmci
        from kernel_granger_baseline import run_kernel_granger
        t0 = time.perf_counter()
        if method == 'TPC':        A = run_tpc(X, alpha=ALPHA)
        elif method == 'PCMCI+':   A = run_pcmci_plus(X, tau_max=TAU, pc_alpha=ALPHA)
        elif method == 'LPCMCI':   A = run_lpcmci(X, tau_max=TAU, pc_alpha=ALPHA)
        elif method == 'KernelGC': A = run_kernel_granger(X, max_lag=TAU, alpha=ALPHA)
        q.put(('ok', round(time.perf_counter() - t0, 2), np.asarray(A)))
    except Exception as exc:
        q.put(('err', str(exc)[:120], None))


def run_baseline(method, X, ctx):
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(method, X, q))
    p.start(); p.join(TIMEOUT)
    if p.is_alive():
        p.terminate(); p.join()
        return ('timeout', float(TIMEOUT), None)
    return q.get() if not q.empty() else ('err', 'no result', None)


def main():
    rows = []
    ctx = mp.get_context('fork')
    # Phase 1: CPU baselines (no CUDA in parent)
    for p in P_VALUES:
        for s in range(SEEDS):
            X, GT = lg_var(p, T, s)
            for m in ['TPC', 'PCMCI+', 'LPCMCI', 'KernelGC']:
                st, rt, A = run_baseline(m, X, ctx)
                cs = directed_cs(A, GT)
                rows.append({'p': p, 'seed': s, 'method': m, 'status': st,
                             'runtime_sec': rt, 'cs': round(cs, 3) if cs == cs else cs})
                print(f"p={p:3d} s={s} {m:9s} {st:8s} rt={rt} cs={cs:.3f}" if cs == cs
                      else f"p={p:3d} s={s} {m:9s} {st:8s} rt={rt} cs=nan", flush=True)
                pd.DataFrame(rows).to_csv(OUT, index=False)
    # Phase 2: CITS-GPU (main proc, CUDA)
    from gpu_cits_lag_cupc_faithful import gpu_cits_lag_cupc_faithful
    _ = gpu_cits_lag_cupc_faithful(lg_var(10, T, 0)[0], ALPHA, TAU)  # warm up
    for p in P_VALUES:
        for s in range(SEEDS):
            X, GT = lg_var(p, T, s)
            t0 = time.perf_counter()
            B = gpu_cits_lag_cupc_faithful(X, ALPHA, TAU)
            rt = round(time.perf_counter() - t0, 2)
            cs = directed_cs(B, GT)
            rows.append({'p': p, 'seed': s, 'method': 'CITS-GPU', 'status': 'ok',
                         'runtime_sec': rt, 'cs': round(cs, 3)})
            print(f"p={p:3d} s={s} {'CITS-GPU':9s} ok       rt={rt} cs={cs:.3f}", flush=True)
            pd.DataFrame(rows).to_csv(OUT, index=False)

    df = pd.DataFrame(rows)
    print("\n=== mean CS by p x method ===")
    print(df.pivot_table(index='p', columns='method', values='cs', aggfunc='mean').round(3))
    print("\n=== mean runtime (s) by p x method ===")
    print(df.pivot_table(index='p', columns='method', values='runtime_sec', aggfunc='mean').round(2))
    print("DONE")


if __name__ == '__main__':
    main()
