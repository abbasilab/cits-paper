"""
rerun_cits_hsic_gpu_fast.py

Same as rerun_cits_hsic_gpu.py but uses the optimized GPUCondIndTester
(gpu_cits_ci_fast) as the CI backend: per-seed cached spline bases +
Demmler-Reinsch eigen-GCV. Algorithm (cits_full) is unchanged.

Usage:
  python rerun_cits_hsic_gpu_fast.py --regimes nonlinnongauss2 --seeds 5
  # shard across GPUs with --seed-start / --seed-end and CUDA_VISIBLE_DEVICES
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np, pandas as pd

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO_CITS = '/home/rbiswas1/repos/cits'
for _p in (_THIS, _REPO_CITS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT_DIR = '/home/rbiswas1/microns/arousal_paper_overleaf/figures/2026-07-06_hsic_gpu_validation'
os.makedirs(OUT_DIR, exist_ok=True)

ALPHA, T, NOISE, TAU, P_PERM = 0.05, 1000, 1.0, 1, 100
PAPER_CS = {'nonlinnongauss1': 0.994, 'nonlinnongauss2': 1.000, 'ctrnn': 0.778}


def _simulate(regime, seed):
    rng = np.random.default_rng(seed); eta = NOISE
    if regime == 'nonlinnongauss1':
        X = np.zeros((4, T))
        for t in range(1, T):
            e = rng.uniform(0, eta, 4); X[0,t],X[1,t]=e[0],e[1]
            X[2,t]=4*np.sin(X[0,t-1])-3*np.sin(X[1,t-1])+e[2]; X[3,t]=3*X[2,t-1]+e[3]
        gt=np.array([[0,0,1,0],[0,0,1,0],[0,0,0,1],[0,0,0,0]])
    elif regime == 'nonlinnongauss2':
        X = np.zeros((4, T))
        for t in range(1, T):
            e=rng.uniform(0,eta,4); X[0,t]=e[0]; X[1,t]=4*X[0,t-1]+e[1]
            X[2,t]=3*np.sin(X[0,t-1])+e[2]
            X[3,t]=8*np.log(np.abs(X[1,t-1])+1e-8)+9*np.log(np.abs(X[2,t-1])+1e-8)+e[3]
        gt=np.array([[0,1,1,0],[0,0,0,1],[0,0,0,1],[0,0,0,0]])
    elif regime == 'ctrnn':
        n=4; tau_j=10.0; g=float(np.exp(1)); w=np.zeros((n,n)); w[0,2]=w[1,2]=w[2,3]=10.0
        X=np.zeros((n,T)); X[:,0]=rng.normal(1,eta,n)
        for t in range(1,T):
            e=rng.normal(1,eta,n); X[:,t]=X[:,t-1]+g*((-X[:,t-1]+w.T@np.tanh(X[:,t-1]))/tau_j+e)
        gt=np.array([[0,0,1,0],[0,0,1,0],[0,0,0,1],[0,0,0,0]])
    else:
        raise ValueError(regime)
    return X.astype(np.float64), gt, gt.astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--regimes', nargs='+', default=['nonlinnongauss1','nonlinnongauss2','ctrnn'])
    ap.add_argument('--seeds', type=int, default=50)
    ap.add_argument('--seed-start', type=int, default=0)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    out_csv = os.path.join(OUT_DIR,
        f"simulation_results_directed_cits_hsic_gpu_fast{('_'+args.tag) if args.tag else ''}.csv")

    import torch
    from cits import methods as cm
    from directed_metrics import compute_directed_metrics
    from gpu_cits_ci_fast import GPUCondIndTester
    assert torch.cuda.is_available()
    gen = torch.Generator(device='cuda'); gen.manual_seed(20260706)

    cache = {}
    def hsic_condind_fast(A, B, S, data, reps=None):
        key = id(data)
        t = cache.get(key)
        if t is None:
            cache.clear()
            t = GPUCondIndTester(np.asarray(data, dtype=np.float64).T,
                                 sig=1.0, p=P_PERM, generator=gen)
            cache[key] = t
        return t.pval(int(A), int(B), S)
    cm.hsic_condind = hsic_condind_fast

    rows, t0all = [], time.time()
    for regime in args.regimes:
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            X, gt_uw, gt_w = _simulate(regime, seed)
            t0 = time.perf_counter()
            adj = cm.cits_full(X, TAU, ALPHA, cond_dep='cond_dep_hsic')
            pred = (np.asarray(adj) != 0).astype(int); np.fill_diagonal(pred, 0)
            el = time.perf_counter() - t0
            z=np.zeros_like(gt_uw); zf=np.zeros_like(gt_w)
            m = compute_directed_metrics(pred, gt_w, zf, zf, zf, gt_uw, z, z)
            rows.append({'model':regime,'seed':seed,'method':'CITS_HSIC_GPU_fast',
                         'runtime_sec':round(el,2),'directed_CS':m['directed_CS_strict'],
                         'directed_TPR':m['directed_TPR_strict'],'directed_FPR':m['directed_FPR_strict']})
            print(f"[{regime} seed={seed:02d}] CS={m['directed_CS_strict']:.3f} "
                  f"({el:.1f}s, total {time.time()-t0all:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(out_csv, index=False)
    df = pd.DataFrame(rows)
    print(f"\nSAVED -> {out_csv}\nMean CS by regime:")
    s = df.groupby('model')['directed_CS'].agg(['mean','sem','count']).round(3)
    for r in args.regimes:
        sub = df[df.model==r]['directed_CS']
        if len(sub): print(f"  {r:<18} GPU={sub.mean():.3f}±{sub.sem():.3f}  paper={PAPER_CS[r]:.3f}")


if __name__ == '__main__':
    main()
