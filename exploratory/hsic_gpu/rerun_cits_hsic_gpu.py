"""
rerun_cits_hsic_gpu.py

Reproduce the paper's exact-HSIC CITS benchmark using the validated full-GPU
conditional-independence test (gpu_cits_ci.hsic_ci_gpu) in place of kpcalg.

The CITS algorithm is UNCHANGED: we run cits.methods.cits_full exactly as the
paper does, only monkey-patching cits.methods.hsic_condind so the CI test runs
on GPU (residualize-on-S + full-kernel HSIC permutation test) instead of R's
regrXonS + kpcalg::hsic.perm. Both halves were validated against kpcalg
(statistic to 0.01%; conditional decisions 12/12) before this run.

Regimes: nonlinnongauss1, nonlinnongauss2, ctrnn.  Paper CS: 0.994 / 1.000 / 0.778.
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
OUT_CSV = os.path.join(OUT_DIR, 'simulation_results_directed_cits_hsic_gpu.csv')

ALPHA, T, NOISE, TAU, P_PERM = 0.05, 1000, 1.0, 1, 100
PAPER_CS = {'nonlinnongauss1': 0.994, 'nonlinnongauss2': 1.000, 'ctrnn': 0.778}


def _simulate(regime: str, seed: int):
    rng = np.random.default_rng(seed); eta = NOISE
    if regime == 'nonlinnongauss1':
        X = np.zeros((4, T))
        for t in range(1, T):
            e = rng.uniform(0, eta, size=4)
            X[0, t], X[1, t] = e[0], e[1]
            X[2, t] = 4*np.sin(X[0, t-1]) - 3*np.sin(X[1, t-1]) + e[2]
            X[3, t] = 3*X[2, t-1] + e[3]
        gt_uw = np.array([[0,0,1,0],[0,0,1,0],[0,0,0,1],[0,0,0,0]])
    elif regime == 'nonlinnongauss2':
        X = np.zeros((4, T))
        for t in range(1, T):
            e = rng.uniform(0, eta, size=4)
            X[0, t] = e[0]; X[1, t] = 4*X[0, t-1] + e[1]
            X[2, t] = 3*np.sin(X[0, t-1]) + e[2]
            X[3, t] = 8*np.log(np.abs(X[1, t-1])+1e-8) + 9*np.log(np.abs(X[2, t-1])+1e-8) + e[3]
        gt_uw = np.array([[0,1,1,0],[0,0,0,1],[0,0,0,1],[0,0,0,0]])
    elif regime == 'ctrnn':
        n = 4; tau_j = 10.0; g = float(np.exp(1))
        w = np.zeros((n, n)); w[0,2]=w[1,2]=w[2,3]=10.0
        X = np.zeros((n, T)); X[:, 0] = rng.normal(1, eta, size=n)
        for t in range(1, T):
            e = rng.normal(1, eta, size=n); sig = np.tanh(X[:, t-1])
            X[:, t] = X[:, t-1] + g*((-X[:, t-1] + w.T @ sig)/tau_j + e)
        gt_uw = np.array([[0,0,1,0],[0,0,1,0],[0,0,0,1],[0,0,0,0]])
    else:
        raise ValueError(regime)
    return X.astype(np.float64), gt_uw, gt_uw.astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=50)
    ap.add_argument('--regimes', nargs='+',
                    default=['nonlinnongauss1', 'nonlinnongauss2', 'ctrnn'])
    args = ap.parse_args()

    import torch
    from cits import methods as cm
    from directed_metrics import compute_directed_metrics
    import gpu_cits_ci

    assert torch.cuda.is_available(), "CUDA required"
    gen = torch.Generator(device='cuda'); gen.manual_seed(20260706)

    def hsic_condind_gpu(A, B, S, data, reps=None):
        # cits passes `data` as (p_vars x n_samples); kpcalg used data.T (samples x vars).
        samples_x_vars = np.asarray(data, dtype=np.float64).T
        S_list = sorted(int(s) for s in S)
        return gpu_cits_ci.hsic_ci_gpu(samples_x_vars, int(A), int(B), S_list,
                                       sig=1.0, p=P_PERM, generator=gen)
    cm.hsic_condind = hsic_condind_gpu

    rows, t_start = [], time.time()
    for regime in args.regimes:
        for seed in range(args.seeds):
            X, gt_uw, gt_w = _simulate(regime, seed)
            t0 = time.perf_counter()
            adj = cm.cits_full(X, TAU, ALPHA, cond_dep='cond_dep_hsic')
            pred = (np.asarray(adj) != 0).astype(int); np.fill_diagonal(pred, 0)
            elapsed = time.perf_counter() - t0
            z_uw, z_w = np.zeros_like(gt_uw), np.zeros_like(gt_w)
            m = compute_directed_metrics(pred, gt_w, z_w, z_w, z_w, gt_uw, z_uw, z_uw)
            rows.append({'model': regime, 'seed': seed, 'T': T, 'noise': NOISE,
                         'method': 'CITS_HSIC_GPU', 'device': 'cuda',
                         'runtime_sec': round(elapsed, 3),
                         'directed_CS': m['directed_CS_strict'],
                         'directed_TPR': m['directed_TPR_strict'],
                         'directed_FPR': m['directed_FPR_strict'],
                         'n_true_directed': m['n_true_directed'],
                         'n_neg_directed': m['n_neg_directed']})
            print(f"[{regime} seed={seed:02d}] CS={m['directed_CS_strict']:.3f} "
                  f"({elapsed:.1f}s, total {time.time()-t_start:.0f}s)", flush=True)
            if (seed + 1) % 10 == 0:
                pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    df = pd.DataFrame(rows); df.to_csv(OUT_CSV, index=False)
    print(f"\nSAVED -> {OUT_CSV}\n")
    summ = df.groupby('model')['directed_CS'].agg(['mean', 'sem', 'count']).round(3)
    print("Mean CS by regime (GPU) vs paper:")
    for regime in args.regimes:
        if regime in summ.index:
            gm = summ.loc[regime, 'mean']
            print(f"  {regime:<18} GPU={gm:.3f}  paper={PAPER_CS[regime]:.3f}  "
                  f"Δ={gm-PAPER_CS[regime]:+.3f}")


if __name__ == '__main__':
    main()
