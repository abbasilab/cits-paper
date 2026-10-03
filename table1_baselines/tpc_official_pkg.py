"""TPC baseline with the authors' released package (timeawarepc 2.0.2, `cfc_tpc`; R pcalg via rpy2),
replacing the in-repo reimplementation for the small-scale benchmarks (Fig 2 A/B, Table 1).

Settings: maxdelay = 1, alpha = 0.05, isgauss = True (Fisher-z, as in the benchmark), no bootstrap.
The package's own magnitude pruning (|w| <= max|w|/10 removed) is kept, as published.
Scoring is identical to the existing scripts:
  autoregressive paradigms: directed TPR/FPR/CS (strict), self-edges removed (_fig2_baselines_noise.py)
  spiking networks: full-graph CS including self-edges (_spiking_pmatched_cpu.py)
Run in the environment that has timeawarepc + R pcalg (e.g. conda env `timeawarepc_test`):
  N_WORKERS=48 python tpc_official_pkg.py
Outputs (<OUT>/table1_baselines/): tpc_official_fig2.csv, tpc_official_spiking.csv
"""
import os, sys, csv
import numpy as np
from multiprocessing import get_context

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', 'shared')))
from paths import out as _out

NOISE = [0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]
REGIMES = ['lingauss1', 'lingauss2', 'nonlinnongauss1', 'nonlinnongauss2']
NS = int(os.environ.get('NS', '50'))
MOTIFS = {'convergence': [(0, 2), (1, 2), (2, 3)], 'diamond': [(0, 1), (0, 2), (1, 3), (2, 3)],
          'chain': [(0, 1), (1, 2), (2, 3)], 'depression': [(0, 2), (1, 2), (2, 3)]}
T_SPK, P = 2000, 4


def sim_spiking(seed, motif, SH):          # verbatim from _spiking_pmatched_cpu.py
    rng = np.random.default_rng(seed); W = np.zeros((P, P))
    for a, b in MOTIFS[motif]: W[a, b] = 1.4
    b_i = np.log(0.25); k1, k2 = 1.0, 0.4; s = np.zeros((P, T_SPK)); R = np.ones(P); U, tr, dep = 0.7, 15.0, 2.2
    for t in range(1, T_SPK):
        p1 = s[:, t - 1]; p2 = s[:, t - 2] if t >= 2 else np.zeros(P)
        if motif == 'depression':
            drive = dep * (W.T @ (U * R * p1)); R = R + (1 - R) / tr; R = np.clip(R - U * R * p1, 0, 1)
        else:
            drive = W.T @ (k1 * p1 + k2 * p2)
        eta = b_i + drive + SH * p1; s[:, t] = rng.poisson(np.clip(np.exp(eta), 0, 4.0))
    GTc = (W != 0).astype(int)
    GTs = np.eye(P, dtype=int) if SH != 0 else np.zeros((P, P), dtype=int)
    return s.astype(float), GTc, GTs


def cs_full(pred, GT):                      # verbatim scoring from _spiking_pmatched_cpu.py
    TP = FP = 0; nt = int(GT.sum()); nn = P * P - nt
    for i in range(P):
        for j in range(P):
            if GT[i, j]: TP += int(pred[i, j] > 0)
            else: FP += int(pred[i, j] > 0)
    return (TP / nt if nt else 0) - (FP / nn if nn else 0)


def tpc_pkg(X):
    from timeawarepc.tpc import cfc_tpc
    adj, _w = cfc_tpc(X.T, maxdelay=1, alpha=0.05, isgauss=True)
    return (np.asarray(adj) != 0).astype(int)


def task(t):
    kind = t[0]
    try:
        if kind == 'ar':
            _, reg, nz, s = t
            from sim_scm import simulate_extended
            from directed_metrics import compute_directed_metrics
            o = simulate_extended(reg, nz, 1000, s); X = o[0].astype(np.float64)
            A = tpc_pkg(X); np.fill_diagonal(A, 0)
            _, gl_uw, gl_w, gc_uw, gc_w, gb_uw, gb_lw, gb_cw = o
            m = compute_directed_metrics(A, gl_w, gc_w, gb_lw, gb_cw, gl_uw, gc_uw, gb_uw)
            return ('ar', ['TPC', reg, nz, s, m['directed_TPR_strict'], m['directed_FPR_strict'], m['directed_CS_strict']])
        _, block, motif, s = t
        SH = 0.0 if block == 'control' else -1.5
        X, GTc, GTs = sim_spiking(s, motif, SH)
        A = tpc_pkg(X)
        return ('spk', [block, motif, s, cs_full(A, ((GTc + GTs) > 0).astype(int)),
                        float(np.mean([A[i, i] > 0 for i in range(P)]))])
    except Exception as e:                  # report, never silently drop
        return ('err', [repr(t), repr(e)])


if __name__ == '__main__':
    tasks = [('ar', r, nz, s) for r in REGIMES for nz in NOISE for s in range(NS)]
    tasks += [('spk', 'control', m, s) for m in ['convergence', 'diamond', 'chain'] for s in range(NS)]
    tasks += [('spk', 'recurrent', m, s) for m in ['convergence', 'diamond', 'chain', 'depression'] for s in range(NS)]
    nw = int(os.environ.get('N_WORKERS', '48'))
    ar, spk, err = [], [], []
    with get_context('spawn').Pool(nw) as pool:
        for k, (kind, row) in enumerate(pool.imap_unordered(task, tasks, chunksize=4)):
            {'ar': ar, 'spk': spk, 'err': err}[kind].append(row)
            if (k + 1) % 200 == 0:
                print(f'{k + 1}/{len(tasks)} done, errors {len(err)}', flush=True)
    with open(_out('table1_baselines', 'tpc_official_fig2.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['method', 'regime', 'noise', 'seed', 'TPR', 'FPR', 'CS']); w.writerows(sorted(ar, key=lambda r: (r[1], r[2], r[3])))
    with open(_out('table1_baselines', 'tpc_official_spiking.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['block', 'cell', 'seed', 'cs', 'self_frac']); w.writerows(sorted(spk))
    for e in err: print('ERROR', e)
    import pandas as pd
    a = pd.DataFrame(ar, columns=['method', 'regime', 'noise', 'seed', 'TPR', 'FPR', 'CS'])
    print(a[a.noise == 1].groupby('regime')['CS'].agg(['mean', 'std']).round(3))
    b = pd.DataFrame(spk, columns=['block', 'cell', 'seed', 'cs', 'self_frac'])
    print(b.groupby(['block', 'cell'])[['cs', 'self_frac']].agg(['mean', 'std']).round(3))
    print('TPC OFFICIAL DONE', flush=True)
