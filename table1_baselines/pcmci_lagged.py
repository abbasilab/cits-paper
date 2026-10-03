"""PCMCI+ and LPCMCI (tigramite, ParCorr, tau_max = 2, pc_alpha = 0.05; settings as in
_glm_suite_baselines.py) scored on LAGGED links only (tau >= 1), for the small-scale benchmarks
(Fig 2 A/B, Table 1). This matches CITS, which reports lagged edges only, and the simulated ground
truth, which has no same-time edges. Any mark at lag tau >= 1 counts as an edge from source to
target (time order fixes the direction). Self-lags are dropped for the autoregressive paradigms and
kept for the spiking networks, as in the existing scripts. One run gives both Fig 2 and Table 1.

Run in the tigramite environment:  N_WORKERS=48 python pcmci_lagged.py
Outputs (<OUT>/table1_baselines/): pcmci_lagged_fig2.csv, pcmci_lagged_spiking.csv
"""
import os, sys, csv
import numpy as np
from multiprocessing import get_context

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', 'shared')))
sys.path.insert(0, HERE)
from paths import out as _out
from tpc_official_pkg import sim_spiking, cs_full, NOISE, REGIMES, P

TAU_MAX, ALPHA = 2, 0.05
NS = int(os.environ.get('NS', '50'))


def lagged_adj(graph, keep_self):
    A = np.zeros((P, P), int)
    for tau in range(1, graph.shape[2]):
        for i in range(P):
            for j in range(P):
                if (i != j or keep_self) and graph[i, j, tau] != '':
                    A[i, j] = 1
    return A


def run(method, X):
    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    df = pp.DataFrame(X.T.astype(np.float64))
    if method == 'PCMCIplus':
        from tigramite.pcmci import PCMCI
        return PCMCI(dataframe=df, cond_ind_test=ParCorr(), verbosity=0).run_pcmciplus(tau_max=TAU_MAX, pc_alpha=ALPHA)['graph']
    from tigramite.lpcmci import LPCMCI
    return LPCMCI(dataframe=df, cond_ind_test=ParCorr(), verbosity=0).run_lpcmci(tau_max=TAU_MAX, pc_alpha=ALPHA)['graph']


def task(t):
    try:
        if t[0] == 'ar':
            _, method, reg, nz, s = t
            from sim_scm import simulate_extended
            from directed_metrics import compute_directed_metrics
            o = simulate_extended(reg, nz, 1000, s)
            A = lagged_adj(run(method, o[0].astype(np.float64)), keep_self=False)
            _, gl_uw, gl_w, gc_uw, gc_w, gb_uw, gb_lw, gb_cw = o
            m = compute_directed_metrics(A, gl_w, gc_w, gb_lw, gb_cw, gl_uw, gc_uw, gb_uw)
            return ('ar', [method, reg, nz, s, m['directed_TPR_strict'], m['directed_FPR_strict'], m['directed_CS_strict']])
        _, method, block, motif, s = t
        X, GTc, GTs = sim_spiking(s, motif, 0.0 if block == 'control' else -1.5)
        A = lagged_adj(run(method, X), keep_self=True)
        return ('spk', [method, block, motif, s, cs_full(A, ((GTc + GTs) > 0).astype(int)),
                        float(np.mean([A[i, i] > 0 for i in range(P)]))])
    except Exception as e:
        return ('err', [repr(t), repr(e)])


if __name__ == '__main__':
    tasks = []
    for method in ('PCMCIplus', 'LPCMCI'):
        tasks += [('ar', method, r, nz, s) for r in REGIMES for nz in NOISE for s in range(NS)]
        tasks += [('spk', method, 'control', m, s) for m in ['convergence', 'diamond', 'chain'] for s in range(NS)]
        tasks += [('spk', method, 'recurrent', m, s) for m in ['convergence', 'diamond', 'chain', 'depression'] for s in range(NS)]
    ar, spk, err = [], [], []
    with get_context('spawn').Pool(int(os.environ.get('N_WORKERS', '48'))) as pool:
        for k, (kind, row) in enumerate(pool.imap_unordered(task, tasks, chunksize=4)):
            {'ar': ar, 'spk': spk, 'err': err}[kind].append(row)
            if (k + 1) % 500 == 0:
                print(f'{k + 1}/{len(tasks)} done, errors {len(err)}', flush=True)
    with open(_out('table1_baselines', 'pcmci_lagged_fig2.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['method', 'regime', 'noise', 'seed', 'TPR', 'FPR', 'CS']); w.writerows(sorted(ar, key=lambda r: (r[0], r[1], r[2], r[3])))
    with open(_out('table1_baselines', 'pcmci_lagged_spiking.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['method', 'block', 'cell', 'seed', 'cs', 'self_frac']); w.writerows(sorted(spk))
    for e in err: print('ERROR', e)
    try:
        import pandas as pd
    except ImportError:
        print("PCMCI LAGGED DONE (install pandas for the summary)", flush=True); sys.exit(0)
    a = pd.DataFrame(ar, columns=['method', 'regime', 'noise', 'seed', 'TPR', 'FPR', 'CS'])
    print(a[a.noise == 1].groupby(['method', 'regime'])[['TPR', 'FPR', 'CS']].mean().round(3))
    b = pd.DataFrame(spk, columns=['method', 'block', 'cell', 'seed', 'cs', 'self_frac'])
    print(b.groupby(['method', 'block', 'cell'])[['cs', 'self_frac']].mean().round(3))
    print('PCMCI LAGGED DONE', flush=True)
