"""PCMCI+ and LPCMCI with tigramite's non-parametric GPDC test (Gaussian-process regression plus
distance correlation, analytic null) on the non-Gaussian benchmarks: the two non-linear
autoregressive paradigms (all noise levels) and the spiking networks. tau_max = 2,
pc_alpha = 0.05 as elsewhere. (CMIknn with a shuffle test took > 10 min per dataset and is not
practical at this scale.) Each graph is scored two ways from the same run:
  native : lagged links plus same-time links (unoriented same-time links count in both directions),
           as in _glm_suite_baselines._graph_to_adj (main Table 1 convention)
  lagged : lagged links only (tau >= 1), as in pcmci_lagged.py (sensitivity analysis)
Spiking networks are skipped by default (GPDC_SPIKING=1 to include): GPDC's Gaussian-process fits on
2,000-bin spike trains did not finish a single dataset in > 1.5 h, so the spiking benchmark keeps ParCorr.
Requires the `dcor` package for GPDC.  Usage: N_WORKERS=96 python pcmci_gpdc.py
Outputs (<OUT>/table1_baselines/): pcmci_gpdc_fig2.csv, pcmci_gpdc_spiking.csv (column 'scoring')
"""
import os, sys, csv
import numpy as np
from multiprocessing import get_context

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', 'shared')))
sys.path.insert(0, HERE)
from paths import out as _out
from tpc_official_pkg import sim_spiking, cs_full, NOISE, P
from pcmci_lagged import lagged_adj

TAU_MAX, ALPHA = 2, 0.05
NS = int(os.environ.get('NS', '50'))


def native_adj(graph, keep_self):          # same logic as _glm_suite_baselines._graph_to_adj
    A = lagged_adj(graph, keep_self)
    for i in range(P):
        for j in range(P):
            if i == j:
                continue
            mk = graph[i, j, 0]
            if mk in ('-->', 'o->'): A[i, j] = 1
            elif mk in ('<--', '<-o'): A[j, i] = 1
            elif mk in ('o-o', '<->'): A[i, j] = A[j, i] = 1
    return A


def run(method, X):
    from tigramite import data_processing as pp
    from tigramite.independence_tests.gpdc import GPDC
    df = pp.DataFrame(X.T.astype(np.float64))
    if method == 'PCMCIplus':
        from tigramite.pcmci import PCMCI
        return PCMCI(dataframe=df, cond_ind_test=GPDC(significance='analytic'), verbosity=0).run_pcmciplus(tau_max=TAU_MAX, pc_alpha=ALPHA)['graph']
    from tigramite.lpcmci import LPCMCI
    return LPCMCI(dataframe=df, cond_ind_test=GPDC(significance='analytic'), verbosity=0).run_lpcmci(tau_max=TAU_MAX, pc_alpha=ALPHA)['graph']


def task(t):
    try:
        out = []
        if t[0] == 'ar':
            _, method, reg, nz, s = t
            from sim_scm import simulate_extended
            from directed_metrics import compute_directed_metrics
            o = simulate_extended(reg, nz, 1000, s); g = run(method, o[0].astype(np.float64))
            _, gl_uw, gl_w, gc_uw, gc_w, gb_uw, gb_lw, gb_cw = o
            for sc, A in (('native', native_adj(g, False)), ('lagged', lagged_adj(g, False))):
                m = compute_directed_metrics(A, gl_w, gc_w, gb_lw, gb_cw, gl_uw, gc_uw, gb_uw)
                out.append(('ar', [method, sc, reg, nz, s, m['directed_TPR_strict'], m['directed_FPR_strict'], m['directed_CS_strict']]))
            return out
        _, method, block, motif, s = t
        X, GTc, GTs = sim_spiking(s, motif, 0.0 if block == 'control' else -1.5)
        g = run(method, X); GT = ((GTc + GTs) > 0).astype(int)
        for sc, A in (('native', native_adj(g, True)), ('lagged', lagged_adj(g, True))):
            out.append(('spk', [method, sc, block, motif, s, cs_full(A, GT), float(np.mean([A[i, i] > 0 for i in range(P)]))]))
        return out
    except Exception as e:
        return [('err', [repr(t), repr(e)])]


if __name__ == '__main__':
    tasks = []
    for method in ('PCMCIplus', 'LPCMCI'):
        tasks += [('ar', method, r, nz, s) for r in ['nonlinnongauss1', 'nonlinnongauss2'] for nz in NOISE for s in range(NS)]
        if os.environ.get('GPDC_SPIKING', '0') == '1':      # off by default: GPDC's GP fits on 2,000-bin
            tasks += [('spk', method, 'control', m, s) for m in ['convergence', 'diamond', 'chain'] for s in range(NS)]
            tasks += [('spk', method, 'recurrent', m, s) for m in ['convergence', 'diamond', 'chain', 'depression'] for s in range(NS)]
    ar, spk, err = [], [], []
    with get_context('spawn').Pool(int(os.environ.get('N_WORKERS', '96'))) as pool:
        for k, res in enumerate(pool.imap_unordered(task, tasks, chunksize=2)):
            for kind, row in res:
                {'ar': ar, 'spk': spk, 'err': err}[kind].append(row)
            if (k + 1) % 200 == 0:
                print(f'{k + 1}/{len(tasks)} done, errors {len(err)}', flush=True)
    with open(_out('table1_baselines', 'pcmci_gpdc_fig2.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['method', 'scoring', 'regime', 'noise', 'seed', 'TPR', 'FPR', 'CS']); w.writerows(sorted(ar, key=lambda r: tuple(map(str, r[:5]))))
    with open(_out('table1_baselines', 'pcmci_gpdc_spiking.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['method', 'scoring', 'block', 'cell', 'seed', 'cs', 'self_frac']); w.writerows(sorted(spk, key=lambda r: tuple(map(str, r[:5]))))
    for e in err: print('ERROR', e)
    print(f'GPDC DONE: {len(ar)} AR rows, {len(spk)} spiking rows, {len(err)} errors', flush=True)
