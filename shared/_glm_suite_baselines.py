"""
_glm_suite_baselines.py  (tigra env)

PCMCI+ and LPCMCI on the GLM-spiking motifs, scored with the same directed CS.
ParCorr on all motifs; additionally PCMCI+ with a nonlinear CI test (CMIknn) on
the coincidence motif, so the baselines get a fair nonparametric shot where the
coupling is nonlinear.

Run: <tigra env python> _glm_suite_baselines.py --seeds 10
"""
from __future__ import annotations
import argparse, warnings, os, sys
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glm_spiking_sim import simulate_glm_spiking, directed_cs, MOTIFS

TAU_MAX, ALPHA, P = 2, 0.05, 4


def _graph_to_adj(graph, p, keep_self=False):
    adj = np.zeros((p, p), dtype=int)
    for tau in range(1, graph.shape[2]):
        for i in range(p):
            for j in range(p):
                if i == j and not keep_self:
                    continue
                if graph[i, j, tau] != '':
                    adj[i, j] = 1
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            mk = graph[i, j, 0]
            if mk in ('-->', 'o->'): adj[i, j] = 1
            elif mk in ('<--', '<-o'): adj[j, i] = 1
            elif mk in ('o-o', '<->'): adj[i, j] = adj[j, i] = 1
    return adj


def pcmci_plus(X, ci='parcorr'):
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    if ci == 'parcorr':
        from tigramite.independence_tests.parcorr import ParCorr; c = ParCorr()
    else:
        from tigramite.independence_tests.cmiknn import CMIknn
        c = CMIknn(significance='shuffle_test')
    df = pp.DataFrame(X.T.astype(np.float64))
    g = PCMCI(dataframe=df, cond_ind_test=c, verbosity=0).run_pcmciplus(
        tau_max=TAU_MAX, pc_alpha=ALPHA)['graph']
    return _graph_to_adj(g, P)


def lpcmci(X):
    from tigramite import data_processing as pp
    from tigramite.lpcmci import LPCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    df = pp.DataFrame(X.T.astype(np.float64))
    g = LPCMCI(dataframe=df, cond_ind_test=ParCorr(), verbosity=0).run_lpcmci(
        tau_max=TAU_MAX, pc_alpha=ALPHA)['graph']
    return _graph_to_adj(g, P)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', type=int, default=10)
    args = ap.parse_args()
    print(f"{'motif':13s} {'PCMCI+(parc)':>13s} {'LPCMCI':>8s} {'PCMCI+(cmiknn)':>15s}")
    for m in MOTIFS:
        pp_, lp_, ck_ = [], [], []
        for s in range(args.seeds):
            X, GT, _ = simulate_glm_spiking(s, motif=m)
            try: pp_.append(directed_cs(pcmci_plus(X, 'parcorr'), GT)[0])
            except Exception as e: pp_.append(np.nan)
            try: lp_.append(directed_cs(lpcmci(X), GT)[0])
            except Exception as e: lp_.append(np.nan)
            if m == 'coincidence':
                try: ck_.append(directed_cs(pcmci_plus(X, 'cmiknn'), GT)[0])
                except Exception as e: ck_.append(np.nan)
        ck = f"{np.nanmean(ck_):.3f}" if ck_ else "  --"
        print(f"{m:13s} {np.nanmean(pp_):>13.3f} {np.nanmean(lp_):>8.3f} {ck:>15s}", flush=True)


if __name__ == '__main__':
    main()
