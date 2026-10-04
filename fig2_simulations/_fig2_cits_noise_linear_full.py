"""CITS on the linear-Gaussian paradigms with the exhaustive conditioning-set search the paper
specifies for small graphs (cits.methods.cits_full, partial correlation, tau=1, alpha=0.05),
all noise levels, 50 seeds. Output rows in the Fig 2 CSV format."""
import sys, os, csv, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared')))
from paths import out as _out
NOISE = [0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]
def run(a):
    reg, nz, s = a
    from sim_scm import simulate_extended
    from directed_metrics import compute_directed_metrics
    from cits import methods
    o = simulate_extended(reg, nz, 1000, s); X = o[0].astype(float)
    B = (np.asarray(methods.cits_full(X, 1, 0.05)) != 0).astype(int); np.fill_diagonal(B, 0)
    _, gl_uw, gl_w, gc_uw, gc_w, gb_uw, gb_lw, gb_cw = o
    m = compute_directed_metrics(B, gl_w, gc_w, gb_lw, gb_cw, gl_uw, gc_uw, gb_uw)
    return ['CITS', reg, nz, s, m['directed_TPR_strict'], m['directed_FPR_strict'], m['directed_CS_strict']]
if __name__ == '__main__':
    tasks = [(r, nz, s) for r in ['lingauss1', 'lingauss2'] for nz in NOISE for s in range(50)]
    with Pool(64) as p:
        rows = p.map(run, tasks, chunksize=4)
    with open(_out('fig2_simulations', '_fig2_cits_noise50.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['method', 'regime', 'noise', 'seed', 'TPR', 'FPR', 'CS']); w.writerows(rows)
    import pandas as pd
    d = pd.read_csv(_out('fig2_simulations', '_fig2_cits_noise50.csv'))
    print(d.groupby(['regime', 'noise']).CS.mean().unstack('noise').round(3))
