"""Kernel Granger causality with the authors' original code (Marinazzo, Pellicoro & Stramaglia 2008;
https://github.com/danielemarinazzo/KernelGrangerCausality, commit c043147), run unchanged in GNU
Octave, replacing the in-repo random-Fourier-feature variant for the small-scale benchmarks
(Fig 2 A/B, Table 1).

Settings (pre-specified, the authors' recommendation for non-linear data): polynomial kernel
('p') of order 2, model order m = 2 (the lag used by the benchmark's kernel Granger), the
toolbox's built-in Bonferroni-corrected significance filter. In Octave the toolbox's call to
MATLAB's [r, p] = corr(x, y) is served by octave_shims/corr.m (Pearson r with the identical
two-sided t-test p-value); the toolbox code itself is unchanged. An edge i -> j is present when
cb(i, j) > 0. Granger methods cannot represent self-connections, so the diagonal is zero.
Scoring is identical to the existing scripts (directed strict CS for autoregressive paradigms;
full-graph CS for spiking networks).

Usage: KGC_DIR=<toolbox dir> OCTAVE=<octave-cli> N_WORKERS=48 python kgc_official_marinazzo.py
Outputs (<OUT>/table1_baselines/): kgc_official_fig2.csv, kgc_official_spiking.csv
"""
import os, sys, csv, subprocess, tempfile
import numpy as np
from scipy.io import savemat, loadmat
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', 'shared')))
from paths import out as _out
from sim_scm import simulate_extended
from directed_metrics import compute_directed_metrics
from tpc_official_pkg import sim_spiking, cs_full, NOISE, REGIMES, P

KGC_DIR = os.environ['KGC_DIR']
OCTAVE = os.environ.get('OCTAVE', 'octave-cli')
NS = int(os.environ.get('NS', '50'))
NW = int(os.environ.get('N_WORKERS', '48'))
KTYPE, KPAR, M = 'p', 2, 2

DRIVER = r"""
addpath('%(kgc)s'); addpath('%(shim)s');  %% corr.m shim: MATLAB [r,p]=corr in Octave
S = load('%(inp)s');
n = numel(S.datas); cbs = zeros(n, %(p)d, %(p)d); fails = zeros(n, 1);
for k = 1:n
  try
    [cb, ifail] = causality(S.datas{k}, '%(kt)s', %(kp)d, %(m)d);
    cbs(k, :, :) = reshape(cb, [1, %(p)d, %(p)d]); fails(k) = ifail;
  catch err
    fails(k) = -1; fprintf(2, 'call %%d failed: %%s\n', k, err.message);
  end
end
save('-mat7-binary', '%(out)s', 'cbs', 'fails');
"""


def main():
    tasks = [('ar', r, nz, s) for r in REGIMES for nz in NOISE for s in range(NS)]
    tasks += [('spk', 'control', m, s) for m in ['convergence', 'diamond', 'chain'] for s in range(NS)]
    tasks += [('spk', 'recurrent', m, s) for m in ['convergence', 'diamond', 'chain', 'depression'] for s in range(NS)]
    sims = []
    for t in tasks:
        if t[0] == 'ar':
            o = simulate_extended(t[1], t[2], 1000, t[3]); sims.append(o)
        else:
            X, GTc, GTs = sim_spiking(t[3], t[2], 0.0 if t[1] == 'control' else -1.5); sims.append((X, GTc, GTs))
    work = tempfile.mkdtemp(prefix='kgc_', dir=os.environ.get('KGC_TMP'))
    shards = np.array_split(np.arange(len(tasks)), NW)

    def run_shard(k):
        idx = shards[k]
        if len(idx) == 0:
            return k, None
        inp, outp, drv = (os.path.join(work, f'{n}{k}.{e}') for n, e in (('in', 'mat'), ('out', 'mat'), ('drv', 'm')))
        datas = np.empty(len(idx), dtype=object)
        for a, i in enumerate(idx):
            datas[a] = np.asarray(sims[i][0], dtype=float).T          # (n_samples, n_vars)
        savemat(inp, {'datas': datas})
        open(drv, 'w').write(DRIVER % dict(kgc=KGC_DIR, shim=os.path.join(HERE, 'octave_shims'), inp=inp, out=outp, p=P, kt=KTYPE, kp=KPAR, m=M))
        env = dict(os.environ); env.setdefault('OCTAVE_HOME', os.path.dirname(os.path.dirname(OCTAVE)))
        wd = os.path.join(work, f'wd{k}'); os.makedirs(wd, exist_ok=True)   # toolbox writes a scratch file into cwd
        r = subprocess.run([OCTAVE, '--no-gui', '--quiet', drv], capture_output=True, text=True, env=env, cwd=wd)
        if 'failed:' in r.stderr:
            print(f'shard {k}:', '\n'.join(l for l in r.stderr.splitlines() if 'failed:' in l)[:1000], flush=True)
        if r.returncode != 0 or not os.path.exists(outp):
            return k, ('ERR', r.stderr[-2000:])
        d = loadmat(outp)
        return k, (d['cbs'], d['fails'].ravel())

    res = {}
    with ThreadPoolExecutor(NW) as ex:
        for k, v in ex.map(run_shard, range(NW)):
            res[k] = v
    ar, spk, nfail = [], [], 0
    for k, idx in enumerate(shards):
        v = res[k]
        if v is None:
            continue
        if isinstance(v[0], str):
            print('SHARD ERROR', k, v[1]); continue
        cbs, fails = v
        for a, i in enumerate(idx):
            t = tasks[i]
            if fails[a] != 0:
                nfail += 1
            A = (np.nan_to_num(cbs[a]) > 0).astype(int); np.fill_diagonal(A, 0)
            if t[0] == 'ar':
                _, gl_uw, gl_w, gc_uw, gc_w, gb_uw, gb_lw, gb_cw = sims[i]
                m = compute_directed_metrics(A, gl_w, gc_w, gb_lw, gb_cw, gl_uw, gc_uw, gb_uw)
                ar.append(['KernelGC', t[1], t[2], t[3], m['directed_TPR_strict'], m['directed_FPR_strict'], m['directed_CS_strict']])
            else:
                X, GTc, GTs = sims[i]
                spk.append([t[1], t[2], t[3], cs_full(A, ((GTc + GTs) > 0).astype(int)), 0.0])
    with open(_out('table1_baselines', 'kgc_official_fig2.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['method', 'regime', 'noise', 'seed', 'TPR', 'FPR', 'CS']); w.writerows(sorted(ar, key=lambda r: (r[1], r[2], r[3])))
    with open(_out('table1_baselines', 'kgc_official_spiking.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['block', 'cell', 'seed', 'cs', 'self_frac']); w.writerows(sorted(spk))
    import pandas as pd
    a = pd.DataFrame(ar, columns=['method', 'regime', 'noise', 'seed', 'TPR', 'FPR', 'CS'])
    print(f'runs {len(ar) + len(spk)} of {len(tasks)}; toolbox failures (ifail != 0) {nfail}')
    print(a[a.noise == 1].groupby('regime')['CS'].agg(['mean', 'std']).round(3))
    b = pd.DataFrame(spk, columns=['block', 'cell', 'seed', 'cs', 'self_frac'])
    print(b.groupby(['block', 'cell'])['cs'].agg(['mean', 'std']).round(3))
    print('KGC OFFICIAL DONE', flush=True)


if __name__ == '__main__':
    main()
