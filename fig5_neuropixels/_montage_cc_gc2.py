#!/usr/bin/env python3
"""Fig 5A GC2 column (bruceR conditional Granger, VAR order 1) on the SAME concatenated
90 s window. Run in the gc2r conda env. Outputs cmp_cc_<stim>_GC2_68.npy."""
import sys, io, numpy as np, pickle as pkl
from contextlib import redirect_stdout, redirect_stderr
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import FloatVector, DataFrame
from rpy2.rlike.container import OrdDict
gcpack = importr('bruceR'); vars = importr('vars')

SESS, DATA = 791319847, '/home/rbiswas1/citsproject/data'
OUT = '/home/rbiswas1/microns/CITS_manuscript/figures'
STIMS = ['natural_scenes', 'static_gratings', 'gabors']
WBINS, ALPHA = 9000, 0.05
labels_ordered = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam',
                  'CA1', 'CA2', 'CA3', 'DG', 'SUB', 'POL', 'LGv', 'LP']
uu = np.load(f'{OUT}/cits_v2_union_units.npy'); ul = np.load(f'{OUT}/cits_v2_union_labels.npy', allow_pickle=True)
perm = []
for l in labels_ordered:
    perm += list(np.where(ul == l)[0])
perm = np.array(perm); u2u = {int(u): i for i, u in enumerate(uu)}


def emb(A, gids):
    U = np.zeros((68, 68))
    for i in range(A.shape[0]):
        ui = u2u.get(gids[i])
        if ui is None:
            continue
        for j in range(A.shape[1]):
            uj = u2u.get(gids[j])
            if uj is None or ui == uj:
                continue
            U[ui, uj] = A[i, j]
    return U[np.ix_(perm, perm)]


def gc2(data, alpha):
    p = data.shape[1]
    od = OrdDict([('V%d' % c, FloatVector(np.asarray(data[:, c], dtype=float))) for c in range(p)])
    df = DataFrame(od); vm = vars.VAR(df, 1); trap = io.StringIO()
    with redirect_stderr(trap), redirect_stdout(trap):
        grangerout = gcpack.granger_causality(vm)
    res = grangerout.rx2('result')
    eq = [str(x) for x in res.rx2('Equation')]; ex = [str(x) for x in res.rx2('Excluded')]
    pv_all = [float(x) if x is not None else np.nan for x in res.rx2('p.Chisq')]
    fv_all = [float(x) if x is not None else np.nan for x in res.rx2('F')]
    Fmag = np.zeros((p, p)); adj = np.zeros((p, p))
    for r in range(len(eq)):
        if ex[r] == 'ALL':
            continue
        i = int(eq[r][1:]); j = int(ex[r][1:]); fv = fv_all[r]; pv = pv_all[r]
        Fmag[j, i] = fv if np.isfinite(fv) else 0.0
        if np.isfinite(pv) and pv <= alpha:
            adj[j, i] = 1
    return Fmag, adj


for stim in STIMS:
    P = np.load(f'{DATA}/P_raw_{stim}.npy'); D = P.reshape(-1, P.shape[2])
    mask = np.asarray(pkl.load(open(f'{DATA}/ID{SESS}_{stim}_units2use_stim_{stim}.p', 'rb')))
    uidx = np.where(mask)[0] if mask.dtype == bool else mask; gids = [int(g) for g in uidx]
    M = D[:WBINS, uidx].astype(float)
    Fmag, adj = gc2(M, ALPHA)
    np.save(f'{OUT}/cmp_cc_{stim}_GC2_68.npy', emb(Fmag, gids))
    np.save(f'{OUT}/cmp_cc_{stim}_GC2adj_68.npy', emb(adj, gids))
    print(f'[{stim}] GC2 cc done M={M.shape}', flush=True)
print('CC GC2 DONE', flush=True)
