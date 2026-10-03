#!/usr/bin/env python3
"""Consistency over many 90 s windows (matching Fig 5A's window size), unsmoothed 10 ms.
Concatenate all presentations (raw P, unsmoothed), tile into non-overlapping 90 s windows
(~9000 bins each), run directed Version B on each window, and aggregate presence/direction.
Each window has ~9000 bins (high power, like 5A), so consistency across them is meaningful.
Outputs cits_v2_directedB_W90WIN_<stim>_{fwd,w,pboot}68.npy."""
import os, sys, numpy as np, pickle as pkl, time
AD = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared'))   # repo shared/ (was analysis/functional_circuitry)
# this folder holds _neuropixels_versionB_pooled.py (was CITS_manuscript/figures)
for p_ in (AD, os.path.dirname(os.path.abspath(__file__))):
    if p_ not in sys.path:
        sys.path.insert(0, p_)
from _neuropixels_versionB_pooled import versionB_directed, permute, unit_to_union
from paths import outdir as _outdir, neuropixels

SESS, DATA = 791319847, neuropixels()   # was citsproject/data
OUT = _outdir('fig5_neuropixels')   # was CITS_manuscript/figures
STIMS = ['natural_scenes', 'static_gratings', 'gabors']
if os.environ.get('ONLY_STIM'):
    STIMS = [os.environ['ONLY_STIM']]
WBINS = 9000                                              # 90 s at 10 ms

for stim in STIMS:
    P = np.load(f'{DATA}/P_raw_{stim}.npy')               # (n_pres, 26, 555) unsmoothed
    D = P.reshape(-1, P.shape[2])                          # (total_bins, 555) concatenated
    mask = np.asarray(pkl.load(open(f'{DATA}/ID{SESS}_{stim}_units2use_stim_{stim}.p', 'rb')))
    units_idx = np.where(mask)[0] if mask.dtype == bool else mask
    gids = [int(g) for g in units_idx]; p = len(units_idx)
    Dsub = D[:, units_idx]
    nwin = Dsub.shape[0] // WBINS
    pres = np.zeros((68, 68)); oneway = np.zeros((68, 68))
    wsum = np.zeros((68, 68)); wcnt = np.zeros((68, 68))
    for w in range(nwin):
        Xw = Dsub[w * WBINS:(w + 1) * WBINS]
        Xc = Xw - Xw.mean(0)
        t0 = time.time()
        dir_adj, W = versionB_directed(Xc.T)
        for i in range(p):
            ui = unit_to_union.get(gids[i])
            if ui is None:
                continue
            for j in range(p):
                uj = unit_to_union.get(gids[j])
                if uj is None or ui == uj:
                    continue
                if dir_adj[i, j]:
                    pres[ui, uj] += 1; wsum[ui, uj] += abs(W[i, j]); wcnt[ui, uj] += 1
                    if not dir_adj[j, i]:
                        oneway[ui, uj] += 1
        print(f'[{stim}] win {w+1}/{nwin}: {time.time()-t0:.1f}s dir_edges={int(dir_adj.sum())}', flush=True)
    fwd = pres / nwin
    pboot = np.divide(oneway, pres, out=np.zeros_like(oneway), where=pres > 0)
    w_ = np.divide(wsum, wcnt, out=np.zeros_like(wsum), where=wcnt > 0)
    pm = lambda A: A[np.ix_(permute, permute)]
    np.save(f'{OUT}/cits_v2_directedB_W90WIN_{stim}_fwd68.npy', pm(fwd))
    np.save(f'{OUT}/cits_v2_directedB_W90WIN_{stim}_w68.npy', pm(w_))
    np.save(f'{OUT}/cits_v2_directedB_W90WIN_{stim}_pboot68.npy', pm(pboot))
    print(f'[{stim}] DONE nwin={nwin}  edges@0.7={int(((fwd>=0.7)|(fwd.T>=0.7)).sum())} '
          f'@0.8={int(((fwd>=0.8)|(fwd.T>=0.8)).sum())}', flush=True)
print('ALL DONE', flush=True)
