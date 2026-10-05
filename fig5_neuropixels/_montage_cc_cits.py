#!/usr/bin/env python3
"""Align Fig 5A CITS column to the SAME data construction as 5B/C: unsmoothed 10 ms,
concatenated presentations, first 90 s window. contemporaneous CITS, SIGNED LSCM weights.
Outputs cmp_cc_<stim>_CITS_68.npy (cc = concatenated-presentation window)."""
import os, sys, numpy as np, pickle as pkl
# repo shared/ (was analysis/functional_circuitry) and this folder (was CITS_manuscript/figures)
for p_ in (os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared')),
           os.path.dirname(os.path.abspath(__file__))):
    if p_ not in sys.path:
        sys.path.insert(0, p_)
from gpu_cits_lag_cupc_faithful import gpu_cits_lag_cupc_faithful
from _pc_raw import pc_skeleton_raw
from _pc_orientation import orient_v_structures
from _union_cpdag import build_union
from _lscm_refit import lscm_refit_cpdag
from paths import outdir as _outdir, result, neuropixels
def _union(f): return result('fig5_neuropixels', f, fallback=neuropixels(f))   # cits_v2_union_*.npy: $CITS_PAPER_OUT/fig5_neuropixels/ if present, else $NEUROPIXELS_DATA

SESS, DATA = 791319847, neuropixels()   # was citsproject/data
OUT = _outdir('fig5_neuropixels')   # was CITS_manuscript/figures
STIMS = ['natural_scenes', 'static_gratings', 'gabors']
AL, TAU, WBINS = 0.05, 1, 9000
labels_ordered = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam',
                  'CA1', 'CA2', 'CA3', 'DG', 'SUB', 'POL', 'LGv', 'LP']
uu = np.load(_union('cits_v2_union_units.npy')); ul = np.load(_union('cits_v2_union_labels.npy'), allow_pickle=True)
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


for stim in STIMS:
    P = np.load(f'{DATA}/P_raw_{stim}.npy'); D = P.reshape(-1, P.shape[2])
    mask = np.asarray(pkl.load(open(f'{DATA}/ID{SESS}_{stim}_units2use_stim_{stim}.p', 'rb')))
    uidx = np.where(mask)[0] if mask.dtype == bool else mask; gids = [int(g) for g in uidx]
    M = D[:WBINS, uidx]                                  # first 90 s window, concatenated presentations
    X = (M - M.mean(0)).T                                # (p, T) centered
    cB = np.asarray(gpu_cits_lag_cupc_faithful(X, alpha=AL, tau=TAU))
    pc_skel, _, sep, _ = pc_skeleton_raw(X.T, alpha=AL, use_gpu=True, verbose=False)
    pc_G = np.asarray(orient_v_structures(pc_skel, sep))
    up, uskel, et = build_union(cB.astype(float), pc_skel, pc_G, sign_amb_mat=None, tau=TAU)
    uB, _ = lscm_refit_cpdag(X.T, pc_G, extra_parents_per_child=up, verbose=False)
    W = np.nan_to_num(np.asarray(uB))                    # SIGNED LSCM weights
    np.save(f'{OUT}/cmp_cc_{stim}_CITS_68.npy', emb(W, gids))
    print(f'[{stim}] CITS cc done: nonzero={int((W!=0).sum())}', flush=True)
print('CC CITS DONE', flush=True)
