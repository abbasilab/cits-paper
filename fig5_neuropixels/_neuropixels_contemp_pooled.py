#!/usr/bin/env python3
"""Pooled contemporaneous CITS (DIRECTED) FC for the Neuropixels stimtypes figure.

Canonical MICrONS contemporaneous CITS, pooled per stimulus (like stim_fc_pipeline):
pool trials -> center per trial, W-align, concat -> cap at NCAP windows ->
run contemporaneous CITS once:
  cuPC lagged (gpu_cits_lag_cupc_faithful, directed by time)
  + PC-contemporaneous (pc_skeleton_raw GPU) + v-structures ONLY (no Meek)
  + build_union + single LSCM refit (signed weights).

Pooling gives the power needed to ORIENT edges (per-short-trial cannot).
Outputs a DIRECTED union graph per stimulus, region-ordered 68-frame:
  cits_v2_pooledB_<stim>_dir68.npy      (i->j = 1; undirected sets both)
  cits_v2_pooledB_<stim>_w68.npy        (|LSCM weight|, signed magnitude)
  cits_v2_pooledB_summary.json

Run: CUDA_VISIBLE_DEVICES=0 python _neuropixels_contemp_pooled.py [--ncap 400] [--stim S] [--validate]
"""
import os, sys, json, time, argparse
import numpy as np
import pickle as pkl

AD = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared'))   # repo shared/ (was analysis/functional_circuitry)
for p in (AD,):
    if p not in sys.path:
        sys.path.insert(0, p)

from paths import outdir as _outdir, result, neuropixels
def _union(f): return result('fig5_neuropixels', f, fallback=neuropixels(f))   # cits_v2_union_*.npy: $CITS_PAPER_OUT/fig5_neuropixels/ if present, else $NEUROPIXELS_DATA

TAU, ALPHA, W = 1, 0.05, 4
SESS, BIN = 791319847, 0.01
DATA = neuropixels()   # was citsproject/data
OUT = _outdir('fig5_neuropixels')   # was CITS_manuscript/figures
STIMS = ['natural_scenes', 'static_gratings', 'gabors']

labels_ordered = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam',
                  'CA1', 'CA2', 'CA3', 'DG', 'SUB', 'POL', 'LGv', 'LP']
union_units = np.load(_union('cits_v2_union_units.npy'))
union_labels = np.load(_union('cits_v2_union_labels.npy'), allow_pickle=True)
permute = []
for lab in labels_ordered:
    permute += list(np.where(union_labels == lab)[0])
permute = np.array(permute)
unit_to_union = {int(u): i for i, u in enumerate(union_units)}


def pool_X(stim, units_idx, ncap):
    blocks = []
    for idx in range(60):
        f = f'{DATA}/ID{SESS}_{stim}_bin_{BIN}_X_idx-{idx}.p'
        if not os.path.exists(f):
            continue
        raw = np.asarray(pkl.load(open(f, 'rb')), float)
        b = (raw[:, units_idx] - raw[:, units_idx].mean(0)).T          # (p, L) centered per trial
        L = (b.shape[1] // W) * W
        if L >= W:
            blocks.append(b[:, :L])
    X = np.concatenate(blocks, axis=1)
    return X[:, :ncap * W]                                              # cap N windows


def contemp_directed(X):
    """Return (dir_adj pxp int: i->j=1 undirected both, |LSCM| weights pxp)."""
    from gpu_cits_lag_cupc_faithful import gpu_cits_lag_cupc_faithful
    from _pc_raw import pc_skeleton_raw
    from _pc_orientation import orient_v_structures
    from _union_cpdag import build_union
    from _lscm_refit import lscm_refit_cpdag
    cB = gpu_cits_lag_cupc_faithful(X, alpha=ALPHA, tau=TAU)            # rolled lagged binary (directed)
    Xtp = X.T
    pc_skel, pc_r0, sep_sets, inactive = pc_skeleton_raw(Xtp, alpha=ALPHA, use_gpu=True, verbose=False)
    pc_G = np.asarray(orient_v_structures(pc_skel, sep_sets))           # NO MEEK
    union_parents, union_skel, edge_type = build_union(
        cB.astype(float), pc_skel, pc_G, sign_amb_mat=None, tau=TAU)
    union_B, _ = lscm_refit_cpdag(Xtp, pc_G, extra_parents_per_child=union_parents, verbose=False)
    union_B = np.nan_to_num(np.asarray(union_B), nan=0.0, posinf=0.0, neginf=0.0)
    p = X.shape[0]
    dir_adj = np.zeros((p, p), int)
    cBm = np.asarray(cB)
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            present = (union_skel[i, j] != 0) or (union_skel[j, i] != 0)
            if not present:
                continue
            # orientation: contemp directed (pc_G asymmetric) or lagged (cB) -> i->j;
            # otherwise undirected -> set both.
            contemp_dir = (pc_G[i, j] != 0 and pc_G[j, i] == 0)
            lagged_dir = (cBm[i, j] != 0 and cBm[j, i] == 0)
            if contemp_dir or lagged_dir:
                dir_adj[i, j] = 1
            elif (pc_G[i, j] != 0 and pc_G[j, i] != 0):
                dir_adj[i, j] = 1  # undirected; both set when j loops
    return dir_adj, np.abs(union_B)


def run(stim, ncap, validate=False):
    mask = np.asarray(pkl.load(open(f'{DATA}/ID{SESS}_{stim}_units2use_stim_{stim}.p', 'rb')))
    units_idx = np.where(mask)[0] if mask.dtype == bool else mask
    gids = [int(g) for g in units_idx]
    p = len(units_idx)
    X = pool_X(stim, units_idx, ncap)
    t0 = time.time()
    dir_adj, Wt = contemp_directed(X)
    dt = time.time() - t0
    # counts
    A = dir_adj.astype(bool)
    directed = int((A & ~A.T).sum())
    undirected = int((A & A.T).sum() // 2)
    print(f'[{stim}] pooled N={X.shape[1]//W} windows, p={p}: {dt:.1f}s  '
          f'directed={directed}  undirected={undirected}', flush=True)
    if validate:
        return {'stim': stim, 'directed': directed, 'undirected': undirected, 'secs': round(dt, 1)}
    # embed to 68 frame
    D = np.zeros((68, 68), int); Wn = np.zeros((68, 68))
    for i in range(p):
        ui = unit_to_union.get(gids[i])
        if ui is None:
            continue
        for j in range(p):
            uj = unit_to_union.get(gids[j])
            if uj is None or ui == uj:
                continue
            if dir_adj[i, j]:
                D[ui, uj] = 1; Wn[ui, uj] = Wt[i, j]
    np.save(f'{OUT}/cits_v2_pooledB_{stim}_dir68.npy', D[np.ix_(permute, permute)])
    np.save(f'{OUT}/cits_v2_pooledB_{stim}_w68.npy', Wn[np.ix_(permute, permute)])
    return {'stim': stim, 'p': p, 'n_windows': X.shape[1] // W,
            'directed': directed, 'undirected': undirected, 'secs': round(dt, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ncap', type=int, default=400)
    ap.add_argument('--stim', default=None)
    ap.add_argument('--validate', action='store_true')
    args = ap.parse_args()
    stims = [args.stim] if args.stim else STIMS
    summ = {'method': 'pooled contemporaneous CITS (cuPC lagged + PC-contemp, v-structures no Meek, '
                      'union + single LSCM), N-capped', 'ncap': args.ncap, 'stimuli': {}}
    for stim in stims:
        summ['stimuli'][stim] = run(stim, args.ncap, validate=args.validate)
    if not args.validate:
        json.dump(summ, open(f'{OUT}/cits_v2_pooledB_summary.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
