#!/usr/bin/env python3
"""Fig 5A panels = magnitude AFTER significance (p<=0.05) for every method, so the
visible density is the honest matched-significance density.
  Pearson Corr: signed r where |r| significant (p<=0.05)
  GC1: pairwise-Granger F where ssr_ftest p<=0.05 (statsmodels)
  GC2: bruceR F where p.Chisq<=0.05 (precomputed cmp_cc_GC2 * GC2adj)
  CITS: signed LSCM weight (already the inferred sparse graph)
Outputs cmp_ccsig_<stim>_{CORR,GC1,GC2,CITS}_68.npy."""
import numpy as np, pickle as pkl, warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.stattools import grangercausalitytests
DATA='/home/rbiswas1/citsproject/data'; SESS=791319847; WBINS=9000; AL=0.05
OUT='/home/rbiswas1/microns/CITS_manuscript/figures'
STIMS=['natural_scenes','static_gratings','gabors']
lo=['VISp','VISl','VISrl','VISal','VISpm','VISam','CA1','CA2','CA3','DG','SUB','POL','LGv','LP']
uu=np.load(f'{OUT}/cits_v2_union_units.npy'); ul=np.load(f'{OUT}/cits_v2_union_labels.npy',allow_pickle=True)
perm=[]
for l in lo: perm+=list(np.where(ul==l)[0])
perm=np.array(perm); u2u={int(u):i for i,u in enumerate(uu)}
def emb(A,gids):
    U=np.zeros((68,68))
    for i in range(A.shape[0]):
        ui=u2u.get(gids[i])
        if ui is None: continue
        for j in range(A.shape[1]):
            uj=u2u.get(gids[j])
            if uj is None or ui==uj: continue
            U[ui,uj]=A[i,j]
    return U[np.ix_(perm,perm)]
thr_r=np.tanh(1.959963985/np.sqrt(WBINS-3))       # |r| significant at p<=0.05
for s in STIMS:
    P=np.load(f'{DATA}/P_raw_{s}.npy'); Dm=P.reshape(-1,P.shape[2])
    m=np.asarray(pkl.load(open(f'{DATA}/ID{SESS}_{s}_units2use_stim_{s}.p','rb')))
    ui=np.where(m)[0] if m.dtype==bool else m; gids=[int(g) for g in ui]
    Mw=Dm[:WBINS,ui].astype(float); p=Mw.shape[1]
    # CORR: signed r masked by significance
    R=np.corrcoef(Mw.T); np.fill_diagonal(R,0); Rm=np.where(np.abs(R)>=thr_r,R,0.0)
    np.save(f'{OUT}/cmp_ccsig_{s}_CORR_68.npy', emb(Rm,gids))
    # GC1: F where p<=0.05
    F1=np.zeros((p,p))
    for j in range(p):
        for i in range(p):
            if i==j: continue
            try:
                r=grangercausalitytests(Mw[:,[j,i]], maxlag=[1], verbose=False)
                F,pv=r[1][0]['ssr_ftest'][0], r[1][0]['ssr_ftest'][1]
                if pv<=AL: F1[i,j]=F
            except Exception: pass
    np.save(f'{OUT}/cmp_ccsig_{s}_GC1_68.npy', emb(F1,gids))
    # GC2: Fmag * (p<=0.05 adj)   [both already in 68-frame]
    g2=np.load(f'{OUT}/cmp_cc_{s}_GC2_68.npy'); g2a=np.load(f'{OUT}/cmp_cc_{s}_GC2adj_68.npy')
    np.save(f'{OUT}/cmp_ccsig_{s}_GC2_68.npy', np.where(g2a!=0, g2, 0.0))
    # CITS: signed LSCM (already thresholded)
    np.save(f'{OUT}/cmp_ccsig_{s}_CITS_68.npy', np.load(f'{OUT}/cmp_cc_{s}_CITS_68.npy'))
    print(f'{s}: sig-magnitude columns saved', flush=True)
print('SIG-MAG DONE', flush=True)
