#!/usr/bin/env python3
import numpy as np, pickle as pkl, os
FIG='/home/rbiswas1/microns/CITS_manuscript/figures'
DATA='/home/rbiswas1/citsproject/data'
SESS=791319847; WBINS=9000
STIMS=['natural_scenes','static_gratings','gabors']
lo=['VISp','VISl','VISrl','VISal','VISpm','VISam','CA1','CA2','CA3','DG','SUB','POL','LGv','LP']
uu=np.load(f'{FIG}/cits_v2_union_units.npy'); ul=np.load(f'{FIG}/cits_v2_union_labels.npy',allow_pickle=True)
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

print("="*70)
print("METRIC 1/2/3: montage density, AFC-fidelity, |r|-percentile (p<=0.05)")
print("="*70)
agg={m:{'edges':0,'supp':0,'pcts':[]} for m in ['GC1','GC2','CITS']}
tot_pairs=0
for s in STIMS:
    m=np.asarray(pkl.load(open(f'{DATA}/ID{SESS}_{s}_units2use_stim_{s}.p','rb')))
    ui=np.where(m)[0] if m.dtype==bool else m; p=len(ui)
    npairs=p*(p-1); tot_pairs+=npairs
    # full |r| over active pairs, embedded to 68-frame for consistent indexing
    P=np.load(f'{DATA}/P_raw_{s}.npy'); Dm=P.reshape(-1,P.shape[2])
    Mw=Dm[:WBINS,ui].astype(float)
    R=np.corrcoef(Mw.T); np.fill_diagonal(R,0.0)
    gids=[int(g) for g in ui]
    Rfull=emb(np.abs(R),gids)                      # |r| full, 68-frame
    corr=np.load(f'{FIG}/cmp_ccsig_{s}_CORR_68.npy')   # signed r sig
    # active-pair mask in 68-frame
    act=np.array([u2u[int(g)] for g in ui]); act=np.array([np.where(perm==a)[0][0] for a in act])
    mask=np.zeros((68,68),bool); mask[np.ix_(act,act)]=True; np.fill_diagonal(mask,False)
    rvals=Rfull[mask]                              # distribution of |r| over active pairs
    def pct(x): return float((rvals<=x).mean()*100)
    corr_dens=(corr!=0).sum()/npairs
    print(f"\n[{s}] active p={p}, ordered pairs={npairs}")
    print(f"   AFC (Pearson sig) density = {corr_dens:.3f}")
    for meth in ['GC1','GC2','CITS']:
        A=np.load(f'{FIG}/cmp_ccsig_{s}_{meth}_68.npy')
        E=(A!=0)&mask
        ne=int(E.sum()); dens=ne/npairs
        # fidelity: edges whose (i,j) OR (j,i) has sig marginal corr
        csym=(corr!=0)|(corr.T!=0)
        supp=int((E&csym).sum()); fid=supp/ne if ne else float('nan')
        # percentile of |r| for this method's edges
        ii,jj=np.where(E); pcts=[pct(Rfull[a,b]) for a,b in zip(ii,jj)]
        medp=float(np.median(pcts)) if pcts else float('nan')
        print(f"   {meth:4s}: edges={ne:4d} dens={dens:.3f} fidelity={fid*100:5.1f}%  median|r|pct={medp:4.1f}")
        agg[meth]['edges']+=ne; agg[meth]['supp']+=supp; agg[meth]['pcts']+=pcts
print("\n"+"-"*70)
print("POOLED across 3 stimuli (ordered active pairs total = %d):"%tot_pairs)
for meth in ['GC1','GC2','CITS']:
    e=agg[meth]['edges']; su=agg[meth]['supp']; pc=agg[meth]['pcts']
    print(f"   {meth:4s}: density={e/tot_pairs:.3f}  fidelity={su/e*100:.1f}%  median|r|pct={np.median(pc):.1f}")

print("\n"+"="*70)
print("EDGE COUNTS: stimtypes graph at >=90% window-consistency (W90WIN)")
print("="*70)
for s in STIMS:
    fwd=np.load(f'{FIG}/cits_v2_directedB_W90WIN_{s}_fwd68.npy')
    present=(fwd>=0.9)|(fwd.T>=0.9)
    up=np.triu(present,1)
    n_und=int(up.sum())                      # undirected present pairs (edges drawn)
    n_dir_fwd=int(((fwd>=0.9)&~(fwd.T>=0.9)).sum())   # one-directional
    n_bi=int(((fwd>=0.9)&(fwd.T>=0.9)).sum())//2      # reciprocal pairs
    print(f"   {s:16s}: present-pairs(edges)={n_und:3d}  [reciprocal={n_bi}, one-way={n_dir_fwd}]")
# also 80% for the robustness sentence
print("\n  at >=80% (robustness sentence):")
for s in STIMS:
    fwd=np.load(f'{FIG}/cits_v2_directedB_W90WIN_{s}_fwd68.npy')
    present=(fwd>=0.8)|(fwd.T>=0.8)
    print(f"   {s:16s}: present-pairs(edges)={int(np.triu(present,1).sum()):3d}")

print("\n"+"="*70)
print("ACTIVE-NEURON COUNTS + union")
print("="*70)
active={}
for s in STIMS:
    m=np.asarray(pkl.load(open(f'{DATA}/ID{SESS}_{s}_units2use_stim_{s}.p','rb')))
    ui=np.where(m)[0] if m.dtype==bool else m
    active[s]=set(int(g) for g in ui)
    print(f"   {s:16s}: active={len(ui)}")
union=set().union(*active.values())
print(f"   UNION active neurons = {len(union)} ;  union frame size = {len(uu)}")
