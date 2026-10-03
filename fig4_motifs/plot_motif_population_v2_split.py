#!/usr/bin/env python3
"""Three SEPARATE population plots (a Adjacent, b Non-adjacent common-cause/chain/path,
c Non-adjacent collider) -- same data/analysis as plot_motif_population_v2.py, split
into standalone figures. Each: motif schematic (top) + the two bars (A⊥B, A⊥B|S)."""
import os, sys, pickle as pkl
from itertools import combinations
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.transforms import blended_transform_factory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared')))
from paths import outdir as _outdir, neuropixels
from cits.methods import data_transform, partial_corr

DATA_DIR=neuropixels(); OUT=_outdir('fig4_motifs')   # were citsproject/data and CITS_manuscript/figures
SESS,STIM,BIN,IDX,TAU,ALPHA=791319847,'natural_scenes',0.01,0,1,0.05; MARG_DEP=0.1
raw=np.asarray(pkl.load(open(f'{DATA_DIR}/ID{SESS}_{STIM}_bin_{BIN}_X_idx-{IDX}.p','rb')),float)
mask=np.asarray(pkl.load(open(f'{DATA_DIR}/ID{SESS}_{STIM}_units2use_stim_{STIM}.p','rb')))
units_idx=np.where(mask)[0] if mask.dtype==bool else mask
data=raw[:,units_idx]; data=(data-data.mean(0))/np.where(data.std(0)==0,1,data.std(0)); X=data.T
chi=data_transform(X,TAU); p=X.shape[0]; N=chi.shape[1]
SRC_SLOT,TGT_SLOT=2*TAU,2*TAU+1; r_thr=np.tanh(1.959963985/np.sqrt(N-3))
d=pkl.load(open(f'{OUT}/motif_population_v2_records.pkl','rb')); adj=d['adj']; recs=d['records']
g2l={int(g):i for i,g in enumerate(units_idx)}; sepset={}
for r in recs:
    if r['is_edge']==0: sepset[(r['u_local'],r['v_local'])]=tuple(g2l[s] for s in r['sepset_global'])
parents_of={v:[u for u in range(p) if adj[u,v]] for v in range(p)}
def row(v,slot): return slot*p+v
def lagged_pcorr(u,v,cond): return abs(partial_corr(row(u,SRC_SLOT),row(v,TGT_SLOT),set(row(c,SRC_SLOT) for c in cond),chi))
def cond_on_parents(u,v): return lagged_pcorr(u,v,[w for w in parents_of[v] if w!=u])
_pe={}
def pair_eval(x,y):
    if (x,y) in _pe: return _pe[(x,y)]
    marg=lagged_pcorr(x,y,[]); S=sepset.get((x,y),None)
    cond=cond_on_parents(x,y) if S is None else lagged_pcorr(x,y,S); _pe[(x,y)]=(marg,cond); return _pe[(x,y)]
def nonadj_role(a,b):
    ma,ca=pair_eval(a,b); mb,cb=pair_eval(b,a); return max(ma,mb),max(ca,cb)
def adjacent(a,b): return adj[a,b]==1 or adj[b,a]==1
def edge_undirected(u,v):
    c=[]
    if adj[u,v]: c.append(pair_eval(u,v))
    if adj[v,u]: c.append(pair_eval(v,u))
    return max(c,key=lambda t:t[0])
fork_roles={'C_A':[],'C_B':[],'A_B':[]}
for C in range(p):
    ch=[w for w in range(p) if adj[C,w]]
    for A,B in combinations(ch,2):
        if adjacent(A,B): continue
        m,c=nonadj_role(A,B)
        if m<=MARG_DEP: continue
        fork_roles['C_A'].append(pair_eval(C,A)); fork_roles['C_B'].append(pair_eval(C,B)); fork_roles['A_B'].append((m,c))
chain_roles={'A_B':[],'B_C':[],'A_C':[]}
for B in range(p):
    pa=[w for w in range(p) if adj[w,B]]; ch=[w for w in range(p) if adj[B,w]]
    for A in pa:
        for C in ch:
            if A==C or adjacent(A,C): continue
            m,c=nonadj_role(A,C)
            if m<=MARG_DEP: continue
            chain_roles['A_B'].append(pair_eval(A,B)); chain_roles['B_C'].append(pair_eval(B,C)); chain_roles['A_C'].append((m,c))
path_roles={'P1_A':[],'P2_B':[],'P1_P2':[],'A_B':[]}
for P1,P2 in [(i,j) for i in range(p) for j in range(i+1,p) if adjacent(i,j)]:
    chA=[w for w in range(p) if adj[P1,w] and w not in (P1,P2)]; chB=[w for w in range(p) if adj[P2,w] and w not in (P1,P2)]
    for A in chA:
        for B in chB:
            if A==B or adjacent(A,B): continue
            m,c=nonadj_role(A,B)
            if m<=MARG_DEP: continue
            path_roles['A_B'].append((m,c))
coll_roles={'A_B':[]}
for C in range(p):
    pa=[w for w in range(p) if adj[w,C]]
    for A,B in combinations(pa,2):
        if adjacent(A,B): continue
        Sab=sepset.get((A,B),None); Sba=sepset.get((B,A),None)
        if not (Sab is not None and len(Sab)==0 and Sba is not None and len(Sba)==0): continue
        marg=abs(partial_corr(row(A,SRC_SLOT),row(B,SRC_SLOT),set(),chi))
        condC=abs(partial_corr(row(A,SRC_SLOT),row(B,SRC_SLOT),{row(C,TGT_SLOT)},chi))
        coll_roles['A_B'].append((marg,condC))
def pct(lst):
    if not lst: return (0.0,0.0,0)
    m=100.0*np.mean([1.0 if a>=r_thr else 0.0 for a,_ in lst]); c=100.0*np.mean([1.0 if b>=r_thr else 0.0 for _,b in lst]); return float(m),float(c),len(lst)
edges=(fork_roles['C_A']+fork_roles['C_B']+chain_roles['A_B']+chain_roles['B_C']+path_roles['P1_A']+path_roles['P2_B']+path_roles['P1_P2'])
nonadj=fork_roles['A_B']+chain_roles['A_C']+path_roles['A_B']
coll_par=[(m,m) for (m,_c) in coll_roles['A_B']]
CATS={'a':('Adjacent',pct(edges),'adj'),'b':('Non-adjacent\n(common cause / chain / path)',pct(nonadj),'non'),'c':('Non-adjacent (collider)',pct(coll_par),'coll')}

# ---- style ----
C_CORR='#0072B2'; C_UNCORR='#E69F00'; C_UNCORR_TX='#b45f06'
FS=1.9   # global font scale-up (~2x); canvas + node radii scaled to match
plt.rcParams.update({'font.size':int(11*FS),'font.family':'sans-serif','font.sans-serif':['Arial','Liberation Sans','DejaVu Sans'],'axes.linewidth':1.1})
def _node(ax,x,y,l,r=0.15,fs=20):
    # node letters are kept modest so they sit inside the (wide, flat) ellipses without
    # overflowing; only the surrounding labels/ticks/percentages are scaled ~2x.
    ax.add_patch(Circle((x,y),r,facecolor='#e8e8e8',edgecolor='#333',lw=1.8,zorder=3))
    ax.text(x,y,l,ha='center',va='center',fontsize=fs*1.2,fontweight='bold',color='#222',zorder=4)
def _arrow(ax,pa,pb,r=0.15,color='#333',ms=26,lw=2.8):
    dd=np.array(pb,float)-np.array(pa,float); L=np.hypot(*dd); u=dd/L
    ax.add_patch(FancyArrowPatch(tuple(np.array(pa,float)+u*r),tuple(np.array(pb,float)-u*r),arrowstyle='-|>',mutation_scale=ms,lw=lw,color=color,zorder=2))

def schematic(ax,kind):
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
    if kind=='adj':
        _node(ax,0.30,0.58,'A'); _node(ax,0.70,0.58,'B'); _arrow(ax,(0.30,0.58),(0.70,0.58),color=C_CORR,lw=3.4,ms=30)
        ax.text(0.5,0.14,'A, B directly connected',ha='center',fontsize=15*FS,fontweight='bold',color=C_CORR)
    elif kind=='non':
        rm,fm,ma=0.065,12,17
        _node(ax,0.10,0.92,'A',rm,fm); _node(ax,0.35,0.92,'C',rm,fm); _node(ax,0.60,0.92,'B',rm,fm)
        _arrow(ax,(0.35,0.92),(0.10,0.92),r=rm,color=C_CORR,ms=ma,lw=2.4); _arrow(ax,(0.35,0.92),(0.60,0.92),r=rm,color=C_CORR,ms=ma,lw=2.4)
        ax.text(0.90,0.92,'common\ncause',ha='center',va='center',fontsize=11*FS,color='#333')
        _node(ax,0.10,0.62,'A',rm,fm); _node(ax,0.35,0.62,'C',rm,fm); _node(ax,0.60,0.62,'B',rm,fm)
        _arrow(ax,(0.10,0.62),(0.35,0.62),r=rm,color=C_CORR,ms=ma,lw=2.4); _arrow(ax,(0.35,0.62),(0.60,0.62),r=rm,color=C_CORR,ms=ma,lw=2.4)
        ax.text(0.90,0.62,'chain',ha='center',va='center',fontsize=11*FS,color='#333')
        _node(ax,0.07,0.32,'A',rm,fm); _node(ax,0.29,0.32,'C$_1$',rm,fm); _node(ax,0.51,0.32,'C$_2$',rm,fm); _node(ax,0.73,0.32,'B',rm,fm)
        _arrow(ax,(0.29,0.32),(0.07,0.32),r=rm,color=C_CORR,ms=ma,lw=2.4)
        ax.add_patch(FancyArrowPatch((0.29+rm,0.32),(0.51-rm,0.32),arrowstyle='<|-|>',mutation_scale=ma,lw=2.4,color=C_CORR,zorder=2))  # bidirectional
        _arrow(ax,(0.51,0.32),(0.73,0.32),r=rm,color=C_CORR,ms=ma,lw=2.4)
        ax.text(0.90,0.32,'path',ha='center',va='center',fontsize=11*FS,color='#333')
        ax.text(0.5,0.04,'A, B: no direct connection',ha='center',fontsize=15*FS,fontweight='bold',color=C_UNCORR_TX)
    else:  # collider -- wider A/B spread + bigger vertical gap so the A->C, B->C arrows show
        _node(ax,0.5,0.40,'C'); _node(ax,0.24,0.84,'A'); _node(ax,0.76,0.84,'B')
        _arrow(ax,(0.24,0.84),(0.5,0.40),color=C_CORR,lw=3.6,ms=32); _arrow(ax,(0.76,0.84),(0.5,0.40),color=C_CORR,lw=3.6,ms=32)
        ax.text(0.5,0.0,'A, B: no direct connection',ha='center',fontsize=15*FS,fontweight='bold',color=C_UNCORR_TX)

for letter,(title,(mv,cv,n),kind) in CATS.items():
    # 'non' (b): 2-line title + 3-row schematic -> tall strip. 'coll' (c): medium strip.
    # 'adj' (a): the bar plot is almost empty (bars ~0%), so the motif is drawn INSIDE
    # that empty upper region instead of wasting a separate strip.
    if kind == 'non':
        fig = plt.figure(figsize=(7.6, 9.2))
        gsr = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.5], hspace=0.10,
                               top=0.84, bottom=0.11, left=0.22, right=0.96)
        axs = fig.add_subplot(gsr[0]); axb = fig.add_subplot(gsr[1]); sup_y = 0.985
    elif kind == 'coll':
        fig = plt.figure(figsize=(7.4, 7.8))
        gsr = fig.add_gridspec(2, 1, height_ratios=[0.62, 1.6], hspace=0.08,
                               top=0.905, bottom=0.12, left=0.22, right=0.96)
        axs = fig.add_subplot(gsr[0]); axb = fig.add_subplot(gsr[1]); sup_y = 0.975
    else:  # adj -- single bar axes; motif drawn in the empty upper plot area via an inset
        fig = plt.figure(figsize=(7.4, 7.2))
        axb = fig.add_axes([0.22, 0.13, 0.74, 0.73])   # top lowered so the axis clears the title
        axs = axb.inset_axes([0.12, 0.42, 0.80, 0.54]); axs.set_facecolor('none')
        sup_y = 0.975
    schematic(axs,kind)
    bc=C_CORR if kind=='adj' else C_UNCORR
    v1=100-mv; v2=100-cv
    axb.bar([0],[v1],0.62,color=bc,zorder=3); axb.bar([1],[v2],0.62,color=bc,zorder=3)
    axb.text(0,v1+2,f'{v1:.0f}%',ha='center',fontsize=18*FS,fontweight='bold',color='#333')
    axb.text(1,v2+2,f'{v2:.0f}%',ha='center',fontsize=18*FS,fontweight='bold',color='#333')
    axb.set_xticks([0,1]); axb.set_xticklabels([r'$A \perp B$', r'$A \perp B \mid \mathrm{some}\ S$'],fontsize=15*FS)
    axb.set_xlim(-0.6,1.6); axb.set_ylim(0,108); axb.set_yticks([0,25,50,75,100])
    axb.tick_params(axis='y',labelsize=14*FS); axb.spines[['top','right']].set_visible(False)
    axb.set_ylabel(r'% of pairs independent',fontsize=16*FS)
    fig.suptitle(f'{letter}   {title}',fontsize=17*FS,fontweight='bold',x=0.05,ha='left',y=sup_y)
    fig.text(0.5,0.02,f'n = {n:,}',ha='center',fontsize=12*FS,color='#555')
    fig.savefig(f'{OUT}/motif_population_v2_{letter}.png',dpi=200,bbox_inches='tight',facecolor='white')
    fig.savefig(f'{OUT}/motif_population_v2_{letter}.pdf',bbox_inches='tight',facecolor='white')
    print(f'saved motif_population_v2_{letter}  ({title.splitlines()[0]}) n={n} : {v1:.0f}% / {v2:.0f}%')
