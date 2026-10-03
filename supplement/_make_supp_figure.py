"""Supplementary scaling grid (runtime only), organized by sample size. One panel
per sample size N; within each, method runtime vs p is compared. Methods use the
same colorblind-safe Okabe-Ito palette as the main scaling figure. A curve stops
at the largest p a method completed within the 30-min per-graph budget. Accuracy
(combined score) is reported in the main figure; it is omitted here because at low
N the walled baselines vanish, which makes the accuracy panels read misleadingly."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared')))
from paths import out as _out, outdir as _outdir, result

# grid CSVs from fig1_scaling: $CITS_PAPER_OUT/fig1_scaling/ if present, else fig1_scaling/source_data/
d = pd.read_csv(result('fig1_scaling', 'grid_v3.csv'))
if os.path.exists(result('fig1_scaling', 'grid_ext.csv')): d = pd.concat([d, pd.read_csv(result('fig1_scaling', 'grid_ext.csv'))], ignore_index=True)
agg = d.groupby(['method','p','N','status']).agg(cs=('cs','mean'), rt=('runtime_sec','mean')).reset_index()

methods = ['CITS-GPU','PCMCI+','TPC','KernelGC','LPCMCI']
disp = {'CITS-GPU':'CITS'}
col  = {'CITS-GPU':'#0072B2','PCMCI+':'#D55E00','TPC':'#009E73','KernelGC':'#CC79A7','LPCMCI':'#E69F00'}
mk   = {'CITS-GPU':'o','PCMCI+':'s','TPC':'^','KernelGC':'D','LPCMCI':'v'}
N_GRID = [125,250,500,1000]

fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.2), sharex=True, sharey=True)
axes = axes.ravel()
for i, N in enumerate(N_GRID):
    ax = axes[i]
    for m in methods:
        sub = agg[(agg.method==m) & (agg.N==N) & (agg.status=='ok')].sort_values('p')
        if not len(sub): continue
        lw = 2.2 if m=='CITS-GPU' else 1.4
        z  = 3 if m=='CITS-GPU' else 2
        ax.plot(sub.p, sub.rt.clip(lower=0.05), mk[m]+'-', color=col[m], ms=4, lw=lw, zorder=z,
                label=disp.get(m,m))
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_title(f'$N={N}$', fontsize=10, fontweight='bold', loc='left')
    ax.spines[['top','right']].set_visible(False)
    if i in (0,2): ax.set_ylabel('runtime (s)', fontsize=9)
    if i in (2,3): ax.set_xlabel('Number of variables, $p$', fontsize=9)
    if i==0: ax.legend(frameon=False, fontsize=8, ncol=2, loc='upper left')
fig.suptitle('Runtime of all methods across $p$, at each sample size $N$', fontsize=11, y=0.995)
fig.tight_layout(rect=[0,0,1,0.97])
fig.savefig(_out('supplement', 'scaling_grid_supp.png'), dpi=170, bbox_inches='tight')
for MS in (_outdir('supplement'),):   # was CITS_manuscript/figures and .../final_figures_2026-08-31 (two identical copies)
    fig.savefig(f'{MS}/scaling_supp.png', dpi=170, bbox_inches='tight')
    fig.savefig(f'{MS}/scaling_supp.pdf', bbox_inches='tight')
print(f"saved runtime-only scaling_supp.{{png,pdf}} to {_outdir('supplement')}")
