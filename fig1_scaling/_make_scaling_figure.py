"""Main scaling figure from the certified 3-seed grid (grid_v3.csv).
a: runtime vs p at each method's N* (min N reaching mean CS>=0.95); compute walls
   (all N time out) marked x. b: best accuracy (max mean CS over N) vs p.
Nature-style: open axes, colorblind-safe, italic math symbols."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family']='sans-serif'
plt.rcParams['font.sans-serif']=['Arial','Liberation Sans','Nimbus Sans','DejaVu Sans']  # Arial (Liberation Sans = Arial-metric)
_FS=2.0   # global font scale (2x)
plt.rcParams.update({'font.size':10*_FS,'axes.titlesize':13*_FS,'axes.labelsize':11*_FS,
    'xtick.labelsize':10*_FS,'ytick.labelsize':10*_FS,'legend.fontsize':8.5*_FS,'legend.title_fontsize':8.5*_FS})

import os
d = pd.read_csv('grid_v3.csv')
if os.path.exists('grid_ext.csv'):           # p=1000 walls for TPC/KernelGC
    d = pd.concat([d, pd.read_csv('grid_ext.csv')], ignore_index=True)
ok = d[d.status=='ok'].copy()
# runtime from the certified clean seeds (0-2) only
rt_agg = ok.groupby(['method','p','N']).agg(rt=('runtime_sec','mean'))
# CS pooled over certified + extra accuracy seeds (uniform n; extra seeds are CS-only)
cs_src = ok[['method','p','N','seed','cs']].copy()
if os.path.exists('grid_seeds_ext.csv'):
    ext = pd.read_csv('grid_seeds_ext.csv'); ext = ext[ext.status=='ok']
    cs_src = pd.concat([cs_src, ext[['method','p','N','seed','cs']]], ignore_index=True)
cs_agg = cs_src.groupby(['method','p','N']).agg(cs=('cs','mean'), cs_sd=('cs','std'),
         nseed=('cs','count'))
agg = cs_agg.join(rt_agg).reset_index()
print('CS seeds/point: min=%d max=%d (accuracy error bars = SD over these)' % (agg.nseed.min(), agg.nseed.max()))

def nstar_rows(m):
    # N* = sample size at which mean CS first crosses 0.95, estimated by
    # interpolating between the two bracketing measured N; runtime at N* is
    # interpolated log-linearly from the two measured runtimes. This removes
    # the grid-quantization jitter of reporting the smallest *tested* N.
    sub = agg[agg.method==m]; out=[]
    for p in sorted(sub.p.unique()):
        s = sub[sub.p==p].sort_values('N')
        Ns, css, rts = s.N.values.astype(float), s.cs.values, s.rt.values
        if len(Ns) == 0:
            continue
        if css[0] >= 0.95:                      # already met at smallest tested N
            out.append({'p': p, 'N': Ns[0], 'rt': rts[0]}); continue
        for i in range(1, len(Ns)):             # first upward crossing of 0.95
            if css[i] >= 0.95 and css[i-1] < 0.95:
                frac = (0.95 - css[i-1]) / (css[i] - css[i-1])
                Nstar = Ns[i-1] + frac * (Ns[i] - Ns[i-1])
                if rts[i-1] > 0 and rts[i] > 0:
                    rtstar = rts[i-1] * (rts[i] / rts[i-1]) ** frac
                else:
                    rtstar = rts[i-1] + frac * (rts[i] - rts[i-1])
                out.append({'p': p, 'N': Nstar, 'rt': rtstar}); break
    return pd.DataFrame(out)

def maxcs_rows(m):
    sub = agg[agg.method==m]
    return sub.loc[sub.groupby('p').cs.idxmax()].sort_values('p')

def compute_wall(m):
    md = d[d.method==m]
    for p in sorted(md.p.unique()):
        pp = md[md.p==p]
        if (pp.status=='ok').sum()==0 and (pp.status=='timeout').any():
            return p
    return None

order = ['CITS-GPU','PCMCI+','TPC','KernelGC','LPCMCI']
col = {'CITS-GPU':'#0072B2','PCMCI+':'#D55E00','TPC':'#009E73','KernelGC':'#CC79A7','LPCMCI':'#E69F00'}
mk  = {'CITS-GPU':'o','PCMCI+':'s','TPC':'^','KernelGC':'D','LPCMCI':'v'}
disp = {'CITS-GPU':'CITS (Ours)'}   # display label: highlight our method

fig, (axB, axA, axC) = plt.subplots(1, 3, figsize=(20, 6.2))  # left=axB(combined), mid=axC(sample complexity), right=axA(runtime)

FIXED_N = 1000
for m in order:
    s = agg[(agg.method==m) & (agg.N==FIXED_N)].sort_values('p')     # fixed sample size
    if len(s):
        # line ends at each method's last feasible p; absent nodes at larger p = walled
        axA.plot(s.p, s.rt.clip(lower=0.05), mk[m]+'-', color=col[m], label=disp.get(m,m), lw=2, ms=6)
axA.axhline(1800, ls='--', color='0.5', lw=1)
axA.text(6, 1550, '30-min compute budget', color='0.4', fontsize=17, va='top')   # below the line
axA.set_xscale('log'); axA.set_yscale('log')
axA.set_xlabel('Number of variables, $p$'); axA.set_ylabel('Runtime at $N=1000$ (s)')
axA.set_title('b   Runtime scaling', loc='left', fontweight='bold')
axA.spines[['top','right']].set_visible(False)

# Panel a: p grouped along x (categorical), methods dodged within each group,
# light separators between groups, each p labelled below its group.
PS = sorted(agg.p.unique()); xpos = {p: i for i, p in enumerate(PS)}
_ymin = 1.0; nD = len(order); w = 0.14
for j, m in enumerate(order):
    r = maxcs_rows(m); csv = r.cs.values; sdv = r.cs_sd.fillna(0).values
    xs = [xpos[p] + (j - (nD-1)/2) * w for p in r.p]
    lower = sdv; upper = np.minimum(csv + sdv, 1.0) - csv   # CS<=1: clip upper whisker at 1
    # our method gets a solid, opaque connector; baselines stay faint dashed
    if m == 'CITS-GPU':
        axB.plot(xs, csv, ls='-', color=col[m], lw=2.4, alpha=1.0, zorder=2.5)
    else:
        axB.plot(xs, csv, ls='--', color=col[m], lw=1.2, alpha=0.45, zorder=1)
    axB.errorbar(xs, csv, yerr=np.vstack([lower, upper]), fmt=mk[m], color=col[m], ms=6,
                 capsize=2, lw=0, elinewidth=1.4, label=disp.get(m,m), zorder=2)
    _ymin = min(_ymin, float((csv - sdv).min()))
for i in range(len(PS) - 1):                       # separators between p groups
    axB.axvline(i + 0.5, color='0.88', lw=0.8, zorder=0)
axB.axhline(0.95, ls='--', color='0.5', lw=1)
axB.text(len(PS) - 0.55, 0.944, 'CS = 0.95', color='0.4', fontsize=17, ha='right', va='top')  # right, below line
axB.set_xlim(-0.5, len(PS) - 0.5); axB.set_ylim(min(0.80, _ymin - 0.015), 1.01)
axB.set_xticks(range(len(PS))); axB.set_xticklabels([str(p) for p in PS])
axB.set_xlabel('Number of variables, $p$'); axB.set_ylabel('Combined score, CS')
axB.set_title('a   Combined score', loc='left', fontweight='bold')
axB.spines[['top','right']].set_visible(False)

# Panel c: sample complexity -- N* (min N reaching mean CS>=0.95) vs p
for m in order:
    r = nstar_rows(m)
    if len(r):
        axC.plot(r.p, r.N, mk[m]+'-', color=col[m], label=disp.get(m,m), lw=2, ms=6)
axC.set_xscale('log')
axC.set_yticks([125, 250, 500, 1000]); axC.set_yticklabels(['125','250','500','1000'])
axC.set_ylim(60, 1120)
axC.set_xlabel('Number of variables, $p$'); axC.set_ylabel('$N^{*}$: samples for CS $\\geq$ 0.95')
axC.set_title('c   Sample complexity', loc='left', fontweight='bold')
axC.spines[['top','right']].set_visible(False)

# actual p tick labels (not 10^k) on the log-x panels b and c
from matplotlib.ticker import NullLocator
for ax in (axA, axC):
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticks(PS); ax.set_xticklabels([str(p) for p in PS])

# legend placed vertically inside panel a (Combined score), lower-right empty area
from matplotlib.lines import Line2D
handles = [Line2D([0],[0], marker=mk[m], color=col[m], lw=2, ms=8, label=disp.get(m,m)) for m in order]
leg = axB.legend(handles=handles, loc='lower right', ncol=1, frameon=False,
                 fontsize=15, title='Method', title_fontsize=16,
                 labelspacing=0.35, handletextpad=0.5, borderaxespad=0.6)
leg.get_title().set_fontweight('bold')
fig.tight_layout(rect=[0, 0, 1, 1])
fig.savefig('scaling_grid_figure.png', dpi=200, bbox_inches='tight')
fig.savefig('/home/rbiswas1/microns/CITS_manuscript/figures/scaling_grid_figure.pdf',
            bbox_inches='tight')
agg.to_csv('grid_v3_aggregated.csv', index=False)
print("saved scaling_grid_figure.png ; walls:", {m:compute_wall(m) for m in order})
