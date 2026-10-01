#!/usr/bin/env python3
"""Worked-example motif scatter figure (Nature-Neuro style), in the original
cits_fig5_1 spirit. Two motifs: a 3-node FORK and a 4-node PATH.

Each row: pairwise lagged neuron-activity scatters (source at t-1 vs target at t)
with Pearson r. Edges stay correlated; the non-adjacent pair is correlated
marginally, then drops to ~0 when conditioned on the algorithm's recorded
separating set S (honest partial-correlation residuals: residualize BOTH
variables on S). 2x5 grid so the 'non-adjacent' and 'given S' columns align
across rows; the fork row (2 edges) leaves the 3rd edge cell blank."""
import sys, pickle as pkl
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats, linalg

sys.path.insert(0, '/home/rbiswas1/repos/cits')
from cits.methods import data_transform

DATA_DIR = '/home/rbiswas1/citsproject/data'
OUT = '/home/rbiswas1/microns/CITS_manuscript/figures'
SESS, STIM, BIN, IDX, TAU = 791319847, 'natural_scenes', 0.01, 0, 1

raw = np.asarray(pkl.load(open(f'{DATA_DIR}/ID{SESS}_{STIM}_bin_{BIN}_X_idx-{IDX}.p', 'rb')), float)
mask = np.asarray(pkl.load(open(f'{DATA_DIR}/ID{SESS}_{STIM}_units2use_stim_{STIM}.p', 'rb')))
units_idx = np.where(mask)[0] if mask.dtype == bool else mask
data = raw[:, units_idx]
data = (data - data.mean(0)) / np.where(data.std(0) == 0, 1, data.std(0))
X = data.T
chi = data_transform(X, TAU)
p = X.shape[0]; N = chi.shape[1]
SRC, TGT = 2 * TAU, 2 * TAU + 1

d = pkl.load(open(f'{OUT}/motif_population_v2_records.pkl', 'rb'))
adj = d['adj']
g2l = {int(g): i for i, g in enumerate(units_idx)}


def loc(gid):
    return g2l[gid]


def var(gid, slot):
    return chi[slot * p + loc(gid), :]


def resid(y, Zcols):
    Z = np.column_stack([np.ones(len(y))] + list(Zcols))
    beta = linalg.lstsq(Z, y)[0]
    return y - Z.dot(beta)


def pear(a, b):
    return stats.pearsonr(a, b)[0]


def edge_dir(a, b):
    """return (src, tgt) global ids for the directed edge that exists, preferring a->b."""
    if adj[loc(a), loc(b)]:
        return a, b
    return b, a


# ---- the two exemplar motifs (verified in the reproducible graph) ----
# sep = separating set S. Fork: S={272} is the COMMON PARENT of 359 & 400.
# Path 110<-400 - 496->497: clean isolated 4-node path (endpoints share no other
# parents), so conditioning on the two intermediates S={400,496} alone gives
# independence (|r| 0.24 -> 0.02).
FORK = dict(kind='Fork', label='Fork\n359←272→400',
            edges=[(272, 359), (272, 400)], na=(359, 400), sep=[272],
            s_txt='S = {272}  (common parent)')
PATH = dict(kind='Path', label='Path\n110←400–496→497',
            edges=[(400, 110), (496, 497), (400, 496)], na=(110, 497),
            sep=[400, 496],
            s_txt='S = {400, 496}  (the two intermediates)')
motifs = [FORK, PATH]

# Palette matched to the population-stats figure: blue = real connection/correlation,
# orange = conditional independence. Data dots are neutral grey so the two theme colours
# carry the meaning (the population figure has no scatter, so grey is the neutral choice).
C_DOT = '#8A8F96'      # neutral grey data points
C_REL = '#0072B2'      # blue: correlation / connection (edges + marginally-correlated pair)
C_IND = '#E69F00'      # orange: conditional independence (the conditioned panel)
C_IND_TX = '#b45f06'   # darker orange for the "independent" title text (matches pop. figure)
plt.rcParams.update({'font.size': 8, 'font.family': 'sans-serif',
                     'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                     'axes.linewidth': 0.6})

# Layout per motif: a horizontal node-arrow schematic on top, then an "edges" row
# and a "non-adjacent" row, each with a section header ABOVE the row. 3 columns;
# path uses 3 edges, fork 2; the non-adjacent row shows the pair correlated then
# independent once conditioned on its parents.
from matplotlib.gridspec import GridSpec

C_NODE = '#e8e8e8'; C_STROKE = '#333333'   # grey nodes (match population figure)
C_AX = '#666666'        # axis spine / label grey
# 6 sub-columns so a panel spans 2; a 3-panel row fills them, a 2-panel row centres.
NSUB = 6
SPAN3 = [(0, 2), (2, 4), (4, 6)]        # three panels across the full width
SPAN2 = [(1, 3), (3, 5)]                # two panels, centred

# rows per motif block: schematic, edges-header, edges, nonadj-header, nonadj
fig = plt.figure(figsize=(11.4, 12.6))
gs = GridSpec(10, NSUB, figure=fig,
              height_ratios=[0.46, 0.13, 1, 0.13, 1, 0.46, 0.13, 1, 0.13, 1],
              hspace=0.42, wspace=0.55, left=0.07, right=0.98, top=0.975, bottom=0.05)


def draw_motif_h(ax, kind):
    """Horizontal node-and-arrow schematic of the motif (drawn above its rows)."""
    ax.axis('off'); ax.set_aspect('equal'); ax.set_ylim(0, 1)
    R = 0.30

    def node(x, txt):
        ax.add_patch(plt.Circle((x, 0.5), R, fc=C_NODE, ec=C_STROKE, lw=1.6, zorder=3))
        ax.text(x, 0.5, txt, ha='center', va='center', fontsize=12,
                fontweight='bold', color='#222', zorder=4)

    def arrow(x0, x1):                       # blue connections (match population figure)
        u = 1 if x1 > x0 else -1
        ax.annotate('', xy=(x1 - u * R, 0.5), xytext=(x0 + u * R, 0.5),
                    arrowprops=dict(arrowstyle='-|>', color=C_REL, lw=2.4,
                                    mutation_scale=20), zorder=2)

    def line(x0, x1):                        # bidirectional (reciprocal) connection
        ax.annotate('', xy=(x1 - R, 0.5), xytext=(x0 + R, 0.5),
                    arrowprops=dict(arrowstyle='<|-|>', color=C_REL, lw=2.4,
                                    mutation_scale=20), zorder=2)

    if kind == 'Fork':                       # 359 <- 272 -> 400
        ax.set_xlim(0, 4.0)
        node(2.0, '272'); node(0.8, '359'); node(3.2, '400')
        arrow(2.0, 0.8); arrow(2.0, 3.2)
    else:                                    # 110 <- 400 - 496 -> 497
        ax.set_xlim(0, 5.2)
        node(0.8, '110'); node(2.0, '400'); node(3.2, '496'); node(4.4, '497')
        arrow(2.0, 0.8); arrow(3.2, 4.4); line(2.0, 3.2)


def sc(ax, x, y, r, title, xlab, ylab, color=C_REL, tcolor='#333'):
    ax.scatter(x, y, s=6, c=C_DOT, alpha=0.35, edgecolors='none', rasterized=True)
    b = np.polyfit(x, y, 1)
    xg = np.linspace(x.min(), x.max(), 100)
    yhat = b[0] * xg + b[1]
    # 95% confidence band for the regression line
    n = len(x); xbar = x.mean(); Sxx = float(np.sum((x - xbar) ** 2))
    sse = float(np.sum((y - (b[0] * x + b[1])) ** 2))
    s_err = np.sqrt(sse / (n - 2)) if n > 2 else 0.0
    tval = stats.t.ppf(0.975, n - 2)
    se_mean = s_err * np.sqrt(1.0 / n + (xg - xbar) ** 2 / Sxx)
    ax.fill_between(xg, yhat - tval * se_mean, yhat + tval * se_mean,
                    color=color, alpha=0.22, lw=0, zorder=2)
    ax.plot(xg, yhat, color=color, lw=2.4, zorder=3)
    ax.text(0.06, 0.93, f'$r$ = {r:.2f}', transform=ax.transAxes, fontsize=12.5,
            fontweight='bold', va='top', color='#222')
    ax.set_title(title, fontsize=12, pad=5, color=tcolor)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(xlab, fontsize=11.5, labelpad=2, color='#333')
    ax.set_ylabel(ylab, fontsize=11.5, labelpad=2, color='#333')
    ax.spines[['top', 'right']].set_visible(False)      # open axes (Nature style)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_edgecolor(C_AX); ax.spines[sp].set_linewidth(1.0)


def section_header(row, span, txt):
    """Bold section label centred over the panels of its row (span = (c0, c1))."""
    hax = fig.add_subplot(gs[row, span[0]:span[1]]); hax.axis('off')
    hax.text(0.5, 0.0, txt, ha='center', va='bottom', transform=hax.transAxes,
             fontsize=14, fontweight='bold', color='#111', clip_on=False)


for blk, m in enumerate(motifs):
    base = blk * 5
    # schematic strip (spans full width); the diagram itself identifies the motif
    schem = fig.add_subplot(gs[base, :])
    draw_motif_h(schem, m['kind'])
    condstr = ', '.join(str(g) for g in m['sep'])          # actual conditioning neurons (no "S")
    # ---- edges row (header above) ----
    n_edges = len(m['edges'])
    e_spans = SPAN3 if n_edges == 3 else SPAN2
    section_header(base + 1, (e_spans[0][0], e_spans[-1][1]), 'Edges')
    for c in range(n_edges):
        a, b = m['edges'][c]
        s, t = edge_dir(a, b)
        xs_, ys_ = var(s, SRC), var(t, TGT)
        ax = fig.add_subplot(gs[base + 2, e_spans[c][0]:e_spans[c][1]])
        sc(ax, xs_, ys_, pear(xs_, ys_), f'{s} → {t}',
           xlab=f'{s}$_{{t-1}}$', ylab=f'{t}$_{{t}}$', color=C_REL, tcolor=C_REL)
    # ---- non-adjacent row (header above): correlated, then independent ----
    section_header(base + 3, (SPAN2[0][0], SPAN2[-1][1]), 'Non-adjacent')
    na1, na2 = m['na']
    x3, y3 = var(na1, SRC), var(na2, TGT)
    r_marg = pear(x3, y3)
    ax_corr = fig.add_subplot(gs[base + 4, SPAN2[0][0]:SPAN2[0][1]])
    sc(ax_corr, x3, y3, r_marg, f'{na1} & {na2} correlated',
       xlab=f'{na1}$_{{t-1}}$', ylab=f'{na2}$_{{t}}$', color=C_REL, tcolor=C_REL)
    # same pair conditioned on its parents (at t-1): residuals on them; r = partial correlation
    Zcols = [var(g, SRC) for g in m['sep']]
    rx, ry = resid(x3, Zcols), resid(y3, Zcols)
    r_cond = pear(rx, ry)
    # conditioning term carries the t-1 subscript (parents are taken at the source slot)
    cond_lab = (f'{m["sep"][0]}$_{{t-1}}$' if len(m['sep']) == 1
                else f'({condstr})$_{{t-1}}$')
    ax_cond = fig.add_subplot(gs[base + 4, SPAN2[1][0]:SPAN2[1][1]])
    sc(ax_cond, rx, ry, r_cond, f'{na1} $\\perp$ {na2}  |  {condstr}',
       xlab=f'{na1}$_{{t-1}}$ | {cond_lab}', ylab=f'{na2}$_{{t}}$ | {cond_lab}',
       color=C_IND, tcolor=C_IND_TX)
    print(f"{m['kind']}: {na1}-{na2} marginal r={r_marg:.3f} -> partial r={r_cond:.3f} "
          f"| cond on {m['sep']}")

fig.text(0.5, 0.008,
         'Numbers are neuron IDs. Lagged neuron activity: source at $t\\!-\\!1$ (x) vs target at $t$ (y).   '
         'The conditioned panel shows residuals after regressing each neuron on its listed parents; '
         '$r$ is then the partial correlation.   '
         '$\\perp$ denotes conditional independence (correlation gone once the parents are accounted for).',
         ha='center', fontsize=8, color='#444')
fig.savefig(f'{OUT}/motif_examples_scatter_v2.png', dpi=200, bbox_inches='tight')
fig.savefig(f'{OUT}/motif_examples_scatter_v2.pdf', bbox_inches='tight')
print('saved', f'{OUT}/motif_examples_scatter_v2.png')
