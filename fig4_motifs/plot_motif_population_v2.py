#!/usr/bin/env python3
"""POPULATION (stats) version of the exemplar motif plots.

Aggregate over ALL instances of each motif type in the reproducible graph, showing
the same per-pair-role bar structure as the worked examples. Four motif types:

  FORK  A<-C->B   edges C->A, C->B stay dependent; A-B non-adjacent DROPS -> independent
  CHAIN A->B->C   edges A->B, B->C stay dependent; A-C non-adjacent DROPS -> independent
  PATH  A<-P1-P2->B  edges stay dependent; A-B non-adjacent DROPS -> independent
  COLLIDER A->C<-B  REVERSE: parents A,B marginally INDEPENDENT (below cutoff), and
                    conditioning on the shared child C INDUCES dependence -> RISES above
                    cutoff (Berkson). Edges A->C, B->C stay dependent.

Lagged partial correlations (source t-1 vs target t), computed exactly as
run_motif_population_v2.py: edges conditioned on the target's other inferred inputs,
the non-adjacent role on its recorded separating set, the collider parent-pair on the
shared child C."""
import sys, pickle as pkl
from itertools import combinations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.patches import Patch

sys.path.insert(0, '/home/rbiswas1/repos/cits')
from cits.methods import data_transform, partial_corr

DATA_DIR = '/home/rbiswas1/citsproject/data'
OUT = '/home/rbiswas1/microns/CITS_manuscript/figures'
SESS, STIM, BIN, IDX, TAU, ALPHA = 791319847, 'natural_scenes', 0.01, 0, 1, 0.05
MARG_DEP = 0.1

raw = np.asarray(pkl.load(open(f'{DATA_DIR}/ID{SESS}_{STIM}_bin_{BIN}_X_idx-{IDX}.p', 'rb')), float)
mask = np.asarray(pkl.load(open(f'{DATA_DIR}/ID{SESS}_{STIM}_units2use_stim_{STIM}.p', 'rb')))
units_idx = np.where(mask)[0] if mask.dtype == bool else mask
data = raw[:, units_idx]
data = (data - data.mean(0)) / np.where(data.std(0) == 0, 1, data.std(0))
X = data.T
chi = data_transform(X, TAU)
p = X.shape[0]; N = chi.shape[1]
SRC_SLOT, TGT_SLOT = 2 * TAU, 2 * TAU + 1
r_thr = np.tanh(1.959963985 / np.sqrt(N - 3))   # alpha=0.05 |r| cutoff (~0.11)

d = pkl.load(open(f'{OUT}/motif_population_v2_records.pkl', 'rb'))
adj = d['adj']; recs = d['records']
g2l = {int(g): i for i, g in enumerate(units_idx)}
sepset = {}
for r in recs:
    if r['is_edge'] == 0:
        sepset[(r['u_local'], r['v_local'])] = tuple(g2l[s] for s in r['sepset_global'])
parents_of = {v: [u for u in range(p) if adj[u, v]] for v in range(p)}

def row(v, slot):
    return slot * p + v

def lagged_pcorr(u, v, cond_neurons):
    A, B = row(u, SRC_SLOT), row(v, TGT_SLOT)
    S = set(row(c, SRC_SLOT) for c in cond_neurons)
    return abs(partial_corr(A, B, S, chi))

def cond_on_parents(u, v):
    return lagged_pcorr(u, v, [w for w in parents_of[v] if w != u])

_pe = {}
def pair_eval(x, y):
    if (x, y) in _pe:
        return _pe[(x, y)]
    marg = lagged_pcorr(x, y, [])
    S = sepset.get((x, y), None)
    cond = cond_on_parents(x, y) if S is None else lagged_pcorr(x, y, S)
    _pe[(x, y)] = (marg, cond)
    return _pe[(x, y)]

def nonadj_role(a, b):
    ma, ca = pair_eval(a, b); mb, cb = pair_eval(b, a)
    return max(ma, mb), max(ca, cb)

def edge_undirected(u, v):
    cands = []
    if adj[u, v]:
        cands.append(pair_eval(u, v))
    if adj[v, u]:
        cands.append(pair_eval(v, u))
    return max(cands, key=lambda t: t[0])

def adjacent(a, b):
    return adj[a, b] == 1 or adj[b, a] == 1

def collect(role_lists):
    """For each role: (% instances dependent unconditionally, % dependent conditionally, n).
    Dependent = |lagged partial r| >= CI cutoff (p <= 0.05)."""
    out = {}
    for k, v in role_lists.items():
        n = len(v)
        if n == 0:
            out[k] = (0.0, 0.0, 0)
            continue
        pct_marg = 100.0 * np.mean([1.0 if m >= r_thr else 0.0 for m, c in v])
        pct_cond = 100.0 * np.mean([1.0 if c >= r_thr else 0.0 for m, c in v])
        out[k] = (float(pct_marg), float(pct_cond), n)
    return out

# ---- FORK -----------------------------------------------------------------
fork_roles = {'C_A': [], 'C_B': [], 'A_B': []}
for C in range(p):
    ch = [w for w in range(p) if adj[C, w]]
    for A, B in combinations(ch, 2):
        if adjacent(A, B):
            continue
        m_ab, c_ab = nonadj_role(A, B)
        if m_ab <= MARG_DEP:
            continue
        fork_roles['C_A'].append(pair_eval(C, A))
        fork_roles['C_B'].append(pair_eval(C, B))
        fork_roles['A_B'].append((m_ab, c_ab))

# ---- CHAIN ----------------------------------------------------------------
chain_roles = {'A_B': [], 'B_C': [], 'A_C': []}
for B in range(p):
    pa = [w for w in range(p) if adj[w, B]]
    ch = [w for w in range(p) if adj[B, w]]
    for A in pa:
        for C in ch:
            if A == C or adjacent(A, C):
                continue
            m_ac, c_ac = nonadj_role(A, C)
            if m_ac <= MARG_DEP:
                continue
            chain_roles['A_B'].append(pair_eval(A, B))
            chain_roles['B_C'].append(pair_eval(B, C))
            chain_roles['A_C'].append((m_ac, c_ac))

# ---- PATH -----------------------------------------------------------------
path_roles = {'P1_A': [], 'P2_B': [], 'P1_P2': [], 'A_B': []}
adj_pairs = [(i, j) for i in range(p) for j in range(i + 1, p) if adjacent(i, j)]
for P1, P2 in adj_pairs:
    chA = [w for w in range(p) if adj[P1, w] and w not in (P1, P2)]
    chB = [w for w in range(p) if adj[P2, w] and w not in (P1, P2)]
    for A in chA:
        for B in chB:
            if A == B or adjacent(A, B):
                continue
            m_ab, c_ab = nonadj_role(A, B)
            if m_ab <= MARG_DEP:
                continue
            path_roles['P1_A'].append(pair_eval(P1, A))
            path_roles['P2_B'].append(pair_eval(P2, B))
            path_roles['P1_P2'].append(edge_undirected(P1, P2))
            path_roles['A_B'].append((m_ab, c_ab))

# ---- COLLIDER  A->C<-B  (parents A,B non-adjacent) ------------------------
# Berkson check: genuine colliders = parents removed at order 0 (marginally
# independent). The parents A,B both feed child C, so the A-B pair is same-slot
# (both at t-1). Marginal = corr(A(t-1),B(t-1)); conditional = given the shared
# child C at t. Textbook Berkson predicts conditioning INDUCES dependence.
coll_roles = {'A_C': [], 'B_C': [], 'A_B': []}   # A_B stores (marginal, conditional-given-child-C)
for C in range(p):
    pa = [w for w in range(p) if adj[w, C]]
    for A, B in combinations(pa, 2):
        if adjacent(A, B):
            continue
        # keep genuine colliders: parents removed at order 0 (marginally independent)
        Sab = sepset.get((A, B), None); Sba = sepset.get((B, A), None)
        if not (Sab is not None and len(Sab) == 0 and Sba is not None and len(Sba) == 0):
            continue
        marg = abs(partial_corr(row(A, SRC_SLOT), row(B, SRC_SLOT), set(), chi))
        condC = abs(partial_corr(row(A, SRC_SLOT), row(B, SRC_SLOT), {row(C, TGT_SLOT)}, chi))
        coll_roles['A_C'].append(pair_eval(A, C))
        coll_roles['B_C'].append(pair_eval(B, C))
        coll_roles['A_B'].append((marg, condC))

fork_m = collect(fork_roles); chain_m = collect(chain_roles)
path_m = collect(path_roles); coll_m = collect(coll_roles)

# ---------------------------------------------------------------------------
# palette (colorblind-safe blue-vs-orange axis)
# ---------------------------------------------------------------------------
# colorblind-safe (Wong) palette. Bar color encodes the MEASURED relation:
#   correlated (dependent)  -> blue   (matches: a direct A-B edge exists)
#   uncorrelated (independent) -> orange (matches: no direct A-B edge)
C_CORR = '#0072B2'     # correlated / dependent
C_UNCORR = '#E69F00'   # uncorrelated / independent (bars)
C_UNCORR_TX = '#b45f06'  # darker orange for text legibility on white
C_EDGE = C_CORR        # schematic edges drawn in the correlated color
C_MARG = C_CORR        # (detail figure aliases)
C_COND = C_UNCORR
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.family': 'sans-serif',
                     'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                     'axes.linewidth': 0.7})


def pct(lst):
    if not lst:
        return (0.0, 0.0, 0)
    m = 100.0 * np.mean([1.0 if a >= r_thr else 0.0 for a, _ in lst])
    c = 100.0 * np.mean([1.0 if b >= r_thr else 0.0 for _, b in lst])
    return float(m), float(c), len(lst)


# ===========================================================================
# MAIN (striking) figure: adjacency IS the signature of conditional dependence
#   Adjacent (edges) stay dependent given S; non-adjacent pairs become
#   conditionally independent given S. Pools all fork/chain/path/collider roles.
#
# Figure standards applied (Nature Neuroscience quality):
#   - Open axes: top/right spines removed explicitly; spine linewidth 0.8pt
#   - Font: Arial/Helvetica sans-serif (set in rcParams above)
#   - Axis label: 14pt; tick labels: 12pt; legend: 12pt; bar annotations: 12pt
#   - Colors: #4c78a8 blue (unconditional), #f58518 orange (conditional given S)
#     #333333 dark gray for neutral labels; #b34700 dark orange for independence
#   - Verdicts: one per category, same baseline via blended transform, clip_on=False
#   - Thin arc arrow with "97% -> 3%" label highlights the key cond-indep drop
#   - Legend upper-right fills the collider group empty space intentionally
#   - y-limit 108 (tighter headroom); DPI 300; facecolor white; bbox_inches tight
# ===========================================================================
edges = (fork_roles['C_A'] + fork_roles['C_B'] + chain_roles['A_B'] + chain_roles['B_C']
         + path_roles['P1_A'] + path_roles['P2_B'] + path_roles['P1_P2']
         + coll_roles['A_C'] + coll_roles['B_C'])
nonadj = fork_roles['A_B'] + chain_roles['A_C'] + path_roles['A_B']
# collider parents: their separating set is the EMPTY set (removed at order 0),
# so 'conditional given S' == the marginal test -> use (marg, marg)
coll_par = [(m, m) for (m, _c) in coll_roles['A_B']]

cats = [('Adjacent\n(direct edges)', pct(edges), 'adj'),
        ('Non-adjacent\n(common cause / chain / path)', pct(nonadj), 'non'),
        ('Non-adjacent\n(collider parents)', pct(coll_par), 'coll')]

from matplotlib.patches import Circle, FancyArrowPatch


def _node(axx, xx, yy, label, r=0.11, fc='#e8e8e8', fs=11):
    axx.add_patch(Circle((xx, yy), r, facecolor=fc, edgecolor='#333333', lw=1.0, zorder=3))
    axx.text(xx, yy, label, ha='center', va='center', fontsize=fs, fontweight='bold',
             color='#222222', zorder=4)


def _arrow(axx, p1, p2, r=0.11, color='#333333', ms=11, lw=1.4):
    d = np.array(p2, float) - np.array(p1, float); L = np.hypot(*d); u = d / L
    a = np.array(p1, float) + u * r; b = np.array(p2, float) - u * r
    axx.add_patch(FancyArrowPatch(tuple(a), tuple(b), arrowstyle='-|>',
                                  mutation_scale=ms, lw=lw, color=color, zorder=2))


fig = plt.figure(figsize=(8.4, 6.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 2.05], hspace=0.32, wspace=0.10,
                      top=0.84, bottom=0.13)
top = [fig.add_subplot(gs[0, j]) for j in range(3)]
ax = fig.add_subplot(gs[1, :])

# ---- top tier: motif schematics (a/b/c) + dependence relation ----
for t in top:
    t.set_xlim(0, 1); t.set_ylim(0, 1); t.axis('off')
# a  adjacent: A and B directly connected -- edge drawn in the correlated color
_node(top[0], 0.30, 0.62, 'A'); _node(top[0], 0.70, 0.62, 'B')
_arrow(top[0], (0.30, 0.62), (0.70, 0.62), color=C_EDGE, lw=2.0, ms=14)
top[0].text(0.5, 0.20, 'Adjacent', ha='center', va='center', fontsize=10.5,
            fontweight='bold', color='#333333')
top[0].text(0.5, 0.04, 'A, B directly connected', ha='center', va='center',
            fontsize=9.5, fontweight='bold', color=C_CORR)
# b  non-adjacent: THREE representatives, each drawn horizontally on its own row
#    (A, B are the endpoints; real edges in blue; NO direct A-B edge)
rm, fm, _ma = 0.045, 7, 8
# common cause  A <- C -> B
_node(top[1], 0.10, 0.86, 'A', rm, fs=fm); _node(top[1], 0.35, 0.86, 'C', rm, fs=fm)
_node(top[1], 0.60, 0.86, 'B', rm, fs=fm)
_arrow(top[1], (0.35, 0.86), (0.10, 0.86), r=rm, color=C_EDGE, ms=_ma, lw=1.3)
_arrow(top[1], (0.35, 0.86), (0.60, 0.86), r=rm, color=C_EDGE, ms=_ma, lw=1.3)
top[1].text(0.90, 0.86, 'common\ncause', ha='center', va='center', fontsize=7, color='#333333')
# chain  A -> C -> B
_node(top[1], 0.10, 0.60, 'A', rm, fs=fm); _node(top[1], 0.35, 0.60, 'C', rm, fs=fm)
_node(top[1], 0.60, 0.60, 'B', rm, fs=fm)
_arrow(top[1], (0.10, 0.60), (0.35, 0.60), r=rm, color=C_EDGE, ms=_ma, lw=1.3)
_arrow(top[1], (0.35, 0.60), (0.60, 0.60), r=rm, color=C_EDGE, ms=_ma, lw=1.3)
top[1].text(0.90, 0.60, 'chain', ha='center', va='center', fontsize=7, color='#333333')
# path  A <- C - C -> B
_node(top[1], 0.07, 0.34, 'A', rm, fs=fm); _node(top[1], 0.29, 0.34, 'C', rm, fs=fm)
_node(top[1], 0.51, 0.34, 'C', rm, fs=fm); _node(top[1], 0.73, 0.34, 'B', rm, fs=fm)
_arrow(top[1], (0.29, 0.34), (0.07, 0.34), r=rm, color=C_EDGE, ms=_ma, lw=1.3)
top[1].plot([0.29 + rm, 0.51 - rm], [0.34, 0.34], color=C_EDGE, lw=1.3, zorder=2)
_arrow(top[1], (0.51, 0.34), (0.73, 0.34), r=rm, color=C_EDGE, ms=_ma, lw=1.3)
top[1].text(0.90, 0.34, 'path', ha='center', va='center', fontsize=7, color='#333333')
top[1].text(0.5, 0.10, 'Non-adjacent', ha='center', va='center', fontsize=10.5,
            fontweight='bold', color='#333333')
top[1].text(0.5, -0.04, 'A, B: no direct connection', ha='center', va='center',
            fontsize=9.5, fontweight='bold', color=C_UNCORR_TX)
# c  collider: A, B not connected, both point to C
_node(top[2], 0.5, 0.44, 'C'); _node(top[2], 0.3, 0.82, 'A'); _node(top[2], 0.7, 0.82, 'B')
_arrow(top[2], (0.3, 0.82), (0.5, 0.44), color=C_EDGE, lw=1.6, ms=12)
_arrow(top[2], (0.7, 0.82), (0.5, 0.44), color=C_EDGE, lw=1.6, ms=12)
top[2].text(0.5, 0.22, 'Non-adjacent', ha='center', va='center', fontsize=10.5,
            fontweight='bold', color='#333333')
top[2].text(0.5, 0.10, '(collider)', ha='center', va='center', fontsize=9,
            color='#333333')
top[2].text(0.5, -0.04, 'A, B: no direct connection', ha='center', va='center',
            fontsize=9.5, fontweight='bold', color=C_UNCORR_TX)
for j, letter in enumerate(['a', 'b', 'c']):
    top[j].text(-0.02, 1.06, letter, transform=top[j].transAxes,
                fontsize=14, fontweight='bold', va='top', ha='left')

# ---- bottom tier: bars ----
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.8); ax.spines['bottom'].set_linewidth(0.8)
x = np.arange(len(cats)); w = 0.34; off = 0.19
# EVERY group shows the same two tests, so they are directly comparable:
#   bar1 (blue)   = % dependent, unconditional (marginal correlation)
#   bar2 (orange) = % independent, given S      (= 100 - % dependent given S)
# The orange bar is the discriminator: ~1% for adjacent (cannot be separated ->
# connected) vs ~97-98% for non-adjacent / collider (separable -> not connected).
for i, (lab, (mv, cv, n), kind) in enumerate(cats):
    v1 = 100 - mv      # % independent, unconditional
    v2 = 100 - cv      # % independent, given S
    bc = C_CORR if kind == 'adj' else C_UNCORR   # color = truth: connected / not connected
    ax.bar(x[i] - off, v1, w, color=bc, zorder=3, linewidth=0)
    ax.bar(x[i] + off, v2, w, color=bc, zorder=3, linewidth=0)
    ax.text(x[i] - off, v1 + 2.0, f'{v1:.0f}%', ha='center', fontsize=11,
            fontweight='bold', color='#333333')
    ax.text(x[i] + off, v2 + 2.0, f'{v2:.0f}%', ha='center', fontsize=11,
            fontweight='bold', color='#333333')

# interpretation lines placed under the title (all bars are tall, so no in-plot room)

# -- per-bar relation labels at the base: unconditional under blue, conditional under orange
from matplotlib.transforms import blended_transform_factory
_tb = blended_transform_factory(ax.transData, ax.transAxes)
# both bars measure % independent; x-labels state the relation tested (marginal vs given S)
for i in range(len(cats)):
    ax.text(x[i] - off, -0.05, r'$A \perp B$', transform=_tb, ha='center', va='top',
            fontsize=10, color='#333333', clip_on=False)
    ax.text(x[i] + off, -0.05, r'$A \perp B \mid S$', transform=_tb, ha='center', va='top',
            fontsize=10, color='#333333', clip_on=False)

ax.set_xticks([])
ax.set_xlim(-0.5, 2.5)
ax.tick_params(axis='y', labelsize=11, length=4, width=0.8)
ax.set_ylim(0, 108); ax.set_yticks([0, 25, 50, 75, 100])
ax.set_ylabel(r'% of pairs independent ($A \perp B$)', fontsize=12)

fig.suptitle('Adjacent pairs stay dependent; non-adjacent pairs are conditionally independent',
             fontsize=12.5, fontweight='bold', x=0.5, y=0.985)
fig.text(0.5, 0.945, 'stays dependent given any S  $\\rightarrow$  A, B directly connected',
         ha='center', fontsize=10.5, fontweight='bold', color=C_CORR)
fig.text(0.5, 0.917, 'independent given some S  $\\rightarrow$  A, B not connected',
         ha='center', fontsize=10.5, fontweight='bold', color=C_UNCORR_TX)
n_edges, n_nonadj, n_coll = cats[0][1][2], cats[1][1][2], cats[2][1][2]
fig.text(0.5, 0.012,
         f'n = {n_edges:,} edges;  {n_nonadj:,} non-adjacent (common cause / chain / path);  '
         f'{n_coll:,} collider parents.  S = separating set (empty for collider parents).  '
         f'Independent: lagged partial correlation not significant (p $>$ 0.05).',
         ha='center', fontsize=9, color='#555555')
fig.savefig(f'{OUT}/motif_population_v2.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(f'{OUT}/motif_population_v2.pdf', bbox_inches='tight', facecolor='white')
print(f'Figure saved to: {OUT}/motif_population_v2.png')
print('POOLED  %dep uncond -> cond (given S):')
for lab, (mv, cv, n), kind in cats:
    print(f'  {lab.replace(chr(10)," "):40s} n={n:5d}  {mv:5.1f}% -> {cv:5.1f}%')

# ===========================================================================
# DETAIL figure (per motif type) -> motif_population_detail_v2.png
# ===========================================================================
figd, axes = plt.subplots(1, 4, figsize=(14, 3.3), sharey=True,
                          gridspec_kw={'width_ratios': [3, 3, 4, 3]})
panels = [
    (axes[0], f'A   fork   A$\\leftarrow$C$\\rightarrow$B   (n={fork_m["A_B"][2]})', fork_m,
     [('C_A', 'C$\\to$A\n(edge)', 'edge'), ('C_B', 'C$\\to$B\n(edge)', 'edge'),
      ('A_B', 'A - B\n(non-adj)', 'non')]),
    (axes[1], f'B   chain   A$\\rightarrow$B$\\rightarrow$C   (n={chain_m["A_C"][2]})', chain_m,
     [('A_B', 'A$\\to$B\n(edge)', 'edge'), ('B_C', 'B$\\to$C\n(edge)', 'edge'),
      ('A_C', 'A - C\n(non-adj)', 'non')]),
    (axes[2], f'C   path   A$\\leftarrow$P1-P2$\\rightarrow$B   (n={path_m["A_B"][2]})', path_m,
     [('P1_A', 'P1$\\to$A\n(edge)', 'edge'), ('P2_B', 'P2$\\to$B\n(edge)', 'edge'),
      ('P1_P2', 'P1-P2\n(edge)', 'edge'), ('A_B', 'A - B\n(non-adj)', 'non')]),
    (axes[3], f'D   collider   A$\\rightarrow$C$\\leftarrow$B   (n={coll_m["A_B"][2]})', coll_m,
     [('A_C', 'A$\\to$C\n(edge)', 'edge'), ('B_C', 'B$\\to$C\n(edge)', 'edge'),
      ('A_B', 'A - B\n(parents)', 'coll')]),
]
for ax, title, means, roles in panels:
    x = np.arange(len(roles)); w = 0.26; off = 0.16
    for i, (key, lab, kind) in enumerate(roles):
        mv, cv, _ = means[key]
        emph = kind in ('non', 'coll')
        lblcol = '#b34700' if kind == 'non' else ('#1f6f6f' if kind == 'coll' else '#333')
        if emph:
            ax.axvspan(x[i] - 0.5, x[i] + 0.5, color='#f4f4f4', zorder=0)
        ax.bar(x[i] - off, mv, w, color=C_MARG, zorder=3)
        ax.bar(x[i] + off, cv, w, color=C_COND, zorder=3)
        ax.text(x[i] - off, mv + 1.8, f'{mv:.0f}%', ha='center', fontsize=8,
                fontweight='bold' if emph else 'normal', color=lblcol)
        ax.text(x[i] + off, cv + 1.8, f'{cv:.0f}%', ha='center', fontsize=8,
                fontweight='bold' if emph else 'normal', color=lblcol)
        if kind == 'non':
            ax.annotate('', xy=(x[i] + off, cv + 7), xytext=(x[i] - off, mv - 5),
                        arrowprops=dict(arrowstyle='->', color='#8a8a8a', lw=0.9,
                                        connectionstyle='arc3,rad=-0.32'), zorder=4)
            ax.text(x[i] + 0.06, 47, r'$A \perp B \mid S$', ha='center',
                    fontsize=8.5, style='italic', color='#b34700', zorder=5)
        if kind == 'coll':
            ax.text(x[i] + 0.02, 47, r'$A \perp B$' + '\n(non-adjacent)',
                    ha='center', va='center', fontsize=8.5, style='italic',
                    color='#1f6f6f', zorder=5)
    ax.set_xticks(x); ax.set_xticklabels([r[1] for r in roles], fontsize=8.0)
    ax.set_xlim(-0.6, len(roles) - 0.4); ax.set_ylim(0, 108)
    ax.set_title(title, fontsize=9.5, fontweight='bold', loc='left')
axes[0].set_ylabel(r'% of instances dependent (p $\leq$ 0.05)', fontsize=9.5)
handles = [Patch(color=C_MARG, label='unconditional'), Patch(color=C_COND, label='conditional (given S)')]
figd.legend(handles=handles, frameon=False, fontsize=9.5, loc='upper right',
            ncol=2, bbox_to_anchor=(0.99, 1.02))
figd.text(0.5, -0.01, "S = separating set (the neurons conditioned on).",
          ha='center', fontsize=8.0, color='#444')
figd.tight_layout(rect=[0, 0.0, 1, 0.95])
figd.savefig(f'{OUT}/motif_population_detail_v2.png', dpi=185, bbox_inches='tight')
print('saved', f'{OUT}/motif_population_detail_v2.png')
