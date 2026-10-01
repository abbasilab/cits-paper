"""
GRAY recreation of the arousal EM synaptic area-pair barplot (structure in
neutral gray to pair with the color-coded stimulus FC figures). Identical to
figures/plot_em_areapair_synapse_versionBsafe.py except EM_COLOR -> gray and a
new output path; reads the same versionBsafe EM data.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

BASE = ('/home/rbiswas1/microns/arousal_paper_overleaf/figures/'
        '2026-06-01_versionBsafe/fig2')
CSV = os.path.join(BASE, 'em_areapair_synapse_versionBsafe.csv')
NPZ = os.path.join(BASE, 'em_areapair_synapse_versionBsafe.npz')
OUT = ('/home/rbiswas1/microns/analysis/stimulus_fc/'
       'em_areapair_synapse_GRAY.png')
EM_COLOR = '#7a7a7a'  # neutral gray (structure)

WITHIN_PAIRS = [('AL', 'AL'), ('LM', 'LM'), ('RL', 'RL'), ('V1', 'V1')]
BETWEEN_GROUPS = [('LM', 'RL'), ('AL', 'LM'), ('LM', 'V1'),
                  ('AL', 'V1'), ('RL', 'V1'), ('AL', 'RL')]
BETWEEN_PAIRS = [(a, b) for g in BETWEEN_GROUPS for (a, b) in [(g[0], g[1]), (g[1], g[0])]]

df = pd.read_csv(CSV)
rec = {(r.src, r.tgt): r for r in df.itertuples(index=False)}
npz = np.load(NPZ, allow_pickle=True)
boot = npz['boot']
bidx = {p: i for i, p in enumerate(npz['pairs'])}
n_fields = int(npz['n_fields'])

MIN_TOTAL = 2000


def has(pair):
    r = rec.get(pair)
    return r is not None and r.tot >= MIN_TOTAL


def bp(pair):
    return boot[:, bidx[f'{pair[0]}->{pair[1]}']]


def two_sided_p(dist_a, dist_b):
    diff = dist_a - dist_b
    p = 2.0 * min(np.mean(diff > 0), np.mean(diff < 0))
    return max(p, 1.0 / len(diff))


def stars(p):
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else 'n.s.'


within = [p for p in WITHIN_PAIRS if has(p)]
between = [p for p in BETWEEN_PAIRS if has(p)]
nb_groups = [(a, b) for (a, b) in BETWEEN_GROUPS if has((a, b)) or has((b, a))]

within_x = np.arange(len(within), dtype=float)
gap = 0.5
cur = within_x[-1] + 1.0 + gap
bx = {}
between_x = []
for a, b in nb_groups:
    for pr in [(a, b), (b, a)]:
        if has(pr):
            bx[pr] = cur
            between_x.append(cur)
            cur += 1.0
sep_x = (within_x[-1] + between_x[0]) / 2.0
order = within + [pr for g in nb_groups for pr in [(g[0], g[1]), (g[1], g[0])] if has(pr)]
xs = list(within_x) + [bx[p] for p in order[len(within):]]

fig, ax = plt.subplots(figsize=(6.8, 4.0))
for x, pr in zip(xs, order):
    r = rec[pr]
    hatch = '' if pr[0] == pr[1] else '////'
    ax.bar(x, r.frac, width=0.72, color=EM_COLOR, edgecolor='#222222',
           linewidth=0.7, hatch=hatch, zorder=2)
    ax.errorbar(x, r.frac, yerr=[[r.frac - r.ci_lo], [r.ci_hi - r.frac]],
                fmt='none', color='#222222', capsize=3, lw=1.1, zorder=3)

ax.axvline(sep_x, color='#888888', lw=1.2, ls='--', zorder=1)
ax.set_xticks(xs)
ax.set_xticklabels([f'{a}→{b}' for a, b in order], rotation=55, ha='right',
                   fontsize=11)
ax.set_ylabel('Fraction of neuron pairs\nwith synapses', fontsize=15)
ax.tick_params(axis='y', labelsize=13)
fmt = ScalarFormatter(useMathText=True)
fmt.set_scientific(True); fmt.set_powerlimits((0, 0))
ax.yaxis.set_major_formatter(fmt)
ax.yaxis.get_offset_text().set_fontsize(11)
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlim(xs[0] - 0.7, xs[-1] + 0.7)

tr = ax.get_xaxis_transform()
ax.text(np.mean(within_x), 1.14, 'within-area', ha='center', va='bottom',
        fontsize=14, color='#333333', transform=tr)
ax.text(np.mean(between_x), 1.14, 'between-area', ha='center', va='bottom',
        fontsize=14, color='#333333', transform=tr)

ymax = max(rec[p].ci_hi for p in order)
ax.set_ylim(0, ymax * 1.32)


def bracket(x1, x2, y, h, label):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.1, color='#333')
    ax.text((x1 + x2) / 2, y + h, label, ha='center', va='bottom',
            fontsize=12, fontweight='bold', color='#333')


def group_bracket(g1l, g1r, g2l, g2r, y, h, label):
    cx1, cx2 = (g1l + g1r) / 2, (g2l + g2r) / 2
    ax.plot([g1l, g1r], [y, y], lw=1.1, color='#333', clip_on=False)
    ax.plot([cx1, cx1], [y, y + h], lw=1.1, color='#333', clip_on=False)
    ax.plot([cx1, cx2], [y + h, y + h], lw=1.1, color='#333', clip_on=False)
    ax.plot([cx2, cx2], [y, y + h], lw=1.1, color='#333', clip_on=False)
    ax.plot([g2l, g2r], [y, y], lw=1.1, color='#333', clip_on=False)
    ax.text((cx1 + cx2) / 2, y + h * 1.3, label, ha='center', va='bottom',
            fontsize=12, fontweight='bold', color='#333')


if ('AL', 'AL') in within:
    y0 = max(rec[p].ci_hi for p in within)
    hh = ymax * 0.03
    step = hh * 4
    for j, pr in enumerate([p for p in within if p != ('AL', 'AL')]):
        p = two_sided_p(bp(('AL', 'AL')), bp(pr))
        bracket(within_x[0], within_x[within.index(pr)],
                y0 + hh + j * step, hh, stars(p))

focal = [pr for pr in [('AL', 'RL'), ('RL', 'AL')] if has(pr)]
rest = [pr for pr in between if pr not in focal]
if focal and rest:
    d_focal = np.mean([bp(pr) for pr in focal], axis=0)
    d_rest = np.mean([bp(pr) for pr in rest], axis=0)
    p = two_sided_p(d_focal, d_rest)
    yb = max(rec[p2].ci_hi for p2 in between) + ymax * 0.06
    group_bracket(min(bx[pr] for pr in rest), max(bx[pr] for pr in rest),
                  min(bx[pr] for pr in focal), max(bx[pr] for pr in focal),
                  yb, ymax * 0.045, stars(p))

row_y = [-0.46, -0.58]
ax.text(xs[0] - 0.85, row_y[0], 'connected pairs', ha='right', va='center',
        fontsize=9, style='italic', color='#555', transform=tr, clip_on=False)
ax.text(xs[0] - 0.85, row_y[1], 'total pairs', ha='right', va='center',
        fontsize=9, style='italic', color='#555', transform=tr, clip_on=False)


def fmt_n(n):
    return f'{n/1e6:.1f}M' if n >= 1e6 else f'{n/1e3:.0f}k' if n >= 1e3 else f'{n}'


for x, pr in zip(xs, order):
    r = rec[pr]
    ax.text(x, row_y[0], fmt_n(r.syn), ha='center', va='center', fontsize=9,
            color='#333', transform=tr)
    ax.text(x, row_y[1], fmt_n(r.tot), ha='center', va='center', fontsize=9,
            color='#333', transform=tr)

fig.subplots_adjust(bottom=0.38, top=0.9)
plt.savefig(OUT, dpi=150, bbox_inches='tight', facecolor='white')
print(f"saved: {OUT}")
