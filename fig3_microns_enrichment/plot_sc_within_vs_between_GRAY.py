"""
GRAY recreation of the arousal Fig 1 within-vs-between EM synapse plot (structure
in neutral gray to pair with the color-coded stimulus FC figures). Identical to
figures/plot_sc_within_vs_between_versionBsafe.py except EM_COLOR -> gray and a
new output path; reads the same versionBsafe data.
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from matplotlib.ticker import ScalarFormatter
from scipy.stats import fisher_exact


def scientific_yaxis(ax):
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.get_offset_text().set_fontsize(11)


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared')))
from paths import out as _out, result
# from bootstrap_em_within_vs_between_versionBsafe.py (committed copy in source_data/)
CSV = result('fig3_microns_enrichment', 'sc_within_vs_between_synapse_versionBsafe.csv')
OUT = _out('fig3_microns_enrichment', 'sc_within_vs_between_synapse_GRAY.png')

EM_COLOR = '#7a7a7a'   # neutral gray (structure)


def sig_stars(p):
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else 'n.s.'


r = pd.read_csv(CSV).iloc[0]
fw, fb = r['within_rate'], r['between_rate']
lo_w, hi_w = r['within_ci_lo'], r['within_ci_hi']
lo_b, hi_b = r['between_ci_lo'], r['between_ci_hi']
kw, Nw = int(r['w_syn']), int(r['w_tot'])
kb, Nb = int(r['b_syn']), int(r['b_tot'])
stars = sig_stars(fisher_exact([[kw, Nw - kw], [kb, Nb - kb]])[1])

xw, xb = 1.0, 1.5
fig, ax = plt.subplots(figsize=(3.4, 4.2))
ax.bar(xw, fw, width=0.36, color=EM_COLOR, edgecolor='#222222',
       linewidth=0.8, zorder=2)
ax.bar(xb, fb, width=0.36, color=EM_COLOR, edgecolor='#222222',
       linewidth=0.8, hatch='////', zorder=2)
ax.errorbar([xw, xb], [fw, fb],
            yerr=[[fw - lo_w, fb - lo_b], [hi_w - fw, hi_b - fb]],
            fmt='none', color='#222222', capsize=4, lw=1.5, zorder=3)

y_top = max(hi_w, hi_b) * 1.06
ax.plot([xw, xw, xb, xb], [y_top * 0.985, y_top, y_top, y_top * 0.985],
        lw=1.2, color='#333333')
ax.text((xw + xb) / 2, y_top * 1.02, stars, ha='center', va='bottom',
        fontsize=16, fontweight='bold', color='#333333')

ax.set_xticks([])
ax.set_xlim(0.74, 1.76)
ax.set_ylim(0, y_top * 1.18)
ax.set_ylabel('Fraction of neuron pairs\nwith a synapse', fontsize=16)
ax.tick_params(axis='y', labelsize=14)
scientific_yaxis(ax)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

blend = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
bw = 0.18
for patch_x, hatch, text, ha, text_x in [
    (xw, '',     'Within-area',  'right', xw - bw - 0.05),
    (xb, '////', 'Between-area', 'left',  xb + bw + 0.05),
]:
    rect = mpatches.Rectangle((patch_x - bw, -0.16), 2 * bw, 0.09,
                              facecolor=EM_COLOR, edgecolor='#222222',
                              linewidth=0.8, hatch=hatch, transform=blend,
                              clip_on=False)
    ax.add_patch(rect)
    ax.text(text_x, -0.115, text, ha=ha, va='center', fontsize=12,
            transform=blend, clip_on=False)

n_fields = int(r['n_fields'])
fig.text(0.5, -0.16,
         f"Ratio {r['ratio']:.1f}× [{r['ratio_ci_lo']:.1f}×, {r['ratio_ci_hi']:.1f}×], "
         f"field-cluster bootstrap 95% CI (N=5000, {n_fields} fields).\n"
         f'Within-area: {kw:,} synaptic / {Nw:,} pairs.  '
         f'Between-area: {kb:,} / {Nb:,}.',
         ha='center', fontsize=7.5, color='#555')

plt.tight_layout()
fig.subplots_adjust(bottom=0.22)
plt.savefig(OUT, dpi=150, bbox_inches='tight', facecolor='white')
print(f"saved: {OUT}")
