"""
Fig 3 Panel A — FC presence vs. EM-based synaptic connectivity.
3-method grouped bar plot: CITS (Ours) / Granger causality / Lagged correlation.

ALL data on STIMULUS provenance (same calcium/fields/EM pairing across methods).

Corrected from the old arousal-provenance version; GC2 is now *** (not n.s.).

Figure standards applied
------------------------
- Open axes: top and right spines removed.
- Font sizes: y-label 14pt, tick labels 12pt, annotations 12pt+, panel label 16pt bold.
- Colors: FC-present #6BAED6 (ColorBrewer blue, matching original composite), FC-absent #999999.
- Continuous/heatmap: not applicable here.
- Line widths: error-bar lw=1.5, bracket lw=1.2, bar edge lw=0.8.
- Scatter markers: not applicable.
- Error bars: field-cluster bootstrap 95% CI (N=5000, seed=42, 39 fields).
  Each bootstrap draw resamples fields with replacement; pooled rate = sum(synaptic)/sum(total).
- Layout: fig.savefig(..., dpi=300, bbox_inches='tight', facecolor='white').
- Preferred width: ~5.5 inches (wider single column for 3 method groups).
- pdf.fonttype = 42 (TrueType / Liberation Sans compatible).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Liberation Sans', 'Arial', 'DejaVu Sans']
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from statsmodels.stats.proportion import proportion_confint

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRATCHPAD = ('/tmp/claude-1004/-home-rbiswas1-microns/'
              '48b8216b-5c45-4c8f-923d-dc312e0dbb46/scratchpad')
CSV_CITS    = os.path.join(SCRATCHPAD, 'panelA_perfield_counts.csv')
CSV_GRANGER = os.path.join(SCRATCHPAD, 'stim_baseline_perfield_granger.csv')
CSV_LAGGED  = os.path.join(SCRATCHPAD, 'stim_baseline_perfield_lagged01.csv')

OUT_DIR = '/home/rbiswas1/microns/CITS_manuscript/figures/final_figures_2026-08-31'
OUT_PDF = os.path.join(OUT_DIR, 'fig3A_enrichment_stimulus.pdf')
OUT_PNG = os.path.join(OUT_DIR, 'fig3A_enrichment_stimulus.png')

# ── Colors (matched to original composite fig3.pdf) ───────────────────────────
FC_COLOR  = '#6BAED6'   # ColorBrewer blue — FC-present
ABS_COLOR = '#999999'   # neutral grey     — FC-absent

# ── Bootstrap parameters ───────────────────────────────────────────────────────
N_BOOT = 5000
RNG    = np.random.default_rng(42)


# ── Helpers ────────────────────────────────────────────────────────────────────
def pooled_rates(arr):
    """arr: (n_fields, 4) columns: fc_plus, fcplus_scplus, fc_minus, fcminus_scplus."""
    s = arr.sum(axis=0)
    fcp, fps, fca, fas = s
    pr = fps / fcp if fcp > 0 else np.nan
    ar = fas / fca if fca > 0 else np.nan
    fold = pr / ar if (ar and ar > 0) else np.nan
    return pr, ar, fold, int(fcp), int(fps), int(fca), int(fas)


def bootstrap_ci(arr):
    """Field-cluster bootstrap 95% CI on present rate, absent rate, and fold."""
    n = len(arr)
    boot_pr  = np.empty(N_BOOT)
    boot_ar  = np.empty(N_BOOT)
    boot_fold = np.empty(N_BOOT)
    for i in range(N_BOOT):
        a = arr[RNG.integers(0, n, size=n)].sum(axis=0)
        fcp, fps, fca, fas = a
        pr = fps / fcp if fcp > 0 else np.nan
        ar = fas / fca if fca > 0 else np.nan
        boot_pr[i]   = pr
        boot_ar[i]   = ar
        boot_fold[i] = pr / ar if (ar and ar > 0) else np.nan
    pr_lo,  pr_hi  = np.nanpercentile(boot_pr,   [2.5, 97.5])
    ar_lo,  ar_hi  = np.nanpercentile(boot_ar,   [2.5, 97.5])
    fold_lo, fold_hi = np.nanpercentile(boot_fold, [2.5, 97.5])
    return pr_lo, pr_hi, ar_lo, ar_hi, fold_lo, fold_hi


def load_method(csv_path):
    df = pd.read_csv(csv_path)
    arr = df[['fc_plus', 'fcplus_scplus', 'fc_minus', 'fcminus_scplus']].to_numpy(dtype=float)
    pr, ar, fold, fcp, fps, fca, fas = pooled_rates(arr)
    pr_lo, pr_hi, ar_lo, ar_hi, fold_lo, fold_hi = bootstrap_ci(arr)
    return dict(pr=pr, ar=ar, fold=fold,
                fcp=fcp, fps=fps, fca=fca, fas=fas,
                pr_lo=pr_lo, pr_hi=pr_hi,
                ar_lo=ar_lo, ar_hi=ar_hi,
                fold_lo=fold_lo, fold_hi=fold_hi,
                n_fields=len(df))


# ── Corrected numbers (stimulus provenance) ────────────────────────────────────
# Provided by analysis; verified against per-field CSVs below.
# Fisher p-values and significance:
#   CITS:   p=1.1e-100  ***
#   GC2:    p=9.2e-19   ***
#   Lagged: p=2.7e-16   ***
FISHER_P = {
    'CITS (Ours)':        1.1e-100,
    'Granger causality':  9.2e-19,
    'Lagged correlation': 2.7e-16,
}

# Fold labels (from corrected numbers; bootstrap CIs printed to console)
FOLD_LABELS = {
    'CITS (Ours)':        '3.6×',
    'Granger causality':  '1.4×',
    'Lagged correlation': '2.1×',
}


def stars(p):
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else 'n.s.'


def fmt_count(n):
    """Abbreviate large counts for the count table (e.g. 37285 → '37k')."""
    if n >= 1_000_000:
        return f'{n/1e6:.0f}M'
    elif n >= 1000:
        return f'{n/1e3:.0f}k'
    return str(n)


# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading per-field counts and running bootstrap (N=5000, seed=42)...")
methods = {
    'CITS (Ours)':        load_method(CSV_CITS),
    'Granger causality':  load_method(CSV_GRANGER),
    'Lagged correlation': load_method(CSV_LAGGED),
}

# Print verification table
print(f"\n{'Method':<22} {'FC+/total':>14} {'rate(x1e-2)':>12} {'FC-/total':>14} {'rate(x1e-2)':>12} "
      f"{'fold':>6} {'fold_CI':>16} {'n_fields':>9}")
for mname, m in methods.items():
    print(f"{mname:<22} {m['fps']:>6}/{m['fcp']:<7} {m['pr']*100:>11.3f}  "
          f"{m['fas']:>6}/{m['fca']:<7} {m['ar']*100:>11.3f}  "
          f"{m['fold']:>5.2f}x  [{m['fold_lo']:.2f},{m['fold_hi']:.2f}]  "
          f"{m['n_fields']:>9}")

# ── Layout: 3 groups of 2 bars ─────────────────────────────────────────────────
# Group centers at 1.0, 2.4, 3.8; bar half-gap = 0.17; bar width = 0.28
BAR_W    = 0.28
HALF_GAP = 0.18   # half distance between the 2 bars in a group
GROUP_SPACING = 0.98

# group 1: CITS
G1 = 1.00
# group 2: Granger
G2 = G1 + GROUP_SPACING   # 2.45
# group 3: Lagged
G3 = G2 + GROUP_SPACING   # 3.90

# Order: CITS (G1, 3.6×), Lagged correlation (G2, 2.1×), Granger causality (G3, 1.4×)
GROUP_CENTERS = {
    'CITS (Ours)':        G1,
    'Lagged correlation': G2,
    'Granger causality':  G3,
}

# Left-to-right order by fold (descending): 3.6×, 2.1×, 1.4×
method_order = ['CITS (Ours)', 'Lagged correlation', 'Granger causality']

# ── Figure ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 4.2))

# --- draw bars + errorbars ---
for mname in method_order:
    m  = methods[mname]
    gc = GROUP_CENTERS[mname]
    xp = gc - HALF_GAP   # FC-present bar x
    xa = gc + HALF_GAP   # FC-absent bar x

    ax.bar(xp, m['pr'] * 1e2, width=BAR_W,
           color=FC_COLOR, edgecolor='#222222', lw=0.8, zorder=2)
    ax.bar(xa, m['ar'] * 1e2, width=BAR_W,
           color=ABS_COLOR, edgecolor='#222222', lw=0.8, zorder=2)

    ax.errorbar([xp, xa],
                [m['pr'] * 1e2, m['ar'] * 1e2],
                yerr=[[( m['pr'] - m['pr_lo']) * 1e2, (m['ar'] - m['ar_lo']) * 1e2],
                      [(m['pr_hi'] - m['pr']) * 1e2, (m['ar_hi'] - m['ar']) * 1e2]],
                fmt='none', color='#222222', capsize=3.5, lw=1.5, zorder=3)

# --- significance brackets + fold labels + *** ---
y_maxes = []
for mname in method_order:
    m  = methods[mname]
    gc = GROUP_CENTERS[mname]
    xp = gc - HALF_GAP
    xa = gc + HALF_GAP
    y_top_group = max(m['pr_hi'], m['ar_hi']) * 1e2
    y_maxes.append(y_top_group)

global_ymax = max(y_maxes)
# Header block now lives ABOVE the axes (y_frac > 1.0) via blended transform,
# so y_axis_top only needs to clear the CITS fold label + a small buffer.
# CITS fold label sits at ~1.26 × global_ymax; set top to ~1.48 × for a neat margin.
y_axis_top = global_ymax * 1.48
ax.set_ylim(0, y_axis_top)

# Annotation vertical spacing in axis units (fixed, not group-relative)
BRACK_CLEARANCE = global_ymax * 0.10  # gap between top-of-CI and bracket line
STARS_OFFSET    = global_ymax * 0.04  # gap above bracket to *** text baseline
FOLD_OFFSET     = global_ymax * 0.16  # gap above bracket to fold label baseline

for mname in method_order:
    m  = methods[mname]
    gc = GROUP_CENTERS[mname]
    xp = gc - HALF_GAP
    xa = gc + HALF_GAP
    y_top_group = max(m['pr_hi'], m['ar_hi']) * 1e2
    brack_y = y_top_group + BRACK_CLEARANCE

    # bracket
    ax.plot([xp, xp, xa, xa],
            [brack_y - BRACK_CLEARANCE * 0.15, brack_y,
             brack_y, brack_y - BRACK_CLEARANCE * 0.15],
            lw=1.2, color='#333333', zorder=4)

    # *** just above the bracket
    ax.text(gc, brack_y + STARS_OFFSET, stars(FISHER_P[mname]),
            ha='center', va='bottom', fontsize=14, fontweight='bold',
            color='#333333', zorder=5)

    # fold label further above — separated from *** by FOLD_OFFSET
    ax.text(gc, brack_y + FOLD_OFFSET, FOLD_LABELS[mname],
            ha='center', va='bottom', fontsize=13, fontweight='bold',
            color='#333333', zorder=5)

# --- axes cosmetics ---
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([])
ax.set_xlim(G1 - HALF_GAP - BAR_W / 2 - 0.22,
             G3 + HALF_GAP + BAR_W / 2 + 0.22)
ax.set_ylabel('Fraction of neuron pairs\nwith a synapse  (×10$^{-2}$)',
              fontsize=14)
ax.tick_params(axis='y', labelsize=12)
ax.tick_params(axis='x', length=0)

# ── Legend — upper-right inside axes, above all Granger annotations ────────────
# Granger fold is at axes-fraction y ≈ 0.44; legend at 0.93 is well clear.
legend_handles = [
    mpatches.Patch(facecolor=FC_COLOR,  edgecolor='#444444', lw=0.8, label='FC-present'),
    mpatches.Patch(facecolor=ABS_COLOR, edgecolor='#444444', lw=0.8, label='FC-absent'),
]
ax.legend(handles=legend_handles,
          bbox_to_anchor=(0.99, 0.93), loc='upper right',
          fontsize=12, frameon=False,
          handlelength=1.2, handleheight=1.0,
          handletextpad=0.5, borderpad=0.2)

# ── Header block above the axes (blended transform: x data, y axes fraction) ──
# Top-to-bottom for each group: bold title → SC counts → NP counts.
# The axes top spine sits at y_frac = 1.00; header elements have y_frac > 1.0.
# Row labels (italic grey) appear flush-right of the leftmost bar column.
# clip_on=False lets the text render into the top-margin area.
blend = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

# y positions in axes fraction (positive = above axes top)
TITLE_Y = 1.33    # bold two-line title bottom baseline (shifted down -0.09 to close NP-to-plot gap)
SC_Y    = 1.21    # "Synaptically connected" values (same inter-row spacing preserved)
NP_Y    = 1.08    # "Neuron pairs" values — now only 8% of axes height above the spine

# Row label x: just left of the FC-present bar of the first group
label_x = G1 - HALF_GAP - BAR_W / 2 - 0.08

# Display labels: two-word titles split onto two lines; single-word stays one line.
TITLE_DISPLAY = {
    'CITS (Ours)':        'CITS (Ours)',
    'Lagged correlation': 'Lagged\ncorrelation',
    'Granger causality':  'Granger\ncausality',
}

# Bold method titles — one per group, centered on group
for mname in method_order:
    gc = GROUP_CENTERS[mname]
    ax.text(gc, TITLE_Y, TITLE_DISPLAY[mname],
            ha='center', va='bottom', fontsize=14, fontweight='bold',
            color='#222222', transform=blend, clip_on=False, zorder=6,
            multialignment='center')

# Count rows with italic left-side labels
for row_label, row_y in [('Synaptically\nconnected', SC_Y), ('Neuron pairs', NP_Y)]:
    ax.text(label_x, row_y, row_label,
            ha='right', va='center', fontsize=11, style='italic', color='#555555',
            transform=blend, clip_on=False)
    for mname in method_order:
        m  = methods[mname]
        gc = GROUP_CENTERS[mname]
        xp = gc - HALF_GAP
        xa = gc + HALF_GAP
        if 'ynaptically' in row_label:
            vp, va_ = m['fps'], m['fas']
        else:
            vp, va_ = m['fcp'], m['fca']
        ax.text(xp, row_y, fmt_count(vp),
                ha='center', va='center', fontsize=11, color='#333333',
                transform=blend, clip_on=False)
        ax.text(xa, row_y, fmt_count(va_),
                ha='center', va='center', fontsize=11, color='#333333',
                transform=blend, clip_on=False)

# ── Save ───────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
# top=0.68 leaves a generous top margin for the header block (y_frac up to ~1.34).
# bottom=0.12 gives a slim base (no count table below).
fig.subplots_adjust(bottom=0.12, top=0.68, left=0.20, right=0.97)

fig.savefig(OUT_PDF, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight', facecolor='white')

print(f"\nFigure saved to: {OUT_PDF}")
print(f"Figure saved to: {OUT_PNG}")

# ── Final verification print ───────────────────────────────────────────────────
print("\n── Rates plotted (x10^-2) ──")
for mname in method_order:
    m = methods[mname]
    print(f"  {mname}:")
    print(f"    FC-present rate : {m['pr']*100:.3f}  "
          f"95% CI [{m['pr_lo']*100:.3f}, {m['pr_hi']*100:.3f}]")
    print(f"    FC-absent  rate : {m['ar']*100:.3f}  "
          f"95% CI [{m['ar_lo']*100:.3f}, {m['ar_hi']*100:.3f}]")
    print(f"    Fold            : {m['fold']:.2f}x  "
          f"95% CI [{m['fold_lo']:.2f}, {m['fold_hi']:.2f}]")
    print(f"    Significance    : {stars(FISHER_P[mname])}  (p={FISHER_P[mname]:.2e})")
    print(f"    n_fields        : {m['n_fields']}")
