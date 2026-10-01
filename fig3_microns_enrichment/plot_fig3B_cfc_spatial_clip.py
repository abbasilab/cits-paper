"""
plot_fig3B_cfc_spatial_clip.py

Fig 3 panel B — spatial CFC (statistically-causal functional circuitry) network
graph for the CLIP stimulus. Same style / layout / draw functions as the arousal
version (exp_analysis/figures/plot_example_fields_2d_versionBsafe.py). The ONLY
change is the FC adjacency source: instead of the arousal CSV, we use the CLIP
stimulus Version-B FC (fcB) staged per field.

Layout: 4 panels — fields 3,4 (Depth = 320 µm) and fields 5,6 (Depth = 450 µm),
s8/sc9.

Neuron ordering (CRITICAL): fcB rows/cols are over the stimulus pipeline's
per-field union-neuron set, sorted by unit ID. We reconstruct that exact ordering
via nids_for_field() (from panelA_fc_em_enrichment_versionB.py) and index
coords / areas by those IDs in the same order, so edges align with positions.
Verified: arousal field-neuron set == fcB ID set for fields 3-6, so a sort by
unit ID also aligns; we use the fcB nids ordering directly regardless.
"""

import os, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull

STAGE = ('/tmp/claude-1004/-home-rbiswas1-microns/'
         '48b8216b-5c45-4c8f-923d-dc312e0dbb46/scratchpad/panelA_stage')
FCDIR  = f'{STAGE}/fc'
NPYDIR = f'{STAGE}/npy'
OUT    = ('/home/rbiswas1/microns/CITS_manuscript/figures/'
          'final_figures_2026-08-31/fig3B_cfc_spatial_clip')

STIM = 'clip'
SESS, SCAN = 8, 9

AREA_COLORS = {
    'V1': '#0072B2',
    'LM': '#009E73',
    'AL': '#CC79A7',
    'RL': '#D55E00',
}
AREA_FULL = {
    'V1': 'Primary visual (V1)',
    'LM': 'Lateromedial (LM)',
    'AL': 'Anterolateral (AL)',
    'RL': 'Rostrolateral (RL)',
}
BETWEEN_COLOR = '#999999'
WITHIN_ALPHA  = 0.22
BETWEEN_ALPHA = 0.26
ARROW_ALPHA   = 0.55   # arrowheads read clearer than the faint edge lines
ARROW_SIZE    = 4
HULL_MIN_N    = 10
S_DOT         = 10
PAD_FRAC      = 0.03

# (field, depth_label) — fields 3,4 = 320 µm; fields 5,6 = 450 µm
PANELS = [3, 4, 5, 6]

# ── Load shared data ───────────────────────────────────────────────────────────
print('Loading shared data ...')
coords_all = pd.read_pickle('/home/rbiswas1/microns/all_unit_coords.pkl')
areas_all  = pd.read_csv('/home/rbiswas1/microns/all_unit_areas.csv')

_ids    = np.load(f'{NPYDIR}/ids_session{SESS}_scan{SCAN}.npy')
_fields = np.load(f'{NPYDIR}/fields_session{SESS}_scan{SCAN}.npy')
_union  = set(np.load(f'{NPYDIR}/unionids_session{SESS}_scan{SCAN}.npy').tolist())


def nids_for_field(field):
    """fcB per-field unit-ID ordering: union neurons in this field, sorted by ID."""
    idx = [i for i in range(len(_ids))
           if _fields[i] == field and int(_ids[i]) in _union]
    idx.sort(key=lambda i: int(_ids[i]))
    return [int(_ids[i]) for i in idx]


def load_panel(field):
    nids = nids_for_field(field)
    n    = len(nids)

    # fcB adjacency (n x n), binarized like the arousal script
    arr = np.load(f'{FCDIR}/s{SESS}sc{SCAN}f{field}_{STIM}.npz')['fcB'].astype(float)
    assert arr.shape == (n, n), f'fcB {arr.shape} != ({n},{n})'
    np.fill_diagonal(arr, 0.0)
    arr = np.nan_to_num(arr)
    arr[arr == 0] = np.nan

    # coords / areas indexed by the EXACT fcB unit-ID order
    c = coords_all[(coords_all.session == SESS) & (coords_all.scan_idx == SCAN)][
        ['unit_id', 'um_x', 'um_y']]
    a = areas_all[(areas_all.session == SESS) & (areas_all.scan_idx == SCAN)][
        ['unit_id', 'brain_area']]
    ca = c.merge(a, on='unit_id', how='left').set_index('unit_id')
    sub = ca.loc[nids]  # reindex in fcB order
    assert len(sub) == n

    xv = sub.um_x.values.astype(float)
    yv = sub.um_y.values.astype(float)
    ba = sub.brain_area.fillna('Unknown').values

    si, ti    = np.where(~np.isnan(arr))
    n_within  = sum(ba[si[k]] == ba[ti[k]] for k in range(len(si)))
    n_between = len(si) - n_within
    print(f'  s{SESS}/sc{SCAN}/f{field} {STIM}: '
          f'{n} neurons, {n_within} within, {n_between} between')
    return xv, yv, ba, arr, n, n_within, n_between


print('Loading panels ...')
panel_data = []
for field in PANELS:
    xv, yv, ba, tpc, n, nw, nb = load_panel(field)
    panel_data.append(dict(field=field, xv=xv, yv=yv, ba=ba, tpc=tpc, n=n,
                           n_within=nw, n_between=nb))

areas_present = [ar for ar in ['V1', 'LM', 'AL', 'RL']
                 if any((p['ba'] == ar).sum() > 0 for p in panel_data)]

# ── Drawing helpers (identical to base script) ──────────────────────────────────
def draw_hull(ax, xy, color):
    if len(xy) < 4: return
    try:
        hull  = ConvexHull(xy)
        patch = mpatches.Polygon(xy[hull.vertices], closed=True,
                                 facecolor='none', edgecolor=color,
                                 linewidth=1.2, linestyle='--', alpha=0.6, zorder=2)
        ax.add_patch(patch)
    except Exception:
        pass


def draw_connections(ax, xv, yv, ba, tpc):
    si, ti = np.where(~np.isnan(tpc))
    for k in range(len(si)):
        if ba[si[k]] != ba[ti[k]]:
            x1, y1 = xv[si[k]], yv[si[k]]
            x2, y2 = xv[ti[k]], yv[ti[k]]
            ax.plot([x1, x2], [y1, y2],
                    color=BETWEEN_COLOR, lw=0.33, alpha=BETWEEN_ALPHA,
                    zorder=3, solid_capstyle='round')
            # matching arrowhead near target (t=0.85), bolder than the faint line
            t = 0.85
            xm = x1 + t * (x2 - x1)
            ym = y1 + t * (y2 - y1)
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            ax.plot(xm, ym,
                    marker=(3, 0, angle - 90),
                    markersize=ARROW_SIZE, color=BETWEEN_COLOR,
                    alpha=ARROW_ALPHA, zorder=3.5,
                    markeredgewidth=0)
    for area, color in AREA_COLORS.items():
        idx = np.where(ba == area)[0]
        if len(idx) < 2: continue
        block = tpc[np.ix_(idx, idx)]
        rs, cs = np.where(~np.isnan(block))
        mask = rs != cs
        for r, c_ in zip(rs[mask], cs[mask]):
            i, j = idx[r], idx[c_]
            x1, y1, x2, y2 = xv[i], yv[i], xv[j], yv[j]
            ax.plot([x1, x2], [y1, y2],
                    color=color, lw=0.4, alpha=WITHIN_ALPHA,
                    zorder=4, solid_capstyle='round')
            t = 0.85
            xm = x1 + t * (x2 - x1)
            ym = y1 + t * (y2 - y1)
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            ax.plot(xm, ym,
                    marker=(3, 0, angle - 90),
                    markersize=ARROW_SIZE, color=color,
                    alpha=ARROW_ALPHA, zorder=5,
                    markeredgewidth=0)


def draw_neurons(ax, xv, yv, ba):
    unk = ba == 'Unknown'
    if unk.any():
        ax.scatter(xv[unk], yv[unk], s=S_DOT, color='#DDDDDD',
                   edgecolors='none', zorder=5, rasterized=True)
    for area, color in AREA_COLORS.items():
        m = ba == area
        if m.sum() > 0:
            ax.scatter(xv[m], yv[m], s=S_DOT + 4, color=color,
                       edgecolors='none', alpha=0.85, zorder=6, rasterized=True)


def draw_hulls(ax, xv, yv, ba):
    for area, color in AREA_COLORS.items():
        m = ba == area
        if m.sum() >= HULL_MIN_N:
            draw_hull(ax, np.column_stack([xv[m], yv[m]]), color)


def style_ax(ax, xv, yv, n, n_within, n_between, field, show_ylabel,
             shared_ylim=None):
    xmin, xmax = xv.min(), xv.max()
    ymin, ymax = yv.min(), yv.max()
    xpad = (xmax - xmin) * PAD_FRAC
    ypad = (ymax - ymin) * PAD_FRAC
    ax.set_xlim(xmin - xpad, xmax + xpad)
    if shared_ylim is not None:
        ax.set_ylim(shared_ylim[0], shared_ylim[1])
    else:
        ax.set_ylim(ymin - ypad, ymax + ypad)
    ax.set_aspect('equal', adjustable='box')
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_edgecolor('#AAAAAA')
    ax.spines['bottom'].set_edgecolor('#AAAAAA')
    ax.tick_params(labelsize=13, length=3, width=0.7,
                   color='#AAAAAA', labelcolor='#333333')
    ax.set_xlabel('x (µm)', fontsize=14, labelpad=4, color='#333333')
    if show_ylabel:
        ax.set_ylabel('y (µm)', fontsize=14, labelpad=4, color='#333333')
    else:
        ax.tick_params(labelleft=False)
    tile = 'left' if field % 2 == 1 else 'right'
    ax.set_title(f'Field {field} ({tile})\n{n_within:,} within-area edges\n'
                 f'{n_between:,} between-area edges',
                 fontsize=11, pad=5, color='#333333', linespacing=1.4)
    ax.text(0.97, 0.03, f'n = {n:,} units',
            transform=ax.transAxes, fontsize=12,
            va='bottom', ha='right', color='#333333', zorder=11,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2))
    sb_x0 = xmin + xpad + 20
    sb_y0 = ymin + ypad + 20
    ax.plot([sb_x0, sb_x0 + 100], [sb_y0, sb_y0],
            color='black', lw=2, solid_capstyle='butt', zorder=10)
    ax.text(sb_x0 + 50, sb_y0 + (ymax - ymin) * 0.02, '100 µm',
            ha='center', va='bottom', fontsize=12, color='black', zorder=10)


# ── Figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12.5, 7.7), facecolor='white',
                 constrained_layout=False)
# 6 cols: f3, f4, spacer, f5, f6, legend
gs = gridspec.GridSpec(1, 6, figure=fig,
                       width_ratios=[4, 4, 0.4, 4, 4, 1.2],
                       wspace=0.0,
                       left=0.08, right=0.99, top=0.70, bottom=0.12)

axes = [fig.add_subplot(gs[0]),
        fig.add_subplot(gs[1]),
        fig.add_subplot(gs[3]),
        fig.add_subplot(gs[4])]
ax_leg = fig.add_subplot(gs[5])


def pair_ylim(p1, p2):
    ymin = min(p1['yv'].min(), p2['yv'].min())
    ymax = max(p1['yv'].max(), p2['yv'].max())
    pad  = (ymax - ymin) * PAD_FRAC
    return (ymin - pad, ymax + pad)


pairs = [(0, 1), (2, 3)]
shared_ylims = [pair_ylim(panel_data[a], panel_data[b]) for a, b in pairs]
panel_ylim = {}
for (a, b), ylim in zip(pairs, shared_ylims):
    panel_ylim[a] = ylim
    panel_ylim[b] = ylim

for i, (ax, pd_) in enumerate(zip(axes, panel_data)):
    draw_connections(ax, pd_['xv'], pd_['yv'], pd_['ba'], pd_['tpc'])
    draw_neurons(ax, pd_['xv'], pd_['yv'], pd_['ba'])
    draw_hulls(ax, pd_['xv'], pd_['yv'], pd_['ba'])
    style_ax(ax, pd_['xv'], pd_['yv'], pd_['n'],
             pd_['n_within'], pd_['n_between'], pd_['field'],
             show_ylabel=(i == 0),
             shared_ylim=panel_ylim[i])

# ── Depth group labels above each pair (computed from axis positions) ───────────
fig.canvas.draw()
for (a, b), depth in zip(pairs, ['320 µm', '450 µm']):
    xa = axes[a].get_position(); xb = axes[b].get_position()
    xmid = (xa.x0 + xb.x1) / 2
    fig.text(xmid, 0.805, f'Depth = {depth}',
             ha='center', va='bottom', fontsize=14, fontweight='bold',
             color='#333333', transform=fig.transFigure)

# ── Clip stimulus indication (small, unobtrusive label) ─────────────────────────
fig.text(0.08, 0.92, 'Clip stimulus',
         ha='left', va='top', fontsize=11, color='#555555',
         transform=fig.transFigure)

# ── Legend ─────────────────────────────────────────────────────────────────────
ax_leg.set_axis_off()
handles = [
    Line2D([0], [0], color=AREA_COLORS[ar], lw=2.5, alpha=0.85,
           label=AREA_FULL[ar])
    for ar in areas_present
] + [
    Line2D([0], [0], color=BETWEEN_COLOR, lw=1.5, alpha=0.75,
           label='Between-area'),
]
ax_leg.legend(handles=handles, loc='center left',
              fontsize=13, frameon=True, framealpha=0.95,
              edgecolor='#CCCCCC', handlelength=1.6,
              labelspacing=0.8, borderpad=0.8)

fig.savefig(OUT + '.pdf', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(OUT + '.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f'\nSaved: {OUT}.pdf / .png')
