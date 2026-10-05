#!/usr/bin/env python3
"""
regen_cfc_stimtypes.py -- faithful regeneration of Fig. `stimtypegraphs`
(cfc_stimtypes.pdf).

The original composite was NEVER produced by a single cell.  script_matchbarplot.ipynb
emitted ONE graph per stimulus (cell 24, figsize=(12,10)) and ONE bar chart per
broad region (cell 28, figsize=(max(10, n_bars*0.3),6)); the published composite
was assembled by hand from those native-size panels.  Earlier versions of this
script redrew everything into a shared gridspec, which shrank the graph axes and
made node_size=250 look oversized (circle-dim mismatch) and forced seaborn bar
widths that did not match the notebook geometry (bar-dim mismatch).

This version reproduces the ORIGINAL workflow: it renders each panel at its native
size with the VERBATIM notebook code/parameters, then pastes the panels onto a
canvas (PIL) -- so every circle and bar keeps the exact proportions the notebook
produced.

Verbatim from script_matchbarplot.ipynb
---------------------------------------
  Panel A (cell 24): get_clustered_circle_positions_grouped(min_spacing=6,
     base_cluster_radius=3.0); color_group_map (Blues/Reds/Greens/Purples shaded
     0.3->0.9); figsize=(12,10); node_size=250; node-index font_size=12;
     edge width = 2*interp(|w|,[gmin,gmax],[0.1,1.0]); edge alpha = 0.3+0.7*stab;
     edge_color='gray'; connectionstyle='arc3,rad=0.2'.
  Panel B (cell 28): bar_width=0.2; group_spacing=1.0; figsize=(max(10,n*0.3),6);
     Percentage = 100*Count/source_count**2; regions with >5 neurons only;
     stim_color_map natural_scenes #ff7f0e, static_gratings #2ca02c, gabors #d62728;
     ylabel '% of Possible Neuron Pairs'; legend lw=6.

Reproducible-CITS substitution (unchanged from before): the per-block graph is the
reproducible fast_cits_pcorr graph aggregated across all 60 blocks
(stability = fraction of blocks an edge is inferred; avg_strength = mean |signed
conditional partial-corr weight|; edges with stability < 0.8 dropped).  The weight
is lagged partial-correlation r, so the edge-thickness legend is auto-scaled and
labelled with the actual r values.

Output: figures/cfc_stimtypes_v2.pdf  (original NOT overwritten)
(now under $CITS_PAPER_OUT/fig5_neuropixels/, see shared/paths.py)
"""
import os, sys, subprocess
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import networkx as nx
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared')))
from paths import outdir as _outdir, result, neuropixels
def _union(f): return result('fig5_neuropixels', f, fallback=neuropixels(f))   # cits_v2_union_*.npy: $CITS_PAPER_OUT/fig5_neuropixels/ if present, else $NEUROPIXELS_DATA
FIG = _outdir('fig5_neuropixels')                 # was CITS_manuscript/figures (inputs from _stimtypes_90swin_compute.py + outputs)
SCRATCH = _outdir('fig5_neuropixels/panels')      # per-panel PNGs + preview (was a session scratchpad)
STIMS = ['natural_scenes', 'static_gratings', 'gabors']
STIM_TITLE = {'natural_scenes': 'Natural Scenes', 'static_gratings': 'Static Gratings',
              'gabors': 'Gabors'}
MIN_STABILITY = float(os.environ.get('CFC_THRESH', '0.8'))   # trial-presence threshold
ROBUST_P = float(os.environ.get('CFC_ROBUST_P', '0.8'))      # trial-bootstrap one-way cutoff
PREFIX = os.environ.get('CFC_PREFIX', 'cits_v2_directedB')   # input-array prefix (UNSM variant switch)
OUTTAG = os.environ.get('CFC_OUTTAG', '')                    # output-filename suffix
MIN_ALPHA = 0.3
MAX_WIDTH_SCALE = float(os.environ.get('CFC_MAXW', '2.6'))   # max edge line-width scale
MIN_WIDTH_SCALE = float(os.environ.get('CFC_MINW', '0.9'))   # min edge line-width scale
RENDER_DPI = 150

# stimulus colors: VERBATIM from script_matchbarplot.ipynb cell 28 stim_color_map
STIM_COLOR = {'natural_scenes': '#0072B2', 'static_gratings': '#E69F00',
              'gabors': '#009E73'}

# ---- brain-region order + node subregion labels (region-ordered union-68 frame) ----
labels_ordered = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam',
                  'CA1', 'CA2', 'CA3', 'DG', 'SUB', 'POL', 'LGv', 'LP']
union_labels = np.load(_union('cits_v2_union_labels.npy'), allow_pickle=True)
permute = []
for lab in labels_ordered:
    permute += list(np.where(union_labels == lab)[0])
permute = np.array(permute)
unit_labels_permuted = np.array([str(x) for x in union_labels[permute]])   # len 68

# ---- color groups (verbatim from script_matchbarplot.ipynb) ----
visual_cortex = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam']
hippocampus = ['CA1', 'CA2', 'CA3', 'DG', 'SUB']
thalamus = ['LGv', 'LP']
other = ['POL']


import matplotlib.colors as mcolors
# MATCH the comparison montage exactly: colorblind-safe Orange / Teal / Purple gradients
_TEAL = mcolors.LinearSegmentedColormap.from_list('teal', ['#D6EFEA', '#5CB8A8', '#0E6B5B'])
def _shade(regs, cmap, a, b):
    return {r: cmap(a + (b - a) * (k / max(1, len(regs) - 1))) for k, r in enumerate(regs)}
color_group_map = {}
color_group_map.update(_shade(['VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam'], cm.Oranges, 0.28, 0.90))
color_group_map.update(_shade(['CA1', 'CA2', 'CA3', 'DG', 'SUB', 'POL'], _TEAL, 0.18, 0.95))
color_group_map.update(_shade(['LGv', 'LP'], cm.Purples, 0.55, 0.88))


def get_clustered_circle_positions_grouped(labels, min_spacing=6, base_cluster_radius=3.0):
    """VERBATIM from script_matchbarplot.ipynb"""
    pos = {}
    region_indices = defaultdict(list)
    for i, label in enumerate(labels):
        region_indices[label].append(i)
    region_groups = [visual_cortex, hippocampus, thalamus, other]
    region_names = [r for group in region_groups for r in group if r in region_indices]
    num_regions = len(region_names)
    region_sizes = {label: len(region_indices[label]) for label in region_names}
    max_cluster_radius = base_cluster_radius * max(region_sizes.values()) ** 0.5
    circle_radius = min_spacing * num_regions + max_cluster_radius * 2
    theta_regions = np.linspace(0, 2 * np.pi, num_regions, endpoint=False)
    for k, (label, theta) in enumerate(zip(region_names, theta_regions)):
        indices = region_indices[label]
        n = len(indices)
        cluster_radius = base_cluster_radius * (n ** 0.5)
        cx = np.cos(theta) * circle_radius
        cy = np.sin(theta) * circle_radius
        theta_neurons = np.linspace(0, 2 * np.pi, n, endpoint=False)
        for j, idx in enumerate(indices):
            dx = np.cos(theta_neurons[j]) * cluster_radius
            dy = np.sin(theta_neurons[j]) * cluster_radius
            pos[idx] = (cx + dx, cy + dy)
    return pos


def load_stim_edge_data():
    """Directed contemporaneous CITS. An arrow i->j is drawn when it is present (as i->j OR
    i<->j) in >= MIN_STABILITY of the 60 trials (cits_v2_directedB_*_fwd68.npy).
    Bidirectional pairs (both directions >= threshold) render as two arrows.
    avg = |LSCM weight| (thickness); stab = directional stability (transparency)."""
    data = {}
    gmin, gmax = float('inf'), float('-inf')
    for stim in STIMS:
        fwd = np.load(f'{FIG}/{PREFIX}_{stim}_fwd68.npy')    # P(i->j present, incl i<->j)
        w = np.load(f'{FIG}/{PREFIX}_{stim}_w68.npy')        # mean |LSCM weight|
        pboot = np.load(f'{FIG}/{PREFIX}_{stim}_pboot68.npy')# P(one-way i->j) over trial bootstraps
        # an edge is SHOWN if present (either direction) in >= MIN_STABILITY of trials;
        # an ARROW i->j is drawn only if it is one-way in >= ROBUST_P of trial-bootstraps,
        # otherwise the edge is drawn bidirectional.
        present = (fwd >= MIN_STABILITY) | (fwd.T >= MIN_STABILITY)
        avg = np.where(present, w, 0.0)
        # direction: 'pboot' (one-way bootstrap >= ROBUST_P) or 'presence'
        # (i->j present >= MIN_STABILITY but reverse below it => one-way).
        if os.environ.get('CFC_DIRMODE', 'pboot') == 'presence':
            dir68 = ((fwd >= MIN_STABILITY) & (fwd.T < MIN_STABILITY)).astype(float)
        else:
            dir68 = (pboot >= ROBUST_P).astype(float)
        data[stim] = (avg, fwd, dir68)
        wv = np.abs(avg[avg != 0])
        if wv.size:
            gmin = min(gmin, wv.min()); gmax = max(gmax, wv.max())
    return data, gmin, gmax


# ----------------------------------------------------------------------------
# PANEL A -- one native (12,10) graph per stimulus (cell 24, verbatim styling)
# ----------------------------------------------------------------------------
def render_graph(stim, avg_perm, stab_perm, dir_perm, pos, gmin, gmax, out_png):
    # Reproducible edges (per-trial >=80% presence). Direction from the high-power
    # POOLED contemporaneous-CITS graph: draw an arrow where the pooled CFC orients the edge
    # (i->j and not j->i), otherwise an undirected line. Thickness ~ |LSCM weight|,
    # transparency ~ per-trial stability.
    n = len(unit_labels_permuted)
    G = nx.DiGraph(); G.add_nodes_from(range(n))
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.93])   # leave top strip for title
    node_colors = [color_group_map[unit_labels_permuted[k]] for k in range(n)]
    from matplotlib.path import Path as _MP
    from matplotlib.patches import PathPatch as _PP, FancyArrowPatch as _FAP

    def _arc(a, b, rad, color, lw, alpha, headsize, heads):
        """Draw a quadratic-Bezier arc a->b and place arrowhead(s) partway ALONG the
        line (so heads are visible near, not at, the node). heads = list of (t0,t1)."""
        p0 = np.array(pos[a], float); p1 = np.array(pos[b], float)
        mid = (p0 + p1) / 2.0; d = p1 - p0; perp = np.array([d[1], -d[0]]); pc = mid + rad * perp
        ax.add_patch(_PP(_MP([p0, pc, p1], [_MP.MOVETO, _MP.CURVE3, _MP.CURVE3]),
                         fill=False, edgecolor=color, lw=lw, alpha=alpha, zorder=2, capstyle='round'))
        bez = lambda t: (1 - t) ** 2 * p0 + 2 * (1 - t) * t * pc + t ** 2 * p1
        for t0, t1 in heads:
            ax.add_patch(_FAP(bez(t0), bez(t1), arrowstyle='-|>', mutation_scale=headsize,
                              color=color, lw=lw, alpha=alpha, zorder=3))

    n_dir = 0; n_bidir = 0
    for i in range(n):
        for j in range(i + 1, n):
            w = max(avg_perm[i, j], avg_perm[j, i])
            if w == 0:
                continue                        # edge not present (>=80% trials) either way
            dij = bool(dir_perm[i, j])          # robust one-way i->j (bootstrap P>=ROBUST_P)
            dji = bool(dir_perm[j, i])
            # stability >= 0.8 for every drawn edge, so alpha would span only 0.86-1.0
            # (imperceptible); draw all edges at a single fixed opacity instead.
            alpha_val = 0.9
            width_val = 2 * np.interp(abs(w), [gmin, gmax], [MIN_WIDTH_SCALE, MAX_WIDTH_SCALE])
            if dij != dji:
                u, v = (i, j) if dij else (j, i)         # head ~78% along the line
                _arc(u, v, 0.12, '#333333', width_val, alpha_val, 34, [(0.72, 0.80)])
                n_dir += 1
            else:
                # bidirectional: one arc with a head near each end (opposite directions).
                # For short edges (e.g. LGv<->LGv) push heads toward the ends so the two
                # arrowheads don't overlap near the middle.
                L = np.hypot(pos[j][0] - pos[i][0], pos[j][1] - pos[i][1])
                heads = ([(0.85, 0.93), (0.15, 0.07)] if L < 40
                         else [(0.72, 0.80), (0.28, 0.20)])
                _arc(i, j, 0.14, '#333333', width_val, alpha_val, 30, heads)
                n_bidir += 1
    nc = nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=250, ax=ax)
    nc.set_zorder(5)
    _lbls = nx.draw_networkx_labels(G, pos, labels={i: str(i) for i in range(n)},
                                    font_size=11, ax=ax)
    for _t in _lbls.values():
        _t.set_zorder(6)
    ax.set_aspect('equal')      # keep region clusters + ring circular (no horizontal squeeze)
    ax.margins(0.03)
    ax.axis('off')
    fig.text(0.5, 0.965, STIM_TITLE[stim], ha='center', va='center',
             fontsize=44, fontweight='bold')
    fig.savefig(out_png, dpi=RENDER_DPI, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    return n_dir, n_bidir


def render_legend(gmin, gmax, out_png):
    """Shared Panel-A legend: subregion patches + edge-thickness lines + stability bar."""
    fig = plt.figure(figsize=(6.4, 10))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0]); ax.axis('off')
    subregion_order = [r for r in labels_ordered if r in set(unit_labels_permuted)]
    patches = [mpatches.Patch(color=color_group_map[r], label=r) for r in subregion_order]
    leg = ax.legend(handles=patches, title='Brain Subregions', loc='upper center',
                    bbox_to_anchor=(0.5, 0.995), ncol=2, fontsize=34, title_fontsize=34,
                    frameon=False, handlelength=1.3, columnspacing=1.0, labelspacing=0.28)
    ax.add_artist(leg)
    # edge-strength (thickness) key  -- weight is the LSCM edge strength (caption clarifies)
    ax.text(0.5, 0.39, 'Edge strength', ha='center', fontsize=34, transform=ax.transAxes)
    legw = [gmin, (gmin + gmax) / 2, gmax]
    for i, wv in enumerate(legw):
        lwid = 2 * np.interp(wv, [gmin, gmax], [MIN_WIDTH_SCALE, MAX_WIDTH_SCALE])
        y = 0.33 - i * 0.045
        ax.plot([0.22, 0.52], [y, y], transform=ax.transAxes,
                lw=lwid, color='black', solid_capstyle='round')
        ax.text(0.58, y, f'{wv:.2f}', transform=ax.transAxes, va='center', fontsize=30)
    # (Edge Direction key removed -- arrowheads on the graph convey direction)
    # (Edge stability key removed -- all drawn edges have stability >= 0.8, so a
    #  transparency ramp spanning 0.86-1.0 was imperceptible; edges drawn at fixed opacity)
    fig.savefig(out_png, dpi=RENDER_DPI, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


# ----------------------------------------------------------------------------
# PANEL B -- one native bar chart per broad region (cell 28, verbatim geometry)
# ----------------------------------------------------------------------------
def panelB_grouped(stim_edge_data):
    # UNDIRECTED coupling: count each unordered region-pair ONCE and normalise by
    # the number of POSSIBLE undirected pairs (within A: |A|*(|A|-1)/2; between A,B:
    # |A|*|B|). The old code counted both i->j and j->i and divided by source-size**2,
    # which fabricated a directional asymmetry (e.g. VISal->VISp 4% vs VISp->VISal 2%)
    # even though the graph is symmetric.
    region_labels = unit_labels_permuted
    rni = defaultdict(list)
    for idx, label in enumerate(region_labels):
        rni[label].append(idx)
    filtered = [r for r in rni if len(rni[r]) > 5]
    n = len(region_labels)
    directed = os.environ.get('CFC_PANELB', 'directed') == 'directed'

    cnt = defaultdict(int)
    for stim in STIMS:
        avg, stab, _dir = stim_edge_data[stim]
        if directed:
            # DIRECTED: ordered pairs, count i->j present (fwd>=threshold, incl i<->j),
            # normalised by ORDERED possible pairs. Bidirectional edges contribute both ways.
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    ri, rj = region_labels[i], region_labels[j]
                    if ri not in filtered or rj not in filtered:
                        continue
                    if avg[i, j] != 0 and stab[i, j] >= MIN_STABILITY:
                        cnt[(stim, f'{ri} → {rj}')] += 1
        else:
            for i in range(n):
                for j in range(i + 1, n):
                    ri, rj = region_labels[i], region_labels[j]
                    if ri not in filtered or rj not in filtered:
                        continue
                    present = ((avg[i, j] != 0 and stab[i, j] >= MIN_STABILITY) or
                               (avg[j, i] != 0 and stab[j, i] >= MIN_STABILITY))
                    if not present:
                        continue
                    a, b = sorted([ri, rj])
                    cnt[(stim, f'{a} – {b}')] += 1

    sep = ' → ' if directed else ' – '
    rows = []
    for (stim, conn), c in cnt.items():
        a, b = conn.split(sep)
        na, nb = len(rni[a]), len(rni[b])
        if directed:
            possible = na * (na - 1) if a == b else na * nb
        else:
            possible = na * (na - 1) // 2 if a == b else na * nb
        sup = super_of(a) if super_of(a) == super_of(b) else 'Between'
        rows.append({'Stimulus': stim, 'Connection': conn, 'Super': sup,
                     'Count': c, 'Percentage': 100.0 * c / possible if possible else 0.0})
    grouped = (pd.DataFrame(rows) if rows else
               pd.DataFrame(columns=['Stimulus', 'Connection', 'Super', 'Count', 'Percentage']))
    return grouped, rni


def super_of(region):
    if region in visual_cortex:
        return 'Visual Cortex'
    if region in hippocampus:
        return 'Hippocampus'
    if region in thalamus:
        return 'Thalamus'
    return 'Other'


def render_bar(sup, grouped, ncount, out_png, ymax=None):
    """VERBATIM cell-30 geometry: bar_w=1, group_gap=1 (bars touch within a
    connection group; a one-bar gap separates groups), fixed inches-per-bar so
    bar width + gap are IDENTICAL across the three panels. height fixed at 6in.
    Deterministic heights (published composite shows no error bars)."""
    sub = grouped[grouped['Super'] == sup]
    sub = sub[sub['Percentage'] > 0]
    bar_w = 1.0
    group_gap = 1.0
    connections = sorted(sub['Connection'].unique(),
                         key=lambda c: sub[sub['Connection'] == c]['Percentage'].max(),
                         reverse=True)
    xs, heights, colors = [], [], []
    tick_pos, tick_lab = [], []
    cursor = 0.0
    total_slots = 0
    for conn in connections:
        cd = sub[sub['Connection'] == conn]
        bars = [(stim, cd[cd['Stimulus'] == stim]['Percentage'].values[0])
                for stim in STIMS if not cd[cd['Stimulus'] == stim].empty]
        nb = len(bars); start = cursor
        for i, (stim, pct) in enumerate(bars):
            xs.append(start + i * bar_w + bar_w / 2)
            heights.append(pct); colors.append(STIM_COLOR[stim])
        tick_pos.append(start + nb * bar_w / 2); tick_lab.append(conn)
        cursor += nb * bar_w + group_gap
        total_slots += nb + 1

    INCH_PER_SLOT = 0.42            # fixed physical width per bar/gap slot
    fig = plt.figure(figsize=(total_slots * INCH_PER_SLOT + 1.4, 6))
    ax = fig.add_subplot(111)
    ax.bar(xs, heights, color=colors, width=bar_w, align='center')
    ax.set_xlim(-group_gap, cursor)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, rotation=90, fontsize=22)
    ax.tick_params(axis='x', length=0)
    ax.set_ylabel('% of neuron pairs', fontsize=26)
    ax.set_ylim(0, (ymax if ymax else (max(heights) if heights else 1)) * 1.08)
    ax.set_title(f'{sup} (n={ncount})', fontsize=30, pad=14)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=22)
    # no per-panel legend -- a single shared Stimulus legend is added in the composite
    fig.tight_layout()
    fig.savefig(out_png, dpi=RENDER_DPI, bbox_inches='tight')
    plt.close(fig)


def render_panelB_combined(grouped, ncount, out_png):
    """All three regions in ONE figure with a SHARED y-axis (matched range + axis
    width) + a single Stimulus legend styled like the Panel-A legend."""
    supers = ['Visual Cortex', 'Thalamus', 'Hippocampus']
    ymax = float(grouped[grouped['Percentage'] > 0]['Percentage'].max())
    layouts = {}
    for sup in supers:
        sub = grouped[(grouped['Super'] == sup) & (grouped['Percentage'] > 0)]
        conns = sorted(sub['Connection'].unique(),
                       key=lambda c: sub[sub['Connection'] == c]['Percentage'].max(), reverse=True)
        xs, hs, cols, tpos, tlab = [], [], [], [], []
        cur = 0.0
        for conn in conns:
            cd = sub[sub['Connection'] == conn]
            bars = [(st, cd[cd['Stimulus'] == st]['Percentage'].values[0]) for st in STIMS
                    if not cd[cd['Stimulus'] == st].empty]
            nb = len(bars)
            for i, (st, pct) in enumerate(bars):
                xs.append(cur + i + 0.5); hs.append(pct); cols.append(STIM_COLOR[st])
            tpos.append(cur + nb / 2.0); tlab.append(conn); cur += nb + 1
        layouts[sup] = (xs, hs, cols, tpos, tlab, cur)
    widths = [max(layouts[s][5], 1.0) for s in supers]
    fig, axes = plt.subplots(1, 3, figsize=(sum(widths) * 0.66 + 6.0, 6.6),
                             gridspec_kw={'width_ratios': widths, 'wspace': 0.28}, sharey=False)
    for ax, sup in zip(axes, supers):
        xs, hs, cols, tpos, tlab, cur = layouts[sup]
        ax.bar(xs, hs, color=cols, width=1.0, align='center')
        ax.set_xlim(-1, cur); ax.set_ylim(0, ymax * 1.08)
        ax.set_xticks(tpos)
        ax.set_xticklabels(tlab, rotation=45, ha='right', fontsize=22)
        ax.tick_params(axis='x', length=0); ax.tick_params(labelsize=22)
        ax.set_ylabel('% of neuron pairs', fontsize=24)          # each panel keeps its y-label
        ax.set_title(f'{sup} (n={ncount[sup]})', fontsize=30, pad=14)
        ax.spines[['top', 'right']].set_visible(False)
    handles = [Line2D([0], [0], color=STIM_COLOR[s], lw=12, label=STIM_TITLE[s]) for s in STIMS]
    # Panel B is composited at a larger scale than the Panel-A legend, so 25pt here
    # renders at the same on-canvas size as the Panel-A legend's 34pt (measured scale ratio).
    fig.legend(handles=handles, title='Stimulus', loc='center left', frameon=False,
               fontsize=25, title_fontsize=25, bbox_to_anchor=(0.90, 0.5),
               handlelength=1.4, labelspacing=0.5)
    fig.tight_layout(rect=[0, 0, 0.80, 1])                        # reserve the right strip for the legend
    fig.savefig(out_png, dpi=RENDER_DPI, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------------
# PIL composition -- paste native panels, preserving every proportion
# ----------------------------------------------------------------------------
def scaled_to_h(img, h):
    w = int(round(img.width * h / img.height))
    return img.resize((w, h), Image.LANCZOS)


def scaled_to_w(img, w):
    h = int(round(img.height * w / img.width))
    return img.resize((w, h), Image.LANCZOS)


def main():
    stim_edge_data, gmin, gmax = load_stim_edge_data()
    pos = get_clustered_circle_positions_grouped(unit_labels_permuted, min_spacing=8, base_cluster_radius=7.5)
    # --- reposition thalamus clusters (applied to all 3 graphs) ---
    def _cluster(lbl):
        return [i for i in range(len(unit_labels_permuted)) if unit_labels_permuted[i] == lbl]
    lgv = _cluster('LGv')
    lrad = 1.0
    if lgv:
        cx = np.mean([pos[i][0] for i in lgv]); cy = np.mean([pos[i][1] for i in lgv])
        lrad = np.mean([np.hypot(pos[i][0] - cx, pos[i][1] - cy) for i in lgv]) or 1.0
        for i in lgv:                                   # expand + shift right so its self-edge arrow is visible
            x, y = pos[i]; pos[i] = (cx + (x - cx) * 1.45 + lrad * 1.3, cy + (y - cy) * 1.45)
    for i in _cluster('LP'):                            # LP has no edges -> move to upper-right, out of the way
        x, y = pos[i]; pos[i] = (x + lrad * 2.6, y + lrad * 2.2)

    # ---- render Panel A (graphs + shared legend) at native size ----
    edge_counts = {}
    graph_pngs = {}
    for stim in STIMS:
        avg_perm, stab_perm, dir_perm = stim_edge_data[stim]
        p = f'{SCRATCH}/cfc_graph_{stim}.png'
        edge_counts[stim] = render_graph(stim, avg_perm, stab_perm, dir_perm, pos, gmin, gmax, p)
        graph_pngs[stim] = p
    legend_png = f'{SCRATCH}/cfc_legend.png'
    render_legend(gmin, gmax, legend_png)

    # ---- render Panel B (bar charts) at native size ----
    grouped, region_neuron_indices = panelB_grouped(stim_edge_data)
    supers = ['Visual Cortex', 'Thalamus', 'Hippocampus']
    ncount = {'Visual Cortex': sum(len(region_neuron_indices[r]) for r in visual_cortex
                                   if len(region_neuron_indices[r]) > 5),
              'Thalamus': sum(len(region_neuron_indices[r]) for r in thalamus
                              if len(region_neuron_indices[r]) > 5),
              'Hippocampus': sum(len(region_neuron_indices[r]) for r in hippocampus
                                 if len(region_neuron_indices[r]) > 5)}
    panelB_png = f'{SCRATCH}/cfc_panelB.png'
    render_panelB_combined(grouped, ncount, panelB_png)

    # ---- compose (PIL) ----
    PAD = 40
    GPAD = 12                        # tighter gap between the 3 Panel-A graphs
    GH = 820                         # graph row height (px)
    graphs = [scaled_to_h(Image.open(graph_pngs[s]).convert('RGB'), GH) for s in STIMS]
    legend = scaled_to_h(Image.open(legend_png).convert('RGB'), GH)
    rowA_w = sum(g.width for g in graphs) + legend.width + GPAD * len(graphs) + 4 * PAD

    # Panel B: single combined image (shared y-axis + one legend), scaled to a height
    BH = 820
    panelB = scaled_to_h(Image.open(panelB_png).convert('RGB'), BH)

    canvas_w = max(rowA_w, panelB.width + 2 * PAD + 90) + 2 * PAD
    total_h = PAD + GH + PAD + BH + PAD          # single Panel-B row now
    canvas = Image.new('RGB', (canvas_w, total_h), 'white')

    # paste Panel A row (graphs left-to-right with tight gap, shared legend at right)
    x = PAD; y = PAD
    for g in graphs:
        canvas.paste(g, (x, y)); x += g.width + GPAD
    canvas.paste(legend, (x + 3 * PAD, y + (GH - legend.height) // 2))

    # paste Panel B (one image; shifted right so the 'B' letter clears the y-axis)
    y2 = PAD + GH + PAD
    canvas.paste(panelB, (PAD + 90, y2))

    # ---- panel letters A / B + a thin divider so bars read as one Panel B ----
    draw = ImageDraw.Draw(canvas)
    try:
        fpath = f'{matplotlib.get_data_path()}/fonts/ttf/DejaVuSans-Bold.ttf'
        font = ImageFont.truetype(fpath, 92)
    except Exception:
        font = ImageFont.load_default()
    draw.text((14, 6), 'A', fill='black', font=font)
    draw.text((14, y2 + 4), 'B', fill='black', font=font)

    tag = f'_th{int(round(MIN_STABILITY*100))}'
    out_pdf = f'{FIG}/cfc_stimtypes_directed{tag}{OUTTAG}.pdf'
    out_png = f'{FIG}/cfc_stimtypes_directed{tag}{OUTTAG}.png'
    canvas.save(out_pdf, 'PDF', resolution=RENDER_DPI)
    canvas.save(out_png)
    canvas.save(f'{SCRATCH}/cfc_directed{tag}_preview.png')
    print('saved', out_pdf)
    print('\nPer-stimulus directed-edge counts (>= %.0f%% of 60 trials):' % (MIN_STABILITY * 100))
    for stim in STIMS:
        nd, nb = edge_counts[stim]
        print(f'  {stim:16s}: {nd} one-way + {nb} bidirectional = {nd+nb} edges')
    print('\nTop area-pair densities (% of possible neuron pairs):')
    for sup in supers:
        sub = grouped[grouped['Super'] == sup]
        tp = (sub.groupby('Connection')['Percentage'].max()
              .sort_values(ascending=False).head(5))
        print(f'  [{sup}] n={ncount[sup]}')
        for k, v in tp.items():
            print(f'      {k:18s} {v:.2f}%')


if __name__ == '__main__':
    main()
