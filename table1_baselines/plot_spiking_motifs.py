"""
plot_spiking_motifs.py
Seven compact motif glyphs for CITS benchmark Table 1.

Non-recurrent (no self-loops):
  fig_motif_convergence     -- edges 1->3, 2->3, 3->4
  fig_motif_commoncause     -- edges 1->2, 1->3, 2->4, 3->4
  fig_motif_chain           -- edges 1->2, 2->3, 3->4

Recurrent (self-loop on every node):
  fig_motif_convergence_self   -- same edges + self-loops on 1,2,3,4
  fig_motif_commoncause_self   -- same edges + self-loops on 1,2,3,4
  fig_motif_chain_self         -- same edges + self-loops on 1,2,3,4
  fig_motif_depression_self    -- convergence edges + depressing-synapse
                                  glyphs + self-loops on 1,2,3,4

Figure standards applied:
- Universal fixed canvas (same figsize/xlim/ylim for all 7 glyphs) so nodes
  appear identical in size when table cells are scaled to the same height.
- figsize=(2.40, 1.75) exactly proportional to data ranges (2.40 x 1.75 units)
  at 1.0 in/unit scale, so aspect='equal' fills the canvas with zero whitespace.
- subplots_adjust(left=0, right=1, top=1, bottom=0) + no bbox_inches='tight'.
- Axes turned off entirely (pure schematic, no axis lines/ticks/labels).
- Self-loops: open ~300° Arc (gap_deg=60) on the node's free side, with a
  small '->' arrowhead (mutation_scale=ARR_MS*0.45) at the terminating end,
  tangent to the arc, no barb overlap.
- Depressing-synapse glyph: perpendicular bar + filled dot at 72% along edge.
- Font family: Liberation Sans / Arial; pdf.fonttype = 42.
- Node fill #D8D8D8, border/arrow near-black; arrow lw 2.0, mutation_scale 17.
- DPI 300, white background.
- Designed to be legible at ~1.3-1.6 cm table-cell width.
"""

import os
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Liberation Sans', 'Arial', 'DejaVu Sans']

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared')))
from paths import outdir as _outdir
OUT = _outdir('table1_baselines')   # was CITS_manuscript/figures/final_figures_2026-08-31
os.makedirs(OUT, exist_ok=True)

# ── Universal canvas (identical for all 7 glyphs) ─────────────────────────────
# figsize proportional to data ranges so aspect='equal' fills canvas exactly;
# subplots_adjust removes all margins.
# Chain nodes (0.60 spacing, 4 nodes) span x in [-0.08, 1.98]; fits comfortably.
FIGSIZE = (2.40, 1.75)   # inches
XLIM    = (-0.25, 2.15)  # data range 2.40
YLIM    = (-0.55, 1.20)  # data range 1.75
# Canvas centre: x = 0.95, y = 0.325

# ── Style constants ───────────────────────────────────────────────────────────
NODE_R  = 0.13       # radius in data coordinates (= 0.13 in at 1.0 in/unit)
NODE_FC = '#D8D8D8'  # light grey fill
NODE_EC = '#1A1A1A'  # near-black border
NODE_LW = 2.0
ARR_COL = '#222222'  # dark arrow colour
ARR_LW  = 2.0
ARR_MS  = 17         # arrowhead mutation_scale (cross-edges)


# ── Drawing helpers ───────────────────────────────────────────────────────────

def make_ax():
    """Universal fixed canvas: identical figsize, xlim, ylim for every glyph."""
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('white')
    return fig, ax


def draw_node(ax, x, y):
    """Plain grey filled circle, no label."""
    ax.add_patch(Circle((x, y), NODE_R, fc=NODE_FC, ec=NODE_EC,
                         lw=NODE_LW, zorder=4, clip_on=False))


def draw_edge(ax, pos, s, d):
    """Directed '->' arrow offset to node boundaries."""
    x0, y0 = pos[s]
    x1, y1 = pos[d]
    dx, dy = x1 - x0, y1 - y0
    dist = np.hypot(dx, dy)
    ux, uy = dx / dist, dy / dist
    ax.annotate('',
        xy=(x1 - ux*NODE_R, y1 - uy*NODE_R),
        xytext=(x0 + ux*NODE_R, y0 + uy*NODE_R),
        arrowprops=dict(arrowstyle='->', color=ARR_COL,
                        lw=ARR_LW, mutation_scale=ARR_MS,
                        connectionstyle='arc3,rad=0'),
        zorder=3)


def draw_self_loop(ax, x, y, angle_deg=90, loop_r=0.085, gap_deg=60):
    """
    Open-arc self-loop: ~300° arc with a small tangential '->' arrowhead
    at the TERMINATING END (theta2), so the arrowhead cleanly finishes
    the line with zero barb overlap.

    Geometry:
      - Loop circle centre at (NODE_R + loop_r) from node centre in
        direction angle_deg.  Circle radius = loop_r.
      - 60° gap centred on the node-facing side (attach = angle_deg+180).
        Arc goes CCW from (attach + 30) for 300°, ending at (attach - 30).
      - The arc's far side (angle_deg) is the arc midpoint, well clear of
        both endpoints and of all cross-edges.
      - Arrowhead at theta2 (terminating end), tangent CCW, size ARR_MS*0.45
        (noticeably smaller than cross-edge heads).

    Parameters
    ----------
    angle_deg : free-side direction (degrees) — chosen opposite cross-edges.
    loop_r    : satellite circle radius (default 0.085).
    gap_deg   : angular gap left on node-facing side (default 60).
    """
    lx = x + (NODE_R + loop_r) * np.cos(np.radians(angle_deg))
    ly = y + (NODE_R + loop_r) * np.sin(np.radians(angle_deg))

    attach = (angle_deg + 180) % 360           # node-facing direction
    theta1 = (attach + gap_deg / 2.0) % 360    # arc start (CCW)
    theta2 = theta1 + (360.0 - gap_deg)        # arc end = theta1 + 300

    ax.add_patch(Arc(
        (lx, ly), 2*loop_r, 2*loop_r,
        angle=0.0, theta1=theta1, theta2=theta2,
        color=ARR_COL, lw=ARR_LW, zorder=6, clip_on=False,
    ))

    # Small tangential arrowhead at the terminating end
    t2  = np.radians(theta2 % 360)
    px  = lx + loop_r * np.cos(t2)
    py  = ly + loop_r * np.sin(t2)
    tx  = -np.sin(t2)    # CCW tangent direction
    ty  =  np.cos(t2)
    shaft = 0.012
    ax.annotate('',
        xy=(px, py),
        xytext=(px - tx * shaft, py - ty * shaft),
        arrowprops=dict(arrowstyle='->',
                        color=ARR_COL,
                        lw=ARR_LW * 0.75,
                        mutation_scale=ARR_MS * 0.45,
                        shrinkA=0, shrinkB=0),
        annotation_clip=False, zorder=7)


def draw_dep_glyph(ax, pos, s, d, t=0.72):
    """
    Depressing-synapse glyph: perpendicular bar + filled dot at fraction t
    along the edge centreline.
    """
    x0, y0 = pos[s]
    x1, y1 = pos[d]
    dx, dy = x1 - x0, y1 - y0
    dist = np.hypot(dx, dy)
    px, py = -dy / dist, dx / dist   # perpendicular unit vector
    mx, my = x0 + t * dx, y0 + t * dy
    bh = 0.050
    ax.plot([mx - px*bh, mx + px*bh],
            [my - py*bh, my + py*bh],
            color=NODE_EC, lw=2.4, solid_capstyle='round', zorder=6)
    ax.plot(mx, my, 'o', ms=5, color=NODE_EC,
            markeredgecolor='none', zorder=7)


def save_glyph(fig, stem):
    """Save PDF + PNG on fixed canvas — NO bbox_inches='tight'."""
    for ext in ('pdf', 'png'):
        path = os.path.join(OUT, f'{stem}.{ext}')
        fig.savefig(path, dpi=300, facecolor='white')
        print(f'Saved: {path}')
    plt.close(fig)


# ── Node positions ─────────────────────────────────────────────────────────────
# All positions centred in the universal canvas (centre x=0.95, y=0.325).
#
# Convergence / Depression:
#   Layout: 1(upper-left), 2(lower-left) -> 3(centre) -> 4(right).
#   Shifted +0.25 x from base to centre at x~0.90 in canvas.
#   Free-side angles (opposite cross-edges):
#     node1 exits lower-right (~333°) -> free upper-left -> 155°
#     node2 exits upper-right (~27°)  -> free lower-left -> 205°
#     node3 cross-edges at 0°(exit), 153°(in), 207°(in) -> top free -> 90°
#     node4 receives from left (180°) -> upper-right free -> 20°
POS_CONV    = {1:(0.25,0.65),  2:(0.25,0.00),  3:(0.90,0.325), 4:(1.55,0.325)}
EDGES_CONV  = [(1,3), (2,3), (3,4)]
SELF_CONV   = {1:155, 2:205, 3:90, 4:20}

# Common cause + common effect:
#   Layout: 1(left) -> 2(upper-mid), 3(lower-mid) -> 4(right)  [diamond].
#   Shifted +0.30 x to centre at x=0.95 in canvas.
#   Free-side angles:
#     node1 exits upper-right & lower-right -> left free -> 180°
#     node2 receives lower-left, exits lower-right -> top free -> 90°
#     node3 receives upper-left, exits upper-right -> bottom free -> 270°
#     node4 receives upper-left & lower-left -> right free -> 0°
POS_CC      = {1:(0.30,0.325), 2:(0.95,0.70),  3:(0.95,-0.05), 4:(1.60,0.325)}
EDGES_CC    = [(1,2), (1,3), (2,4), (3,4)]
SELF_CC     = {1:180, 2:90, 3:270, 4:0}

# Chain:
#   Layout: 1->2->3->4 horizontal, 0.60 spacing.
#   Shifted +0.05 x to centre at x=0.95 in canvas.
#   All cross-edges horizontal -> top (90°) free for every node.
#   Content x in [-0.08, 1.98]; fits within XLIM=(-0.25, 2.15).
POS_CHAIN   = {1:(0.05,0.325), 2:(0.65,0.325), 3:(1.25,0.325), 4:(1.85,0.325)}
EDGES_CHAIN = [(1,2), (2,3), (3,4)]
SELF_CHAIN  = {1:90, 2:90, 3:90, 4:90}


# ── Glyph 1: Convergence (no self-loops) ─────────────────────────────────────
fig, ax = make_ax()
for n, (x, y) in POS_CONV.items():
    draw_node(ax, x, y)
for s, d in EDGES_CONV:
    draw_edge(ax, POS_CONV, s, d)
save_glyph(fig, 'fig_motif_convergence')

# ── Glyph 2: Common cause (no self-loops) ────────────────────────────────────
fig, ax = make_ax()
for n, (x, y) in POS_CC.items():
    draw_node(ax, x, y)
for s, d in EDGES_CC:
    draw_edge(ax, POS_CC, s, d)
save_glyph(fig, 'fig_motif_commoncause')

# ── Glyph 3: Chain (no self-loops) ───────────────────────────────────────────
fig, ax = make_ax()
for n, (x, y) in POS_CHAIN.items():
    draw_node(ax, x, y)
for s, d in EDGES_CHAIN:
    draw_edge(ax, POS_CHAIN, s, d)
save_glyph(fig, 'fig_motif_chain')

# ── Glyph 4: Convergence self ─────────────────────────────────────────────────
fig, ax = make_ax()
for n, (x, y) in POS_CONV.items():
    draw_node(ax, x, y)
for s, d in EDGES_CONV:
    draw_edge(ax, POS_CONV, s, d)
for n, ang in SELF_CONV.items():
    draw_self_loop(ax, *POS_CONV[n], angle_deg=ang)
save_glyph(fig, 'fig_motif_convergence_self')

# ── Glyph 5: Common cause self ────────────────────────────────────────────────
fig, ax = make_ax()
for n, (x, y) in POS_CC.items():
    draw_node(ax, x, y)
for s, d in EDGES_CC:
    draw_edge(ax, POS_CC, s, d)
for n, ang in SELF_CC.items():
    draw_self_loop(ax, *POS_CC[n], angle_deg=ang)
save_glyph(fig, 'fig_motif_commoncause_self')

# ── Glyph 6: Chain self ───────────────────────────────────────────────────────
fig, ax = make_ax()
for n, (x, y) in POS_CHAIN.items():
    draw_node(ax, x, y)
for s, d in EDGES_CHAIN:
    draw_edge(ax, POS_CHAIN, s, d)
for n, ang in SELF_CHAIN.items():
    draw_self_loop(ax, *POS_CHAIN[n], angle_deg=ang)
save_glyph(fig, 'fig_motif_chain_self')

# ── Glyph 7: Depression self (convergence topology + self-loops, no dep marker)
# The "Short-term depression" label in the table row carries the meaning;
# the synapse-marker (bar+dot) was removed because it collided with arrowheads.
fig, ax = make_ax()
for n, (x, y) in POS_CONV.items():
    draw_node(ax, x, y)
for s, d in EDGES_CONV:
    draw_edge(ax, POS_CONV, s, d)
for n, ang in SELF_CONV.items():
    draw_self_loop(ax, *POS_CONV[n], angle_deg=ang)
save_glyph(fig, 'fig_motif_depression_self')

print('\nAll 7 motif glyphs (PDF + PNG) written to:')
print(f'  {OUT}')
print('\nCanvas: figsize=(2.40, 1.75) in, xlim=(-0.25, 2.15), ylim=(-0.55, 1.20)')
print('NODE_R=0.13; loop_r=0.085; gap_deg=60; arrowhead at terminating end.')
