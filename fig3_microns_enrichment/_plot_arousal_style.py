"""
Stimulus-type FC figures in the arousal-paper Fig 1F / Fig 2A style.

Refinements applied (per coordinator):
  1. Okabe-Ito colorblind palette (sky-blue / orange / reddish-purple), one
     consistent Clip/Monet/Trippy -> color mapping across all figures.
  2. Area-pair x-axis order matches the arousal Fig 2 EM plot
     (plot_em_areapair_synapse_versionBsafe.py / plot_fig2_per_state_barplots):
     within [AL,LM,RL,V1] then between grouped by BETWEEN_GROUPS, both
     directions adjacent, AL<->RL last.
  3. Fig 1F  -> arousal per-state within/between layout, 3 stimuli in place of
     the 2 arousal states (plot_per_state_areas_13sess_versionBsafe.py Plot 1).
     Fig 2A -> arousal per-state stacked layout, one row per stimulus, 16 pairs
     in EM order (plot_fig2_per_state_barplots_13sess_versionBsafe.py).
  Friedman stars in the area-pair panels sit snug above each bar group.

Stats: field-bootstrap 95% CI (descriptive); paired Friedman + pairwise
Wilcoxon-FDR across the 3 stimuli (n=124 fields, paired within field).
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', 'shared')))
from paths import outdir as _outdir, result
CSV  = result('fig3_microns_enrichment', 'results/stimulus_fc_combined.csv')
OUT  = _outdir('fig3_microns_enrichment')
df   = pd.read_csv(CSV)
df['fid'] = list(zip(df['session'], df['scan'], df['field']))

# ── 1. Okabe-Ito palette, fixed mapping ──────────────────────────────────────
COL = {'clip': '#E69F00', 'Monet': '#009E73', 'Trippy': '#CC79A7'}  # Okabe-Ito no-blue trio (avoids arousal Fig1 #2c5282 navy)
LAB = {'clip': 'Clip', 'Monet': 'Monet', 'Trippy': 'Trippy'}
STIMS = ['clip', 'Monet', 'Trippy']          # column / bar order everywhere
N_BOOT = 5000
RNG = np.random.default_rng(42)

# ── 2. EM-matched area-pair order ────────────────────────────────────────────
AREAS = ['AL', 'LM', 'RL', 'V1']
WITHIN_PAIRS = [(a, a) for a in AREAS]
BETWEEN_GROUPS = [('LM', 'RL'), ('AL', 'LM'), ('LM', 'V1'),
                  ('AL', 'V1'), ('RL', 'V1'), ('AL', 'RL')]
BETWEEN_PAIRS = [(a, b) for g in BETWEEN_GROUPS
                 for (a, b) in [(g[0], g[1]), (g[1], g[0])]]
ALL_PAIRS = WITHIN_PAIRS + BETWEEN_PAIRS          # 16, within-first
N_WITHIN = len(WITHIN_PAIRS)
PAIR_LABELS = [f'{a}→{b}' for a, b in ALL_PAIRS]


# ── stats helpers ────────────────────────────────────────────────────────────
def sig_stars(p):
    if p is None or np.isnan(p):
        return 'n.s.'
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'


def fdr(ps):
    ps = np.asarray(ps, float)
    ok = ~np.isnan(ps)
    q = np.full_like(ps, np.nan)
    if ok.sum():
        q[ok] = multipletests(ps[ok], method='fdr_bh')[1]
    return q


def paired(metric):
    """(n_fields, 3) matrix aligned by field, columns = STIMS; NaN rows dropped."""
    piv = df.pivot_table(index='fid', columns='stim', values=metric)
    piv = piv.reindex(columns=STIMS)
    piv = piv.dropna(how='any')
    return piv.values


def friedman_p(M):
    if M.shape[0] < 3:
        return np.nan
    # constant column -> Friedman undefined
    if np.allclose(M.std(axis=0), 0):
        return np.nan
    try:
        return friedmanchisquare(M[:, 0], M[:, 1], M[:, 2]).pvalue
    except ValueError:
        return np.nan


def pairwise_wilcoxon(M):
    ps = []
    for i, j in combinations(range(3), 2):
        d = M[:, i] - M[:, j]
        if M.shape[0] < 1 or np.allclose(d, 0):
            ps.append(np.nan)
        else:
            try:
                ps.append(wilcoxon(M[:, i], M[:, j]).pvalue)
            except ValueError:
                ps.append(np.nan)
    return ps


def bootci(vals, B=N_BOOT):
    vals = np.asarray(vals, float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    if len(vals) == 1:
        return vals[0], vals[0], vals[0]
    idx = RNG.integers(0, len(vals), size=(B, len(vals)))
    bs = vals[idx].mean(axis=1)
    return float(vals.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


VAR_LABEL = {'A': 'Variant A  (CITS-GPU lagged)',
             'B': 'Variant B  (CITS-GPU lagged + PC-contemporaneous, VersionB-safe)'}


# ═════════════════════════════════════════════════════════════════════════════
# FIG 1F  —  within-area vs between-area, 3 stimuli
#   Layout mirrors plot_per_state_areas_13sess_versionBsafe.py Plot 1:
#   rows = metrics (density, strength), cols = scopes (within, between).
# ═════════════════════════════════════════════════════════════════════════════
def fig_within_between(var, fname):
    """
    Arousal Fig 1F layout (plot_per_state_areas_...VersionBsafe Plot 4), but with
    THREE per-stimulus panels in place of the two arousal-state panels.

    One horizontal row of 6 panels:
        [density: Clip | Monet | Trippy]  <gap>  [strength: Clip | Monet | Trippy]
    Each panel = within-area (solid) + between-area (hatched) bars, one stimulus
    colour.  Stimulus title (coloured) sits over the density triplet only, metric
    ylabel on the leftmost panel of each triplet, within/between legend at bottom.
    Stars = within vs between paired Wilcoxon, FDR-BH across the 6 panels.
    """
    import matplotlib.patches as mpatches
    import matplotlib.transforms as mtransforms
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import ScalarFormatter

    metrics = [('_d', 'Edge density'),
               ('_s', 'Edge strength')]
    panel_order = [(suf, mlabel, s) for suf, mlabel in metrics for s in STIMS]

    xw, xb = 1.0, 1.35

    # ---- precompute means/CI and paired within-vs-between p per panel ----------
    cell = {}       # (suf, stim) -> (mean_w, mean_b, ci_lo_w, ci_hi_w, ci_lo_b, ci_hi_b)
    raw_p, pkeys = [], []
    for suf, _mlabel, s in panel_order:
        sub = df[df.stim == s]
        mw, lw, hw = bootci(sub[f'{var}_within{suf}'].values)
        mb, lb, hb = bootci(sub[f'{var}_between{suf}'].values)
        cell[(suf, s)] = (mw, mb, lw, hw, lb, hb)
        # paired Wilcoxon within vs between (fields with both non-NaN)
        pair = sub[[f'{var}_within{suf}', f'{var}_between{suf}']].dropna().values
        if pair.shape[0] >= 1 and not np.allclose(pair[:, 0] - pair[:, 1], 0):
            try:
                p = wilcoxon(pair[:, 0], pair[:, 1]).pvalue
            except ValueError:
                p = np.nan
        else:
            p = np.nan
        raw_p.append(p)
        pkeys.append((suf, s))
    q_map = {k: qq for k, qq in zip(pkeys, fdr(raw_p))}

    # shared y-top per metric (across the 3 stimuli)
    shared_ytop = {}
    for suf, _ in metrics:
        tops = [max(cell[(suf, s)][3], cell[(suf, s)][5]) for s in STIMS]
        shared_ytop[suf] = max(tops) * 1.10

    # ---- layout: 6 axes with small intra-metric gaps + one big metric gap -----
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    wr = [1, 0.25, 1, 0.25, 1, 0.7, 1, 0.25, 1, 0.25, 1]
    gs = GridSpec(1, len(wr), figure=fig, width_ratios=wr, wspace=0.08)
    ax_cols = [0, 2, 4, 6, 8, 10]
    axes = [fig.add_subplot(gs[0, c]) for c in ax_cols]

    for i, ((suf, mlabel, s), ax) in enumerate(zip(panel_order, axes)):
        mw, mb, lw_, hw_, lb_, hb_ = cell[(suf, s)]
        means = [mw, mb]
        yerr = [[mw - lw_, mb - lb_], [hw_ - mw, hb_ - mb]]
        panel_max = max(abs(mw), abs(mb)) or 1

        bar_w = ax.bar([xw], [mw], width=0.22, color=COL[s],
                       edgecolor='#222222', linewidth=0.8, zorder=2)
        bar_b = ax.bar([xb], [mb], width=0.22, color=COL[s], hatch='////',
                       edgecolor='#222222', linewidth=0.8, zorder=2)
        for p_, h_ in [(bar_w[0], mw), (bar_b[0], mb)]:
            p_.set_alpha(0.25 + 0.70 * abs(h_) / panel_max)
        ax.errorbar([xw, xb], means, yerr=yerr, fmt='none',
                    color='#222222', capsize=4, lw=1.5, zorder=3)

        ytop = shared_ytop[suf]
        ax.plot([xw, xw, xb, xb], [ytop * 0.97, ytop, ytop, ytop * 0.97],
                lw=1.2, color='#333333')
        star = sig_stars(q_map[(suf, s)])
        ax.text((xw + xb) / 2, ytop * 1.01, star, ha='center', va='bottom',
                fontsize=15, color='#333333',
                fontweight='bold' if star != 'n.s.' else 'normal')

        ax.set_xticks([])
        ax.set_xlim(0.82, 1.53)
        ax.set_ylim(0, ytop * 1.18)
        ax.tick_params(axis='y', labelsize=11)
        ax.spines[['top', 'right']].set_visible(False)

        leftmost = (i % 3 == 0)   # first panel of each metric triplet
        if suf == '_d' and leftmost:
            fmt = ScalarFormatter(useMathText=True)
            fmt.set_scientific(True)
            fmt.set_powerlimits((0, 0))
            ax.yaxis.set_major_formatter(fmt)
            ax.yaxis.get_offset_text().set_fontsize(9)

        # shared scale within each triplet: keep y-tick labels + ylabel only
        # on the leftmost panel of the triplet (removes crowding).
        if leftmost:
            ax.set_ylabel(mlabel, fontsize=13)
        else:
            ax.tick_params(axis='y', labelleft=False)
        # stimulus title (coloured) over the density triplet only
        if i < 3:
            ax.set_title(LAB[s], fontsize=14, fontweight='bold',
                         color=COL[s], pad=6)

    fig.suptitle(
        f'{VAR_LABEL[var]}\n'
        'Within-area connections vs between-area connections, per stimulus type',
        fontsize=13, fontweight='bold', y=1.10)

    # bottom within / between legend (blended transform on first axis)
    plt.tight_layout(rect=[0, 0.08, 1, 0.92])
    ax0 = axes[0]
    blend = mtransforms.blended_transform_factory(ax0.transData, ax0.transAxes)
    bw = 0.11
    for patch_x, hatch, text, ha, text_x in [
        (xw, '',     'Within-area',  'right', xw - bw - 0.04),
        (xb, '////', 'Between-area', 'left',  xb + bw + 0.04),
    ]:
        rect = mpatches.Rectangle((patch_x - bw, -0.20), 2 * bw, 0.12,
                                  facecolor='#999999', edgecolor='#222222',
                                  linewidth=0.8, hatch=hatch,
                                  transform=blend, clip_on=False)
        ax0.add_patch(rect)
        ax0.text(text_x, -0.14, text, ha=ha, va='center', fontsize=11,
                 transform=blend, clip_on=False)

    path = os.path.join(OUT, fname)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved:', path)


# ═════════════════════════════════════════════════════════════════════════════
# FIG 2A  —  16 directed area-pairs, stimuli STACKED (one row per stimulus)
#   Layout mirrors plot_fig2_per_state_barplots_13sess_versionBsafe.py:
#   within solid + between hatched, dashed separator, section labels,
#   EM-matched area order.  Rows = Clip / Monet / Trippy; cols = density/strength.
#   Per-area-pair Friedman-across-stimuli star sits snug above the top-row bars.
# ═════════════════════════════════════════════════════════════════════════════
BAR_WIDTH = 0.70
SECTION_GAP = 0.5


def _x_positions():
    within_x = np.arange(N_WITHIN, dtype=float)
    cur = within_x[-1] + 1.0 + SECTION_GAP
    between_x = []
    for g in BETWEEN_GROUPS:
        between_x.extend([cur, cur + 1.0])
        cur += 2.0
    all_x = np.array(list(within_x) + between_x)
    sep_x = (within_x[-1] + between_x[0]) / 2.0
    return all_x, within_x, np.array(between_x), sep_x


def _two_sided_p(a, b):
    """Arousal EM convention: 2*min(mean(diff>0),mean(diff<0)), floored at 1/n."""
    diff = np.asarray(a) - np.asarray(b)
    diff = diff[np.isfinite(diff)]
    if len(diff) == 0:
        return np.nan
    p = 2.0 * min(np.mean(diff > 0), np.mean(diff < 0))
    return max(p, 1.0 / len(diff))


def _shared_field_bootstrap(var, suf, stim, B=5000):
    """
    Shared field-resampling bootstrap (arousal EM style): every area-pair is
    averaged over the SAME resampled field set each iteration, so pairwise diffs
    are paired.  Returns (point_means[16], boot[B,16]).
    """
    sub = df[df.stim == stim]
    cols = [f'{var}_{a}->{b}{suf}' for (a, b) in ALL_PAIRS]
    V = sub[cols].values                      # (nfields, 16)
    nf = V.shape[0]
    idx = RNG.integers(0, nf, size=(B, nf))
    boot = np.full((B, len(ALL_PAIRS)), np.nan)
    with np.errstate(invalid='ignore'):
        point = np.nanmean(V, axis=0)
        for k in range(len(ALL_PAIRS)):
            samp = V[idx, k]                  # (B, nf)
            allnan = np.all(np.isnan(samp), axis=1)
            bm = np.nanmean(np.where(np.isnan(samp), np.nan, samp), axis=1)
            bm[allnan] = np.nan
            boot[:, k] = bm
    return point, boot


def fig_area_pairs(var, fname):
    all_x, within_x, between_x, sep_x = _x_positions()
    metrics = [('_d', 'Edge density'), ('_s', 'Edge strength')]
    pos = {p: all_x[i] for i, p in enumerate(ALL_PAIRS)}
    n_fields_stim = {s: int((df.stim == s).sum()) for s in STIMS}

    # per (suf,stim): point means, boot dist, CI; ymax shared across stimuli
    point = {}
    boot = {}
    ci_hi = {}
    ci_lo = {}
    ymax = {}
    for suf, _ in metrics:
        tops = []
        for s in STIMS:
            pt, bt = _shared_field_bootstrap(var, suf, s)
            point[(suf, s)] = pt
            boot[(suf, s)] = bt
            with np.errstate(invalid='ignore'):
                lo = np.nanpercentile(bt, 2.5, axis=0)
                hi = np.nanpercentile(bt, 97.5, axis=0)
            ci_lo[(suf, s)] = lo
            ci_hi[(suf, s)] = hi
            tops.extend([h for h in hi if np.isfinite(h)])
        ymax[suf] = max(tops) if tops else 1.0

    fig, axes = plt.subplots(len(STIMS), len(metrics),
                             figsize=(13, 9), squeeze=False)

    for r, s in enumerate(STIMS):
        for c, (suf, ylabel) in enumerate(metrics):
            ax = axes[r][c]
            means = point[(suf, s)]
            lo = ci_lo[(suf, s)]
            hi = ci_hi[(suf, s)]
            yerr_lo = np.where(np.isfinite(means - lo), means - lo, 0)
            yerr_hi = np.where(np.isfinite(hi - means), hi - means, 0)
            bt = boot[(suf, s)]
            ymx = ymax[suf]

            # within (solid) and between (hatched)
            bw = ax.bar(all_x[:N_WITHIN], np.nan_to_num(means[:N_WITHIN]),
                        width=BAR_WIDTH, color=COL[s], edgecolor='#222222',
                        linewidth=0.6, yerr=[yerr_lo[:N_WITHIN], yerr_hi[:N_WITHIN]],
                        capsize=3, error_kw=dict(lw=1.1, color='#222222'), zorder=2)
            bb = ax.bar(all_x[N_WITHIN:], np.nan_to_num(means[N_WITHIN:]),
                        width=BAR_WIDTH, color=COL[s], edgecolor='#222222',
                        linewidth=0.6, hatch='////',
                        yerr=[yerr_lo[N_WITHIN:], yerr_hi[N_WITHIN:]],
                        capsize=3, error_kw=dict(lw=1.1, color='#222222'), zorder=2)
            mx = np.nanmax(np.abs(means)) or 1
            for p_, h_ in list(zip(bw, means[:N_WITHIN])) + list(zip(bb, means[N_WITHIN:])):
                if np.isfinite(h_):
                    p_.set_alpha(0.55 + 0.45 * abs(h_) / mx)

            ax.axvline(sep_x, color='#888888', lw=1.2, ls='--', zorder=1)
            ax.set_xlim(all_x[0] - 0.7, all_x[-1] + 0.7)
            ax.set_ylim(0, ymx * 1.78)
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(axis='y', labelsize=8)
            ax.set_ylabel(ylabel, fontsize=11, color='#222222',
                          fontweight='bold')

            if r == len(STIMS) - 1:
                ax.set_xticks(all_x)
                ax.set_xticklabels(PAIR_LABELS, rotation=55, ha='right', fontsize=8)
            else:
                ax.set_xticks(all_x)
                ax.set_xticklabels([])

            if r == 0:
                trans = ax.get_xaxis_transform()
                ax.text(np.mean(within_x), 1.26, '← within-area →',
                        ha='center', va='bottom', fontsize=9, color='#333333',
                        transform=trans)
                ax.text(np.mean(between_x), 1.26, '← between-area →',
                        ha='center', va='bottom', fontsize=9, color='#333333',
                        transform=trans)

            # ── arousal EM significance brackets (per stimulus row) ───────────
            def bracket(x1, x2, y, h, label):
                ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.1, color='#333')
                ax.text((x1 + x2) / 2, y + h, label, ha='center', va='bottom',
                        fontsize=9, fontweight='bold', color='#333')

            def group_bracket(g1l, g1r, g2l, g2r, y, h, label):
                cx1, cx2 = (g1l + g1r) / 2, (g2l + g2r) / 2
                for seg in ([g1l, g1r], [g2l, g2r]):
                    ax.plot(seg, [y, y], lw=1.1, color='#333', clip_on=False)
                ax.plot([cx1, cx1], [y, y + h], lw=1.1, color='#333', clip_on=False)
                ax.plot([cx1, cx2], [y + h, y + h], lw=1.1, color='#333', clip_on=False)
                ax.plot([cx2, cx2], [y, y + h], lw=1.1, color='#333', clip_on=False)
                ax.text((cx1 + cx2) / 2, y + h * 1.3, label, ha='center',
                        va='bottom', fontsize=10, fontweight='bold', color='#333')

            idx_of = {p: i for i, p in enumerate(ALL_PAIRS)}
            enough = lambda p: np.isfinite(bt[:, idx_of[p]]).sum() > 0 \
                and np.isfinite(hi[idx_of[p]])

            # 1. within: AL->AL vs each other within pair (stacked brackets)
            within_present = [p for p in WITHIN_PAIRS if enough(p)]
            if ('AL', 'AL') in within_present:
                y0 = max(hi[idx_of[p]] for p in within_present)
                hh = ymx * 0.025
                step = hh * 6.5
                others = [p for p in within_present if p != ('AL', 'AL')]
                for j, pr in enumerate(others):
                    pv = _two_sided_p(bt[:, idx_of[('AL', 'AL')]], bt[:, idx_of[pr]])
                    bracket(pos[('AL', 'AL')], pos[pr],
                            y0 + hh + j * step, hh, sig_stars(pv))

            # 2. between: AL<->RL group vs the rest (one two-group bracket)
            focal = [p for p in [('AL', 'RL'), ('RL', 'AL')] if enough(p)]
            rest = [p for p in BETWEEN_PAIRS if p not in focal and enough(p)]
            if focal and rest:
                d_focal = np.nanmean(np.stack([bt[:, idx_of[p]] for p in focal], 1), axis=1)
                d_rest = np.nanmean(np.stack([bt[:, idx_of[p]] for p in rest], 1), axis=1)
                pv = _two_sided_p(d_focal, d_rest)
                yb = max(hi[idx_of[p]] for p in (focal + rest)) + ymx * 0.06
                group_bracket(min(pos[p] for p in rest), max(pos[p] for p in rest),
                              min(pos[p] for p in focal), max(pos[p] for p in focal),
                              yb, ymx * 0.045, sig_stars(pv))

    fig.suptitle(
        f'{VAR_LABEL[var]}\nDirected area-pair FC by stimulus (EM-matched order)  |  '
        'bars = mean ± field-bootstrap 95% CI\n'
        'solid = within-area, hatched = between-area  |  brackets: AL→AL vs within '
        'pairs, and AL↔RL vs rest (two-sided bootstrap p, per stimulus)',
        fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92], h_pad=5.0)
    # one centered stimulus title per row (spanning both metric columns)
    for r, s in enumerate(STIMS):
        pos_l = axes[r][0].get_position()
        pos_r = axes[r][-1].get_position()
        xc = (pos_l.x0 + pos_r.x1) / 2
        fig.text(xc, pos_l.y1 + 0.006, LAB[s], ha='center', va='bottom',
                 color=COL[s], fontsize=15, fontweight='bold')
    path = os.path.join(OUT, fname)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved:', path)


# ── report ───────────────────────────────────────────────────────────────────
def write_report():
    lines = ["# Stimulus-type FC — arousal Fig 1F / Fig 2A style (both variants)\n",
             "Palette: Okabe-Ito no-blue — Clip #E69F00 (orange), Monet "
             "#009E73 (green), Trippy #CC79A7 (reddish-purple). Edge strength "
             "= mean |weight| of thresholded LSCM edges.",
             "Area-pair order matches the arousal Fig 2 EM plot.\n",
             "## Significance (paired within-field, n=124 fields)\n",
             "| variant | metric | scope | Friedman p | Clip-Monet q | Clip-Trippy q | Monet-Trippy q |",
             "|---|---|---|---|---|---|---|"]
    for var in ['A', 'B']:
        for suf, mname in [('_d', 'density'), ('_s', 'strength')]:
            for scope in ['within', 'between']:
                M = paired(f'{var}_{scope}{suf}')
                fr = friedman_p(M)
                q = fdr(pairwise_wilcoxon(M))
                lines.append(f"| {var} | {mname} | {scope} | "
                             f"{fr:.2e} | {q[0]:.2e} | {q[1]:.2e} | {q[2]:.2e} |")
    lines += ["\n## Direction of means (per scope)\n",
              "| variant | metric | scope | Clip | Monet | Trippy | ranking |",
              "|---|---|---|---|---|---|---|"]
    for var in ['A', 'B']:
        for suf, mname in [('_d', 'density'), ('_s', 'strength')]:
            for scope in ['within', 'between']:
                mus = {s: df.loc[df.stim == s, f'{var}_{scope}{suf}'].mean() for s in STIMS}
                order = sorted(STIMS, key=lambda s: -mus[s])
                rank = ' > '.join(LAB[s] for s in order)
                lines.append(f"| {var} | {mname} | {scope} | "
                             f"{mus['clip']:.4g} | {mus['Monet']:.4g} | "
                             f"{mus['Trippy']:.4g} | {rank} |")
    # per-area-pair Friedman across stimuli (kept out of the figure; here only)
    lines += ["\n## Per-area-pair Friedman across stimuli (paired within-field, FDR-BH)\n",
              "This across-stimulus test is a different question from the figure's "
              "within-vs-between brackets and is reported here only.\n",
              "| variant | metric | area-pair | n fields | Friedman p | FDR q |",
              "|---|---|---|---|---|---|"]
    for var in ['A', 'B']:
        for suf, mname in [('_d', 'density'), ('_s', 'strength')]:
            raw, keys, ns = [], [], []
            for (a, b) in ALL_PAIRS:
                M = paired(f'{var}_{a}->{b}{suf}')
                raw.append(friedman_p(M)); keys.append((a, b)); ns.append(M.shape[0])
            q = fdr(raw)
            for (a, b), p, qq, nf in zip(keys, raw, q, ns):
                lines.append(f"| {var} | {mname} | {a}→{b} | {nf} | "
                             f"{p:.2e} | {qq:.2e} |")
    lines += ["\n## Caveats\n",
              "- All three stimuli use exactly 800 within-trial windows per field "
              "(balanced by design, N_MAX=800). Clip had more raw windows available "
              "(5760 vs 920 for Monet/Trippy) but was capped to match; N* plateau ~500 "
              "so 800 is adequate for all three.",
              "- Fields are clustered within scans (pseudo-replication). Figure CIs are "
              "descriptive field-bootstrap; inference is paired within-field "
              "(Friedman + Wilcoxon-FDR).",
              "- Single animal (MICrONS Minnie65); within-brain descriptive only.",
              "- Variant A = CITS-GPU lagged skeleton only; Variant B adds the "
              "PC-contemporaneous edges + LSCM refit (VersionB-safe). Method is "
              "decoupled from the arousal paper's union-neighbor CITS; see report.",
              "- Area-pairs with few contributing fields (e.g. rare cross-area "
              "combinations) have wide CIs and low-power Friedman tests; read those "
              "stars cautiously."]
    with open(os.path.join(OUT, 'report.md'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('Saved:', os.path.join(OUT, 'report.md'))


if __name__ == '__main__':
    fig_within_between('A', 'stim_fc_variantA_within_between.png')
    fig_within_between('B', 'stim_fc_variantB_within_between.png')
    fig_area_pairs('A', 'stim_fc_variantA_area_pairs.png')
    fig_area_pairs('B', 'stim_fc_variantB_area_pairs.png')
    write_report()
    print('Done.')
