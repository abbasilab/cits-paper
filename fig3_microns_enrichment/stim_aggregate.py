"""Aggregate stim-FC sweep: field-level bootstrap CIs + Friedman + pairwise
Wilcoxon (FDR-BH), and the FOUR figures (A/B x withinbetween/areapairs)."""
import os, sys, glob, itertools
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared')))
from paths import outdir as _outdir

OUT = _outdir('fig3_microns_enrichment/out')
FIG = f'{OUT}/figures'; os.makedirs(FIG, exist_ok=True)
STIMS = ['clip', 'Monet', 'Trippy']
SLAB = {'clip': 'Clip', 'Monet': 'Monet', 'Trippy': 'Trippy'}
COL = {'clip': '#0173B2', 'Monet': '#DE8F05', 'Trippy': '#029E73'}
AREAS = ['AL', 'LM', 'RL', 'V1']
NBOOT = 5000
KEY = ['session', 'scan', 'field']

plt.rcParams.update({'font.size': 8, 'axes.linewidth': 0.8, 'font.family': 'sans-serif',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'xtick.major.width': 0.8, 'ytick.major.width': 0.8, 'figure.dpi': 150})

df = pd.concat([pd.read_csv(f) for f in glob.glob(f'{OUT}/results_gpu*_shard*.csv')], ignore_index=True)
df = df.drop_duplicates(subset=KEY + ['stim'])
print(f"loaded {len(df)} rows; fields/stim:", df.groupby('stim').size().to_dict())

def rng(seed): return np.random.default_rng(seed)

def boot_ci(vals, seed=0):
    vals = np.asarray(vals, float); vals = vals[~np.isnan(vals)]
    if len(vals) < 2: return (np.nan, np.nan, np.nan, len(vals))
    r = rng(seed)
    means = [np.nanmean(r.choice(vals, len(vals), replace=True)) for _ in range(NBOOT)]
    return (float(np.mean(vals)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), len(vals))

def paired_stats(sub, col):
    piv = sub.pivot_table(index=KEY, columns='stim', values=col)
    piv = piv.dropna(subset=STIMS)
    n = len(piv)
    res = {'n_paired': n}
    if n < 3:
        res['friedman_p'] = np.nan
        for a, b in itertools.combinations(STIMS, 2): res[f'w_{a}_{b}_p'] = np.nan
        return res, piv
    try:
        res['friedman_stat'], res['friedman_p'] = stats.friedmanchisquare(*[piv[s].values for s in STIMS])
    except Exception:
        res['friedman_p'] = np.nan
    for a, b in itertools.combinations(STIMS, 2):
        try:
            d = piv[a].values - piv[b].values
            if np.allclose(d, 0): res[f'w_{a}_{b}_p'] = 1.0
            else: _, res[f'w_{a}_{b}_p'] = stats.wilcoxon(piv[a].values, piv[b].values)
        except Exception:
            res[f'w_{a}_{b}_p'] = np.nan
    return res, piv

# ---- metric list ----
BASE = ['within_d', 'within_s', 'between_d', 'between_s', 'overall_d', 'overall_s']
PAIRM = [f'{a}->{b}' for a in AREAS for b in AREAS]

summary_rows, stats_rows = [], []
for var in ['A', 'B']:
    for m in BASE + [f'{p}_d' for p in PAIRM] + [f'{p}_s' for p in PAIRM]:
        col = f'{var}_{m}'
        if col not in df.columns: continue
        for si, st in enumerate(STIMS):
            sub = df[df.stim == st]
            mean, lo, hi, n = boot_ci(sub[col].values, seed=1000 * si + hash(col) % 997)
            summary_rows.append({'variant': var, 'metric': m, 'stim': st,
                                 'mean': mean, 'ci_lo': lo, 'ci_hi': hi, 'n_fields': n})
        res, _ = paired_stats(df, col)
        stats_rows.append({'variant': var, 'metric': m, **res})

summ = pd.DataFrame(summary_rows); summ.to_csv(f'{OUT}/summary.csv', index=False)
stt = pd.DataFrame(stats_rows)
# FDR across pairwise within each variant
for var in ['A', 'B']:
    mask = stt.variant == var
    for a, b in itertools.combinations(STIMS, 2):
        c = f'w_{a}_{b}_p'; idx = stt.index[mask & stt[c].notna()]
        if len(idx):
            stt.loc[idx, f'w_{a}_{b}_fdr'] = multipletests(stt.loc[idx, c], method='fdr_bh')[1]
stt.to_csv(f'{OUT}/stats.csv', index=False)
print("wrote summary.csv, stats.csv")

def star(p):
    if p is None or (isinstance(p, float) and np.isnan(p)): return ''
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else ''

def get(var, m, st, field):
    r = summ[(summ.variant == var) & (summ.metric == m) & (summ.stim == st)]
    return r[field].values[0] if len(r) else np.nan

# ---------- Figure 1/2: within vs between (density + strength) ----------
def fig_withinbetween(var):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))
    for ax, (kind, mets, ylab) in zip(axes, [
            ('Density', ['within_d', 'between_d'], 'Edge density'),
            ('Strength', ['within_s', 'between_s'], 'Mean |weight|')]):
        x = np.arange(2); wbar = 0.25
        for si, st in enumerate(STIMS):
            means = [get(var, m, st, 'mean') for m in mets]
            los = [get(var, m, st, 'mean') - get(var, m, st, 'ci_lo') for m in mets]
            his = [get(var, m, st, 'ci_hi') - get(var, m, st, 'mean') for m in mets]
            ax.bar(x + (si - 1) * wbar, means, wbar, yerr=[los, his], capsize=2,
                   color=COL[st], label=SLAB[st], error_kw={'lw': 0.8})
        # significance (Friedman) above each group
        for xi, m in enumerate(mets):
            fp = stt[(stt.variant == var) & (stt.metric == m)]['friedman_p'].values
            s = star(fp[0]) if len(fp) else ''
            if s:
                top = max(get(var, m, st, 'ci_hi') for st in STIMS)
                ax.text(xi, top * 1.05, s, ha='center', va='bottom', fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(['Within-area', 'Between-area'])
        ax.set_ylabel(ylab); ax.set_title(kind, fontsize=9)
        if kind == 'Density': ax.legend(frameon=False, fontsize=7, loc='upper right')
    fig.suptitle(f'Variant {var}: within- vs between-area FC by stimulus', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fp = f'{FIG}/fig{var}_withinbetween.png'; fig.savefig(fp, bbox_inches='tight'); plt.close(fig)
    return fp

# ---------- Figure 3/4: 16 directed area-pairs (heatmap grid) ----------
def fig_areapairs(var, kind='d'):
    lab = 'density' if kind == 'd' else 'strength'
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.2))
    mats = {}
    for st in STIMS:
        M = np.full((4, 4), np.nan)
        for i, a in enumerate(AREAS):
            for j, b in enumerate(AREAS):
                M[i, j] = get(var, f'{a}->{b}_{kind}', st, 'mean')
        mats[st] = M
    vmax = np.nanmax([np.nanmax(m) for m in mats.values()]) or 1.0
    for ax, st in zip(axes, STIMS):
        im = ax.imshow(mats[st], cmap='magma', vmin=0, vmax=vmax, aspect='equal')
        ax.set_xticks(range(4)); ax.set_xticklabels(AREAS, fontsize=7)
        ax.set_yticks(range(4)); ax.set_yticklabels(AREAS, fontsize=7)
        ax.set_title(SLAB[st], fontsize=9)
        ax.set_xlabel('target');
        if st == STIMS[0]: ax.set_ylabel('source')
        # star significant pairs (Friedman)
        for i, a in enumerate(AREAS):
            for j, b in enumerate(AREAS):
                fp = stt[(stt.variant == var) & (stt.metric == f'{a}->{b}_{kind}')]['friedman_p'].values
                if len(fp) and star(fp[0]):
                    ax.text(j, i, '*', ha='center', va='center', color='cyan', fontsize=8)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02); cb.set_label(f'{lab}')
    fig.suptitle(f'Variant {var}: directed area-pair {lab} by stimulus (* Friedman p<0.05)', fontsize=10)
    fp = f'{FIG}/fig{var}_areapairs_{kind}.png'; fig.savefig(fp, bbox_inches='tight'); plt.close(fig)
    return fp

made = []
for var in ['A', 'B']:
    made.append(fig_withinbetween(var))
    made.append(fig_areapairs(var, 'd'))
    made.append(fig_areapairs(var, 's'))
print("figures:", *made, sep='\n  ')
