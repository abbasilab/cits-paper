"""
Fig 7 panel A (single-method, OUR cits-gpu Version B).

FC-presence vs EM-based synaptic connectivity, pooled overall across the 3
stimuli. Mirrors the arousal versionBsafe enrichment
(analysis/functional_circuitry/bootstrap_sc_enrichment_per_state_versionBsafe.py,
state='all'): same EM field keys, matched-df units mapped to FC matrix indices,
directed pairs (i != j), cellwise counting, field-cluster bootstrap (N=5000) for
the fold CI. The ONLY change: the FC edge set is our stimulus Version-B fcB
(cuPC lagged + PC-contemporaneous + union + LSCM refit), unioned over the 3
stimulus graphs per field (= the mean|FC|!=0 'all' analog).

FC matrices + calcium metadata were staged from gpu-2; EM data is local.
(Paths now come from shared/paths.py: FC from $CITS_PAPER_OUT/fig3_microns_enrichment/out/fc.)
"""
import os, sys, glob
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.patches as mpatches
from scipy.stats import fisher_exact
from statsmodels.stats.proportion import proportion_confint

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared')))
from paths import out as _out, OUT_ROOT, MICRONS_SAVES, microns_saves, arousal_figs

# Version-B FC written by stim_run.py (was staged from gpu-2 into a session scratchpad);
# calcium metadata (ids/fields/unionids) from data_prep/ in $MICRONS_SAVES/calcium_npy.
FCDIR, NPYDIR = os.path.join(OUT_ROOT, 'fig3_microns_enrichment', 'out', 'fc'), microns_saves('calcium_npy')
SAVES = os.path.join(MICRONS_SAVES, '')
EM_KEYS_NPZ = arousal_figs('2026-04-29', 'fig4', 'bootstrap_sf_correlation_13sess.npz')
OUT_PNG = _out('fig3_microns_enrichment', 'panelA_fc_em_enrichment_versionB.png')
OUT_CSV = _out('fig3_microns_enrichment', 'panelA_fc_em_enrichment_versionB.csv')
STIMS = ['clip', 'Monet', 'Trippy']
FC_COLOR = '#0072B2'   # Okabe-Ito blue (FC-present); colorblind-safe, distinct from B-E trio
ABS_COLOR = '#b8b8b8'
N_BOOT = 5000
RNG = np.random.default_rng(42)


def nids_for_field(s, sc, field):
    ids = np.load(f'{NPYDIR}/ids_session{s}_scan{sc}.npy')
    fields = np.load(f'{NPYDIR}/fields_session{s}_scan{sc}.npy')
    union = set(np.load(f'{NPYDIR}/unionids_session{s}_scan{sc}.npy').tolist())
    idx = [i for i in range(len(ids)) if fields[i] == field and int(ids[i]) in union]
    idx.sort(key=lambda i: int(ids[i]))
    return [int(ids[i]) for i in idx]


def fc_present_union(s, sc, field, p):
    """OR of fcB!=0 across available stimulus graphs (= mean|FC|!=0 'all' analog)."""
    present = np.zeros((p, p), dtype=bool)
    got = 0
    for st in STIMS:
        fp = f'{FCDIR}/s{s}sc{sc}f{field}_{st}.npz'
        if not os.path.exists(fp):
            continue
        m = np.load(fp)['fcB']
        if m.shape != (p, p):
            continue
        present |= (m != 0)
        got += 1
    return (present if got else None), got


def counts_from_field(s, sc, field, matched_df, sc_pairs):
    nids = nids_for_field(s, sc, field)
    p = len(nids)
    if p < 2:
        return None
    present, got = fc_present_union(s, sc, field, p)
    if present is None:
        return None
    id_to_idx = {u: i for i, u in enumerate(nids)}
    fu = matched_df[(matched_df['session'] == s) &
                    (matched_df['scan_idx'] == sc) &
                    (matched_df['field'] == field)]
    idx_to_pt = {}
    for u, pt in zip(fu['unit_id'].astype(np.int64).values,
                     fu['pt_root_id_v1718'].astype(np.int64).values):
        if int(u) in id_to_idx:
            idx_to_pt[id_to_idx[int(u)]] = int(pt)
    m_idx = sorted(idx_to_pt.keys())
    fcp = fps = fca = fas = 0
    for i in m_idx:
        for j in m_idx:
            if i == j:
                continue
            fc = bool(present[i, j])
            sc_syn = (idx_to_pt[i], idx_to_pt[j]) in sc_pairs
            if fc:
                fcp += 1; fps += int(sc_syn)
            else:
                fca += 1; fas += int(sc_syn)
    return {'field_key': (s, sc, field), 'n_matched': len(m_idx), 'n_stim': got,
            'fc_plus': fcp, 'fcplus_scplus': fps,
            'fc_minus': fca, 'fcminus_scplus': fas}


def rates(agg):
    fcp, fps, fca, fas = agg
    pr = fps / fcp if fcp else np.nan
    ar = fas / fca if fca else np.nan
    fold = pr / ar if ar else np.nan
    return pr, ar, fold


def stars(p):
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else 'n.s.'


def main():
    print("Loading EM keys, matched_df, synapses ...")
    d = np.load(EM_KEYS_NPZ, allow_pickle=True)
    em_keys = [tuple(int(x) for x in k) for k in d['field_keys']]
    matched_df = pickle.load(open(SAVES + 'matched_df_v1718.pkl', 'rb'))
    matched_df = matched_df.dropna(subset=['pt_root_id_v1718']).copy()
    matched_df['pt_root_id_v1718'] = matched_df['pt_root_id_v1718'].astype(np.int64)
    syn = pickle.load(open(SAVES + 'synapses_matcheddf_frompre_v1718.pkl', 'rb'))
    sc_pairs = set()
    for pre_id, df_s in syn.items():
        if hasattr(df_s, 'columns') and 'post_pt_root_id' in df_s.columns:
            for post in df_s['post_pt_root_id'].astype(np.int64).values:
                sc_pairs.add((int(pre_id), int(post)))
    print(f"  em_keys={len(em_keys)}  sc_pairs={len(sc_pairs)}")

    rows = []
    for (s, sc, field) in em_keys:
        r = counts_from_field(s, sc, field, matched_df, sc_pairs)
        if r is not None:
            rows.append(r)
    pf = pd.DataFrame(rows)
    print(f"  usable fields = {len(pf)}")
    pf.to_csv(_out('fig3_microns_enrichment', 'panelA_perfield_counts.csv'), index=False)

    agg = pf[['fc_plus', 'fcplus_scplus', 'fc_minus', 'fcminus_scplus']].sum().to_numpy()
    fcp, fps, fca, fas = (int(x) for x in agg)
    pr, ar, fold = rates(agg)

    # Katz log-RR 95% CI (a=fps,n1=fcp,c=fas,n2=fca), matching TASK 6 convention
    se = np.sqrt(1/fps - 1/fcp + 1/fas - 1/fca)
    katz_lo = np.exp(np.log(fold) - 1.96*se); katz_hi = np.exp(np.log(fold) + 1.96*se)
    print(f"  KATZ 95% CI = ({katz_lo:.4f}, {katz_hi:.4f})")

    # field-cluster bootstrap for fold CI
    arr = pf[['fc_plus', 'fcplus_scplus', 'fc_minus', 'fcminus_scplus']].to_numpy()
    n = len(arr)
    fb = np.empty(N_BOOT)
    for i in range(N_BOOT):
        a = arr[RNG.integers(0, n, size=n)].sum(axis=0)
        fb[i] = rates(a)[2]
    fold_lo, fold_hi = np.nanpercentile(fb, 2.5), np.nanpercentile(fb, 97.5)

    # Wilson CIs on the pooled proportions
    pr_lo, pr_hi = proportion_confint(fps, fcp, alpha=0.05, method='wilson')
    ar_lo, ar_hi = proportion_confint(fas, fca, alpha=0.05, method='wilson')
    _, fisher_p = fisher_exact([[fps, fcp - fps], [fas, fca - fas]])

    print(f"\n  FC-present: {fps}/{fcp} = {pr*100:.3f}%  [{pr_lo*100:.3f},{pr_hi*100:.3f}]")
    print(f"  FC-absent : {fas}/{fca} = {ar*100:.3f}%  [{ar_lo*100:.3f},{ar_hi*100:.3f}]")
    print(f"  fold = {fold:.2f}x  [{fold_lo:.2f},{fold_hi:.2f}]   Fisher p = {fisher_p:.2e}  {stars(fisher_p)}")

    pd.DataFrame([{
        'n_fields': len(pf),
        'fc_plus': fcp, 'fcplus_scplus': fps, 'present_rate': pr,
        'present_ci_lo': pr_lo, 'present_ci_hi': pr_hi,
        'fc_minus': fca, 'fcminus_scplus': fas, 'absent_rate': ar,
        'absent_ci_lo': ar_lo, 'absent_ci_hi': ar_hi,
        'fold': fold, 'fold_ci_lo': fold_lo, 'fold_ci_hi': fold_hi,
        'fisher_p': fisher_p,
    }]).to_csv(OUT_CSV, index=False)

    return
    # ---- render (arousal Fig 1E single-method style) ----
    xw, xb = 1.0, 1.5
    fig, ax = plt.subplots(figsize=(3.0, 4.2))
    ax.bar(xw, pr * 1e2, width=0.36, color=FC_COLOR, edgecolor='#222222', lw=0.8, zorder=2)
    ax.bar(xb, ar * 1e2, width=0.36, color=ABS_COLOR, edgecolor='#222222', lw=0.8, zorder=2)
    ax.errorbar([xw, xb], [pr * 1e2, ar * 1e2],
                yerr=[[(pr - pr_lo) * 1e2, (ar - ar_lo) * 1e2],
                      [(pr_hi - pr) * 1e2, (ar_hi - ar) * 1e2]],
                fmt='none', color='#222222', capsize=4, lw=1.5, zorder=3)
    y_top = max(pr_hi, ar_hi) * 1e2 * 1.08
    ax.plot([xw, xw, xb, xb], [y_top * 0.985, y_top, y_top, y_top * 0.985], lw=1.2, color='#333')
    ax.text((xw + xb) / 2, y_top * 1.015, f'{stars(fisher_p)}', ha='center', va='bottom',
            fontsize=15, fontweight='bold', color='#333')
    ax.text((xw + xb) / 2, y_top * 1.11, f'{fold:.1f}×', ha='center', va='bottom',
            fontsize=13, fontweight='bold', color='#333')
    ax.set_xticks([])
    ax.set_xlim(0.72, 1.78)
    ax.set_ylim(0, y_top * 1.28)
    ax.set_ylabel('Fraction of neuron pairs\nwith a synapse  (×10$^{-2}$)', fontsize=13)
    ax.tick_params(axis='y', labelsize=12)
    ax.spines[['top', 'right']].set_visible(False)

    blend = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    bw = 0.18
    for px, col, txt, ha, tx in [(xw, FC_COLOR, 'FC-present', 'right', xw - bw - 0.04),
                                 (xb, ABS_COLOR, 'FC-absent', 'left', xb + bw + 0.04)]:
        ax.add_patch(mpatches.Rectangle((px - bw, -0.14), 2 * bw, 0.075, facecolor=col,
                     edgecolor='#222', lw=0.8, transform=blend, clip_on=False))
        ax.text(tx, -0.103, txt, ha=ha, va='center', fontsize=11, transform=blend, clip_on=False)
    # count rows
    for yy, lab, va, vb in [(-0.24, 'synaptically\nconnected', fps, fas),
                            (-0.35, 'neuron pairs', fcp, fca)]:
        ax.text(0.58, yy, lab, ha='right', va='center', fontsize=8, style='italic',
                color='#555', transform=blend, clip_on=False)
        ax.text(xw, yy, f'{va:,}', ha='center', va='center', fontsize=8.5, color='#333',
                transform=blend, clip_on=False)
        ax.text(xb, yy, f'{vb:,}', ha='center', va='center', fontsize=8.5, color='#333',
                transform=blend, clip_on=False)
    ax.set_title('FC presence vs.\nEM-based synaptic connectivity', fontsize=12, fontweight='bold')
    fig.subplots_adjust(bottom=0.30, top=0.86)
    plt.savefig(OUT_PNG, dpi=170, bbox_inches='tight', facecolor='white')
    print(f"\nsaved: {OUT_PNG}\nsaved: {OUT_CSV}")


if __name__ == '__main__':
    main()
