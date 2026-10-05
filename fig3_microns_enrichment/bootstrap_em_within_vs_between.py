"""
Within-area vs between-area synapse fraction on the contemporaneous-CITS universe.

EM anatomy is method-independent, but this restricts to the SAME pair universe as
the contemporaneous-CITS FC-present/absent enrichment so the two fig1_sc panels are
comparable: the 39 EM-coregistration fields, matched units mapped to the FC
matrix indices, directed pairs (i != j), synapses_..._v1718, cellwise counting.

For each field, every ordered matched pair with two KNOWN areas is classified
within-area (same area) or between-area (different area) and scored for a synapse.
Pairs are pooled across fields for the point estimate; the ~39 fields are the
cluster-bootstrap unit (N=5000) for the 95% CI.

Output (fig1_sc/): sc_within_vs_between_synapse.csv
"""
import os
import sys
import glob
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared')))
from paths import outdir as _outdir, MICRONS_SAVES, arousal_figs, microns_meta
SAVES = os.path.join(MICRONS_SAVES, '')
EM_KEYS_NPZ = arousal_figs('2026-04-29', 'fig4', 'bootstrap_sf_correlation_13sess.npz')
METHOD_DIR = SAVES + 'cits_plus_pc_versionBsafe_2026-05-28/'
OUTDIR = _outdir('fig3_microns_enrichment')   # was arousal_paper_overleaf/figures/2026-06-01_versionBsafe/exp_analysis/fig1_sc
ALL_GLOB = 'output_*pupil*_session{s}_scan{sc}_field{f}.csv'

N_BOOT = 5000
RNG = np.random.default_rng(42)


def field_matrix_shape(sess, scan, field):
    files = sorted(glob.glob(os.path.join(
        METHOD_DIR, ALL_GLOB.format(s=sess, sc=scan, f=field))))
    files = [f for f in files if not f.endswith('_skeleton.csv')
                              and not f.endswith('_edgetype.csv')]
    for fp in files:
        try:
            return pd.read_csv(fp, header=None).shape[0]
        except Exception:
            continue
    return None


def field_maps(sess, scan, field, matched_df, area_map, p):
    """Matrix index -> (pt_root_id, area) for matched units with a known area."""
    pkl = SAVES + f'statement_dfs_session{sess}_scan{scan}.pkl'
    if not os.path.exists(pkl):
        return None
    try:
        with open(pkl, 'rb') as fh:
            names, dfs = pickle.load(fh)
        df = dfs[0] if names else None
        if df is None:
            return None
        full_uids = df[df['field'] == field].sort_values('ID')['ID'].astype(np.int64).values
    except Exception:
        return None
    if len(full_uids) != p:
        return None
    uid_to_idx = {u: i for i, u in enumerate(full_uids)}
    fu = matched_df[(matched_df['session'] == sess) &
                    (matched_df['scan_idx'] == scan) &
                    (matched_df['field'] == field)].sort_values('unit_id')
    idx_pt, idx_area = {}, {}
    for uid, pt in zip(fu['unit_id'].astype(np.int64).values,
                       fu['pt_root_id_v1718'].astype(np.int64).values):
        if uid in uid_to_idx:
            ar = area_map.get((sess, scan, int(uid)))
            if ar is not None:
                idx_pt[uid_to_idx[uid]] = pt
                idx_area[uid_to_idx[uid]] = ar
    return idx_pt, idx_area


def per_field_counts(em_keys, matched_df, area_map, sc_pairs):
    rows = []
    for (sess, scan, field) in em_keys:
        p = field_matrix_shape(sess, scan, field)
        if p is None:
            continue
        maps = field_maps(sess, scan, field, matched_df, area_map, p)
        if maps is None:
            continue
        idx_pt, idx_area = maps
        m_idx = sorted(idx_pt.keys())
        wtot = wsyn = btot = bsyn = 0
        for i in m_idx:
            for j in m_idx:
                if i == j:
                    continue
                sc = (idx_pt[i], idx_pt[j]) in sc_pairs
                if idx_area[i] == idx_area[j]:
                    wtot += 1; wsyn += int(sc)
                else:
                    btot += 1; bsyn += int(sc)
        rows.append({'field_key': (sess, scan, field),
                     'w_tot': wtot, 'w_syn': wsyn,
                     'b_tot': btot, 'b_syn': bsyn})
    return pd.DataFrame(rows)


def rates(agg):
    wtot, wsyn, btot, bsyn = agg
    wr = wsyn / wtot if wtot else np.nan
    br = bsyn / btot if btot else np.nan
    ratio = wr / br if br else np.nan
    return wr, br, ratio


def bootstrap(pf, n_boot=N_BOOT):
    arr = pf[['w_tot', 'w_syn', 'b_tot', 'b_syn']].to_numpy()
    n = len(arr)
    wr = np.empty(n_boot); br = np.empty(n_boot); rt = np.empty(n_boot)
    for i in range(n_boot):
        agg = arr[RNG.integers(0, n, size=n)].sum(axis=0)
        wr[i], br[i], rt[i] = rates([agg[0], agg[1], agg[2], agg[3]])
    ci = lambda a: (np.nanpercentile(a, 2.5), np.nanpercentile(a, 97.5))
    return ci(wr), ci(br), ci(rt)


def main():
    print("Loading EM field keys + matched_df + areas + synapse table...")
    d = np.load(EM_KEYS_NPZ, allow_pickle=True)
    em_keys = [tuple(int(x) for x in k) for k in d['field_keys']]
    matched_df = pickle.load(open(SAVES + 'matched_df_v1718.pkl', 'rb'))
    matched_df = matched_df.dropna(subset=['pt_root_id_v1718']).copy()
    matched_df['pt_root_id_v1718'] = matched_df['pt_root_id_v1718'].astype(np.int64)
    areas = pd.read_csv(microns_meta('all_unit_areas.csv'))
    area_map = {(int(s), int(sc), int(u)): a for s, sc, u, a in
                areas[['session', 'scan_idx', 'unit_id', 'brain_area']].itertuples(index=False)}
    syn = pickle.load(open(SAVES + 'synapses_matcheddf_frompre_v1718.pkl', 'rb'))
    sc_pairs = set()
    for pre_id, df_s in syn.items():
        if hasattr(df_s, 'columns') and 'post_pt_root_id' in df_s.columns:
            for post in df_s['post_pt_root_id'].astype(np.int64).values:
                sc_pairs.add((int(pre_id), int(post)))

    print(f"Computing per-field within/between counts ({len(em_keys)} fields)...")
    pf = per_field_counts(em_keys, matched_df, area_map, sc_pairs)
    print(f"  fields used: {len(pf)}")

    agg = pf[['w_tot', 'w_syn', 'b_tot', 'b_syn']].sum().to_numpy()
    wr, br, ratio = rates(agg)
    (wr_lo, wr_hi), (br_lo, br_hi), (rt_lo, rt_hi) = bootstrap(pf)
    row = {
        'n_fields': len(pf),
        'w_tot': int(agg[0]), 'w_syn': int(agg[1]),
        'b_tot': int(agg[2]), 'b_syn': int(agg[3]),
        'within_rate': wr, 'within_ci_lo': wr_lo, 'within_ci_hi': wr_hi,
        'between_rate': br, 'between_ci_lo': br_lo, 'between_ci_hi': br_hi,
        'ratio': ratio, 'ratio_ci_lo': rt_lo, 'ratio_ci_hi': rt_hi,
    }
    print(f"  within  = {wr*100:.4f}%  ({int(agg[1])}/{int(agg[0])})  "
          f"[{wr_lo*100:.4f},{wr_hi*100:.4f}]")
    print(f"  between = {br*100:.4f}%  ({int(agg[3])}/{int(agg[2])})  "
          f"[{br_lo*100:.4f},{br_hi*100:.4f}]")
    print(f"  ratio (within/between) = {ratio:.2f}  [{rt_lo:.2f},{rt_hi:.2f}]")

    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, 'sc_within_vs_between_synapse.csv')
    pd.DataFrame([row]).to_csv(out, index=False)
    pf.to_csv(os.path.join(OUTDIR, 'sc_within_vs_between_perfield.csv'),
              index=False)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
