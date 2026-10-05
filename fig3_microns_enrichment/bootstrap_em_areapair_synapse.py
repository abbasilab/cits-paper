"""
Per-area-pair EM synapse fraction on the contemporaneous-CITS universe.

Same universe as the FC fig2 barplots and the within/between EM plot: 39 EM
fields, matched units mapped to FC matrix indices, directed pairs (i != j, no
autapses), synapses_..._v1718. For every ordered area pair (src_area ->
tgt_area) counts total matched pairs and synaptically-connected pairs.

Pairs are pooled across fields for the point estimate; the ~39 fields are the
cluster-bootstrap unit (N=5000) for the 95% CIs and pairwise significance.

Outputs (fig2/):
  em_areapair_synapse.csv   per area-pair frac + CI + counts
  em_areapair_synapse.npz   bootstrap frac arrays (for brackets)
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
OUTDIR = _outdir('fig3_microns_enrichment')   # was arousal_paper_overleaf/figures/2026-06-01_versionBsafe/fig2
ALL_GLOB = 'output_*pupil*_session{s}_scan{sc}_field{f}.csv'

# Same pair order as the FC fig2 barplots (plot_fig2_per_state_barplots_..._versionBsafe.py)
UNIQUE_AREAS = ['AL', 'LM', 'RL', 'V1']
WITHIN_PAIRS = [(a, a) for a in UNIQUE_AREAS]
BETWEEN_GROUPS = [('LM', 'RL'), ('AL', 'LM'), ('LM', 'V1'),
                  ('AL', 'V1'), ('RL', 'V1'), ('AL', 'RL')]
BETWEEN_PAIRS = [(a, b) for g in BETWEEN_GROUPS for (a, b) in [(g[0], g[1]), (g[1], g[0])]]
ALL_PAIRS = WITHIN_PAIRS + BETWEEN_PAIRS
PAIR_IDX = {p: i for i, p in enumerate(ALL_PAIRS)}

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
            if ar in UNIQUE_AREAS:
                idx_pt[uid_to_idx[uid]] = pt
                idx_area[uid_to_idx[uid]] = ar
    return idx_pt, idx_area


def per_field_counts(em_keys, matched_df, area_map, sc_pairs):
    """Return array (n_fields, n_pairs, 2) of [tot, syn] and the field keys."""
    rows, keys = [], []
    for (sess, scan, field) in em_keys:
        p = field_matrix_shape(sess, scan, field)
        if p is None:
            continue
        maps = field_maps(sess, scan, field, matched_df, area_map, p)
        if maps is None:
            continue
        idx_pt, idx_area = maps
        m_idx = sorted(idx_pt.keys())
        counts = np.zeros((len(ALL_PAIRS), 2))
        for i in m_idx:
            ai = idx_area[i]
            for j in m_idx:
                if i == j:
                    continue
                pr = (ai, idx_area[j])
                k = PAIR_IDX.get(pr)
                if k is None:
                    continue
                counts[k, 0] += 1
                if (idx_pt[i], idx_pt[j]) in sc_pairs:
                    counts[k, 1] += 1
        rows.append(counts)
        keys.append((sess, scan, field))
    return np.stack(rows), keys


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

    print(f"Computing per-field area-pair counts ({len(em_keys)} fields)...")
    pf, keys = per_field_counts(em_keys, matched_df, area_map, sc_pairs)
    n_fields = pf.shape[0]
    print(f"  fields used: {n_fields}")

    # Aggregate point estimates
    agg = pf.sum(axis=0)                       # (n_pairs, 2)
    frac = np.where(agg[:, 0] > 0, agg[:, 1] / np.maximum(agg[:, 0], 1), np.nan)

    # Field-cluster bootstrap
    boot = np.empty((N_BOOT, len(ALL_PAIRS)))
    for b in range(N_BOOT):
        s = pf[RNG.integers(0, n_fields, size=n_fields)].sum(axis=0)
        boot[b] = np.where(s[:, 0] > 0, s[:, 1] / np.maximum(s[:, 0], 1), np.nan)
    ci_lo = np.nanpercentile(boot, 2.5, axis=0)
    ci_hi = np.nanpercentile(boot, 97.5, axis=0)

    rows = []
    for k, (a, b) in enumerate(ALL_PAIRS):
        rows.append({'src': a, 'tgt': b, 'pair': f'{a}->{b}',
                     'kind': 'within' if a == b else 'between',
                     'tot': int(agg[k, 0]), 'syn': int(agg[k, 1]),
                     'frac': frac[k], 'ci_lo': ci_lo[k], 'ci_hi': ci_hi[k]})
        print(f"  {a:>2}->{b:<2} {'W' if a==b else 'B'}  "
              f"frac={frac[k]*1e3:7.4f}e-3  syn={int(agg[k,1]):>6}  tot={int(agg[k,0]):>9}  "
              f"[{ci_lo[k]*1e3:.4f},{ci_hi[k]*1e3:.4f}]e-3")

    os.makedirs(OUTDIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        os.path.join(OUTDIR, 'em_areapair_synapse.csv'), index=False)
    np.savez(os.path.join(OUTDIR, 'em_areapair_synapse.npz'),
             boot=boot, pairs=np.array([f'{a}->{b}' for a, b in ALL_PAIRS]),
             n_fields=n_fields)
    print(f"\nSaved CSV + NPZ to {OUTDIR}")


if __name__ == '__main__':
    main()
