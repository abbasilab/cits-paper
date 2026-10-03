import glob, re, os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared')))
from paths import outdir as _outdir, result, microns_saves
OUT=_outdir('fig3_microns_enrichment/data_prep/calcium_npy')   # was <MICRONS_SAVES>/calcium_npy
for f in sorted(glob.glob(microns_saves('merged_df_session*_scan*.pkl'))):
    m=re.search(r'session(\d+)_scan(\d+)',f); s,sc=m.group(1),m.group(2)
    d=pd.read_pickle(f)
    ids=d['ID'].to_numpy(); fields=d['field'].to_numpy()
    np.save(f'{OUT}/ids_session{s}_scan{sc}.npy', ids)
    np.save(f'{OUT}/fields_session{s}_scan{sc}.npy', fields)
    # sanity: row count must match calcium.npy
    # calcium written by _convert_merged.py (falls back to an existing <MICRONS_SAVES>/calcium_npy)
    cal=np.load(result('fig3_microns_enrichment/data_prep', f'calcium_npy/calcium_session{s}_scan{sc}.npy', fallback=microns_saves('calcium_npy', f'calcium_session{s}_scan{sc}.npy')), mmap_mode='r')
    assert cal.shape[0]==len(ids), f"mismatch {f}: cal {cal.shape[0]} ids {len(ids)}"
    print(f"s{s}sc{sc}: {len(ids)} neurons, fields {sorted(set(fields.tolist()))[:8]}, ID dtype {ids.dtype}", flush=True)
print("DONE")
