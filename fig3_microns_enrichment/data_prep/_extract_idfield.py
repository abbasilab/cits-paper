import glob, re, os, numpy as np, pandas as pd
OUT='/data1/rb1/microns/saves/calcium_npy'
for f in sorted(glob.glob('/data1/rb1/microns/saves/merged_df_session*_scan*.pkl')):
    m=re.search(r'session(\d+)_scan(\d+)',f); s,sc=m.group(1),m.group(2)
    d=pd.read_pickle(f)
    ids=d['ID'].to_numpy(); fields=d['field'].to_numpy()
    np.save(f'{OUT}/ids_session{s}_scan{sc}.npy', ids)
    np.save(f'{OUT}/fields_session{s}_scan{sc}.npy', fields)
    # sanity: row count must match calcium.npy
    cal=np.load(f'{OUT}/calcium_session{s}_scan{sc}.npy', mmap_mode='r')
    assert cal.shape[0]==len(ids), f"mismatch {f}: cal {cal.shape[0]} ids {len(ids)}"
    print(f"s{s}sc{sc}: {len(ids)} neurons, fields {sorted(set(fields.tolist()))[:8]}, ID dtype {ids.dtype}", flush=True)
print("DONE")
