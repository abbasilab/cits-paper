import glob, re, numpy as np, pandas as pd
S='/data1/rb1/microns/saves'; O='/data1/rb1/microns/saves/calcium_npy'
scans=set()
for f in glob.glob(f'{S}/merged_df_session*_scan*.pkl'):
    m=re.search(r'session(\d+)_scan(\d+)',f); scans.add((m.group(1),m.group(2)))
for s,sc in sorted(scans):
    try:
        ids=[]
        for stim in ['clip','Monet','Trippy']:
            d=pd.read_pickle(f'{S}/filtered_{stim}_neurons_session{s}_scan{sc}.pkl')
            ids.append(np.asarray(d['ID']))
        union=np.unique(np.concatenate(ids))
        np.save(f'{O}/unionids_session{s}_scan{sc}.npy', union)
        print(f"s{s}sc{sc}: union_ids={len(union)}", flush=True)
    except Exception as e:
        print(f"s{s}sc{sc}: ERR {str(e)[:100]}", flush=True)
print("DONE")
