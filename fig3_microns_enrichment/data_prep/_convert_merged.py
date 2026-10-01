import sys, time, re, os, numpy as np, pandas as pd
f = sys.argv[1]
m = re.search(r'session(\d+)_scan(\d+)', f); s, sc = m.group(1), m.group(2)
outdir = '/data1/rb1/microns/saves/calcium_npy'; os.makedirs(outdir, exist_ok=True)
t0 = time.time()
df = pd.read_pickle(f)
t1 = time.time()
cols = list(df.columns)
cal_cols = [c for c in cols if str(c).startswith('calcium')]
print(f"loaded in {t1-t0:.0f}s | df.shape={df.shape} | n_calcium_cols={len(cal_cols)} | sample cols={[str(c) for c in cols[:3]]} ... calcium sample={[str(c) for c in cal_cols[:2]]}", flush=True)
# rows = neurons; calcium cols = timepoints  -> (neurons, T)
arr = df[cal_cols].to_numpy(dtype=np.float32)
# unit ordering
uid = df['unit_id'].to_numpy() if 'unit_id' in df.columns else np.asarray(df.index)
np.save(f'{outdir}/calcium_session{s}_scan{sc}.npy', arr)
np.save(f'{outdir}/units_session{s}_scan{sc}.npy', uid)
print(f"SAVED calcium ({arr.shape}, {arr.nbytes/1e9:.2f}G) + units ({uid.shape}) | total {time.time()-t0:.0f}s", flush=True)
