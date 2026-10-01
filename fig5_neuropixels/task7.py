import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"
import pickle as pkl, numpy as np, sys, time
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.multitest import multipletests

DATA = "/home/rbiswas1/citsproject/data/"
SESS = 791319847
stims = ['natural_scenes', 'static_gratings', 'gabors']

all_p = []
per_stim = {}
for stim in stims:
    mask = pkl.load(open(f"{DATA}ID{SESS}_{stim}_units2use_stim_{stim}.p", "rb"))
    active = np.where(mask)[0]
    # concatenate 60 blocks in order for each active neuron
    blocks = []
    for idx in range(60):
        X = pkl.load(open(f"{DATA}ID{SESS}_{stim}_bin_0.01_X_idx-{idx}.p", "rb"))
        blocks.append(X[:, active])
    series = np.vstack(blocks)  # (T_total, n_active)
    stim_p = []
    t0 = time.time()
    for j in range(series.shape[1]):
        s = series[:, j].astype(float)
        s = s - s.mean()
        sd = s.std()
        s = s / sd if sd > 0 else s
        try:
            p = adfuller(s)[1]
        except Exception:
            p = np.nan
        stim_p.append(p)
    print(f"  [{stim} done, {time.time()-t0:.1f}s]", flush=True)
    stim_p = np.array(stim_p)
    per_stim[stim] = stim_p
    all_p.extend(stim_p.tolist())
    print(f"{stim}: n_active={len(active)}, T_concat={series.shape[0]}, "
          f"raw p<0.05 = {int(np.sum(stim_p<0.05))}/{len(stim_p)}, max_p={np.nanmax(stim_p):.4g}")

all_p = np.array(all_p)
print(f"\nTOTAL series = {len(all_p)}")
print(f"Raw ADF p<0.05 (stationary) = {int(np.sum(all_p<0.05))}/{len(all_p)}")
print(f"Max p among all = {np.nanmax(all_p):.6g}")

rej, p_adj, _, _ = multipletests(all_p, alpha=0.05, method='fdr_bh')
print(f"\nAfter FDR-BH (alpha=0.05):")
print(f"  Total stationary (reject unit-root null) = {int(rej.sum())}/{len(all_p)}")
# per-stim after FDR (global correction, split back)
i = 0
for stim in stims:
    k = len(per_stim[stim])
    print(f"  {stim}: {int(rej[i:i+k].sum())}/{k} stationary after FDR")
    i += k
print(f"  Max FDR-adjusted p = {np.nanmax(p_adj):.6g}")
