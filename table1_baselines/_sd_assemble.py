"""Assemble mean +- s.d. for the unified benchmark table from all per-seed sources."""
import numpy as np, pandas as pd, os, glob

DIR = os.path.dirname(os.path.abspath(__file__))
REC = '/home/rbiswas1/microns/arousal_paper_overleaf/figures/2026-06-06_directed_cs'
AR = ['lingauss1', 'lingauss2', 'nonlinnongauss1', 'nonlinnongauss2']
SPK = ['convergence', 'diamond', 'depression']            # tags: conv,cce,depr
TAG = {'convergence': 'conv', 'diamond': 'cce', 'depression': 'depr'}
cells = {}          # (method, paradigm) -> np.array of per-seed CS

def rec(fname, model):
    d = pd.read_csv(os.path.join(REC, f'simulation_results_directed_{fname}.csv'))
    return d[d['model'] == model]['directed_CS'].to_numpy()

# --- AR baselines from recorded CSVs ---
for r in AR:
    cells[('TPC', r)] = rec('tpc_original', r)
    cells[('GC1', r)] = rec('granger', r)
    cells[('KernelGC', r)] = rec('kernel_gc', r)
    cells[('PCMCIplus', r)] = rec('pcmci_plus', r)
    cells[('LPCMCI', r)] = rec('lpcmci', r)
# AR CITS: lingauss from recorded (pcorr); nlng from gpu2 RCIT
_cual=pd.read_csv(os.path.join(DIR,'_sd_ar_lingauss_cupc.csv'))
cells[('CITS','lingauss1')]=_cual[_cual.cell=='CITScupc_lingauss1']['cs'].to_numpy()
cells[('CITS','lingauss2')]=_cual[_cual.cell=='CITScupc_lingauss2']['cs'].to_numpy()

# --- gpu2 RCIT: CITS nlng1/2 + spiking depression ---
g = pd.read_csv(os.path.join(DIR, '_sd_gpu2_rcit.csv'))
cells[('CITS', 'nonlinnongauss1')] = g[g.cell == 'CITS_nonlinnongauss1']['cs'].to_numpy()
cells[('CITS', 'nonlinnongauss2')] = g[g.cell == 'CITS_nonlinnongauss2']['cs'].to_numpy()
cells[('CITS', 'depression')] = g[g.cell == 'CITS_spk_depression']['cs'].to_numpy()

# --- CITS convergence/diamond: faithful neighbor-restricted PC-skeleton + pcorr (cuPC) ---
_cu=pd.read_csv(os.path.join(DIR,'_sd_spk_cits_cupc_convdia.csv'))
cells[('CITS','convergence')]=_cu[_cu.cell=='CITScupc_convergence']['cs'].to_numpy()
cells[('CITS','diamond')]=_cu[_cu.cell=='CITScupc_diamond']['cs'].to_numpy()

# --- local: spiking GC1/GC2/TPC/KernelGC + AR GC2 ---
L = pd.read_csv(os.path.join(DIR, '_sd_local.csv'))
for m in SPK:
    for meth in ['GC1', 'GC2', 'TPC', 'KernelGC']:
        cells[(meth, m)] = L[L.cell == f'{meth}_spk_{TAG[m]}']['cs'].to_numpy()
for r in AR:
    cells[('GC2', r)] = L[L.cell == f'GC2_{r}']['cs'].to_numpy()

# --- tigra: spiking PCMCI+/LPCMCI ---
T = pd.read_csv(os.path.join(DIR, '_sd_tigra.csv'))
for m in SPK:
    cells[('PCMCIplus', m)] = pd.to_numeric(T[T.cell == f'PCMCIplus_spk_{TAG[m]}']['cs'], errors='coerce').dropna().to_numpy()
    cells[('LPCMCI', m)] = pd.to_numeric(T[T.cell == f'LPCMCI_spk_{TAG[m]}']['cs'], errors='coerce').dropna().to_numpy()

METHS = ['CITS', 'TPC', 'GC1', 'GC2', 'KernelGC', 'PCMCIplus', 'LPCMCI']
PARADS = AR + SPK
def fmt(a):
    return f"{a.mean():.3f}$\\pm${a.std():.3f}" if a is not None and len(a) else "NA"
print(f"{'paradigm':18s} " + " ".join(f"{m:>16s}" for m in METHS))
for r in PARADS:
    print(f"{r:18s} " + " ".join(f"{fmt(cells.get((m,r))):>16s}" for m in METHS))
# save
rows=[]
for r in PARADS:
    for m in METHS:
        a=cells.get((m,r))
        rows.append({'paradigm':r,'method':m,'mean':a.mean() if a is not None and len(a) else np.nan,
                     'sd':a.std() if a is not None and len(a) else np.nan,'n':len(a) if a is not None else 0})
pd.DataFrame(rows).to_csv(os.path.join(DIR,'_sd_assembled.csv'),index=False)
print("\nsaved _sd_assembled.csv")
