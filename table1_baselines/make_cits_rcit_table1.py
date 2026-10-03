"""CITS column of Table 1 for the RCIT cells (non-linear AR and both spiking blocks), with the
packaged cits.cits_rcit (cits >= 1.9.0), uncapped conditioning, K=25, alpha=0.05, tau=1, seeds 0-49.
Writes $CITS_PAPER_OUT/table1_baselines/cits_rcit_uncapped_perseed.csv (the committed copy is in
source_data/) and prints mean +/- s.d. per cell and the
self-edge rates. Device: env CITS_DEV (default cuda:0; 'cpu' works, slower).
Run from this folder with ../shared on PYTHONPATH."""
import os, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'shared'))
from paths import out as _out
from cits import cits_rcit
from sim_scm import simulate_extended
from directed_metrics import compute_directed_metrics
DEV = os.environ.get('CITS_DEV', 'cuda:0'); NS = 50; T = 2000; p = 4
MOTIFS = {'convergence': [(0,2),(1,2),(2,3)], 'diamond': [(0,1),(0,2),(1,3),(2,3)],
          'chain': [(0,1),(1,2),(2,3)], 'depression': [(0,2),(1,2),(2,3)]}
def sim(seed, motif, SH):   # verbatim from _spiking_pmatched_cits.py
    rng = np.random.default_rng(seed); W = np.zeros((p,p))
    for a,b in MOTIFS[motif]: W[a,b] = 1.4
    b_i = np.log(0.25); k1, k2 = 1.0, 0.4; s = np.zeros((p,T)); R = np.ones(p); U, tr, dep = 0.7, 15.0, 2.2
    for t in range(1,T):
        p1 = s[:,t-1]; p2 = s[:,t-2] if t >= 2 else np.zeros(p)
        if motif == 'depression':
            drive = dep*(W.T@(U*R*p1)); R = R+(1-R)/tr; R = np.clip(R-U*R*p1,0,1)
        else: drive = W.T@(k1*p1+k2*p2)
        eta = b_i+drive+SH*p1; s[:,t] = rng.poisson(np.clip(np.exp(eta),0,4.0))
    GTs = np.eye(p,dtype=int) if SH != 0 else np.zeros((p,p),dtype=int)
    return s.astype(float), (W != 0).astype(int), GTs
def cs(pred, GT):
    nt = int(GT.sum()); nn = p*p-nt
    TP = int(((pred > 0) & (GT == 1)).sum()); FP = int(((pred > 0) & (GT == 0)).sum())
    return (TP/nt if nt else 0)-(FP/nn if nn else 0)
rows = []
for reg in ['nonlinnongauss1', 'nonlinnongauss2']:
    for s in range(NS):
        o = simulate_extended(reg, 1.0, 1000, s); X = o[0].astype(np.float64)
        B = (cits_rcit(X, seed=s, device=DEV) != 0).astype(int); np.fill_diagonal(B, 0)
        _, gl_uw, gl_w, gc_uw, gc_w, gb_uw, gb_lw, gb_cw = o
        rows.append(dict(block='autoregressive', cell=reg, seed=s, cs=compute_directed_metrics(
            B, gl_w, gc_w, gb_lw, gb_cw, gl_uw, gc_uw, gb_uw)['directed_CS_strict'], self_frac=np.nan))
for SH, block in ((0.0, 'control'), (-1.5, 'recurrent')):
    for motif in ['convergence', 'diamond', 'chain'] + (['depression'] if SH != 0 else []):
        for s in range(NS):
            X, GTc, GTs = sim(s, motif, SH)
            B = (cits_rcit(X, seed=s, device=DEV) != 0).astype(int)
            rows.append(dict(block=block, cell=motif, seed=s, cs=cs(B, ((GTc+GTs) > 0).astype(int)),
                             self_frac=float(np.mean(np.diag(B) > 0))))
df = pd.DataFrame(rows)
os.makedirs(os.path.join(HERE, 'source_data'), exist_ok=True)
df.to_csv(_out('table1_baselines', 'cits_rcit_uncapped_perseed.csv'), index=False)
print(df.groupby(['block', 'cell'], sort=False)['cs'].agg(['mean', 'std']).round(3))
print(df.groupby('block', sort=False)['cs'].mean().round(3))
print('self-edge rate:', df.groupby('block', sort=False)['self_frac'].mean().round(4).to_dict())
