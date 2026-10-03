import numpy as np, sys, os, warnings, csv; warnings.filterwarnings('ignore')
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared')))
from paths import out as _out
from sim_scm import simulate_extended
from glm_spiking_sim import simulate_glm_spiking, directed_cs as gdcs
from directed_metrics import compute_directed_metrics
from cits import cits_rcit  # cits >= 1.9.0 (pip install cits[rcit]); uncapped conditioning, as in the paper
def dcs_ar(pred,out):
    _,gl_uw,gl_w,gc_uw,gc_w,gb_uw,gb_lw,gb_cw=out
    return compute_directed_metrics(pred,gl_w,gc_w,gb_lw,gb_cw,gl_uw,gc_uw,gb_uw)['directed_CS_strict']
out=open(_out('table1_baselines', '_sd_gpu2_rcit.csv'),'w',newline=''); w=csv.writer(out); w.writerow(['cell','seed','cs'])
for reg in ['nonlinnongauss1','nonlinnongauss2']:
    for s in range(50):
        o=simulate_extended(reg,1.0,1000,s); X=o[0].astype(np.float64)
        B=cits_rcit(X,alpha=0.05,tau=1,K=25,max_cond_size=None,seed=s,device=os.environ.get('CITS_DEV', 'cuda:0'),null='gamma')
        p=(np.asarray(B)!=0).astype(int); np.fill_diagonal(p,0); w.writerow([f'CITS_{reg}',s,dcs_ar(p,o)])
    out.flush()
for m in ['depression']:
    for s in range(50):
        X,GT,_=simulate_glm_spiking(s,motif=m)
        B=cits_rcit(X,alpha=0.05,tau=1,K=25,max_cond_size=None,seed=s,device=os.environ.get('CITS_DEV', 'cuda:0'),null='gamma')
        p=(np.asarray(B)!=0).astype(int); np.fill_diagonal(p,0); w.writerow([f'CITS_spk_{m}',s,gdcs(p,GT)[0]])
    out.flush()
out.close()
import pandas as _pd
_d = _pd.read_csv(_out('table1_baselines', '_sd_gpu2_rcit.csv'))
print(_d.groupby('cell')['cs'].agg(['mean', 'std', 'count']).round(3))
print("gpu2 rcit done")
