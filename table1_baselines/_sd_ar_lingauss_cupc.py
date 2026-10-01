import numpy as np, sys, os, warnings, csv; warnings.filterwarnings('ignore')
os.environ.setdefault('CUPC_DIR','/home/rbiswas1/repos/cupc')
sys.path.insert(0,'.'); sys.path.insert(0,'/home/rbiswas1/repos/cits')
from sim_scm import simulate_extended
from directed_metrics import compute_directed_metrics
from gpu_cits_lag_cupc import gpu_cits_lag_cupc
def dcs(pred,out):
    _,gl_uw,gl_w,gc_uw,gc_w,gb_uw,gb_lw,gb_cw=out
    return compute_directed_metrics(pred,gl_w,gc_w,gb_lw,gb_cw,gl_uw,gc_uw,gb_uw)['directed_CS_strict']
w=csv.writer(open('_sd_ar_lingauss_cupc.csv','w',newline='')); w.writerow(['cell','seed','cs'])
for reg in ['lingauss1','lingauss2']:
    vals=[]
    for s in range(50):
        out=simulate_extended(reg,1.0,1000,s); X=out[0].astype(np.float64)
        B=gpu_cits_lag_cupc(X,alpha=0.05,tau=1)
        p=(np.asarray(B)!=0).astype(int); np.fill_diagonal(p,0); c=dcs(p,out); vals.append(c); w.writerow([f'CITScupc_{reg}',s,c])
    vals=np.array(vals); print(f"{reg}: CITS cuPC (nbr-restricted, pcorr) mean={vals.mean():.3f} sd={vals.std():.3f}",flush=True)
